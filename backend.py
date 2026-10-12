import json
import re
import time
from datetime import datetime

import requests
from flask import Response, request, stream_with_context

import config
import db
import rag_engine
from config import special_instructions


def get_failover_provider_candidates(requested_model=None):
    """
    Builds a prioritized list of provider configurations dynamically from the database.
    - Candidate 1 (Primary): requested_model or db.get_system_config('default_model') or first active provider
    - Candidate 2 (Fallback): db.get_system_config('fallback_model') from database (must be active)
    Returns: list of provider dictionaries from database (zero hardcoding).
    """
    providers = db.get_all_llm_providers() or []
    if not providers:
        return []

    default_model = db.get_system_config('default_model') or ''
    fallback_model = db.get_system_config('fallback_model') or ''

    primary_target = requested_model or default_model
    candidates = []
    seen_ids = set()

    # 1. Primary candidate: match model_name with requested/default from active providers
    primary_provider = None
    if primary_target:
        primary_provider = next((p for p in providers if p.get('model_name') == primary_target and p.get('is_active')), None)
    # If not specifically matched, use first available active provider from database
    if not primary_provider:
        primary_provider = next((p for p in providers if p.get('is_active')), None)

    if primary_provider:
        candidates.append(primary_provider)
        seen_ids.add(primary_provider.get('id'))

    # 2. Designated Single Fallback candidate: match fallback_model from database and must be active
    if fallback_model:
        fallback_provider = next(
            (p for p in providers if p.get('model_name') == fallback_model and p.get('is_active') and p.get('id') not in seen_ids),
            None
        )
        if fallback_provider:
            candidates.append(fallback_provider)
            seen_ids.add(fallback_provider.get('id'))

    # Strictly limit to Primary + 1 Fallback model only (no endless looping across other models)
    return candidates


class Backend_Api:
    def __init__(self, bp, config: dict) -> None:
        """
        Initialize the Backend_Api class.
        :param app: Flask application instance
        :param config: Configuration dictionary
        """
        self.bp = bp
        self.routes = {
            '/backend-api/v2/conversation': {
                'function': self._conversation,
                'methods': ['POST']
            }
        }

    def _conversation(self):
        """  
        Handles the conversation route.  

        :return: Response object containing the generated conversation stream  
        """
        conversation_id = request.json.get('conversation_id', '')
        workspace_id = request.json.get('workspace_id')

        try:
            jailbreak = request.json.get('jailbreak', 'default')
            default_model = db.get_system_config('default_model') or ''
            if not default_model:
                first_active = next((p for p in db.get_all_llm_providers() if p.get('is_active')), None)
                default_model = first_active.get('model_name') if first_active else ''
            model = request.json.get('model') or default_model
            messages = build_messages(jailbreak)

            # Persist user prompt to database
            try:
                user_msg_content = request.json.get('meta', {}).get('content', {}).get('parts', [{}])[0].get('content', '')
                if user_msg_content and conversation_id:
                    db.add_message(conversation_id, 'user', user_msg_content, workspace_id=workspace_id)
            except Exception as dbe:
                print(f"[DB] Note: {dbe}")

            # Generate response with failover candidates
            def local_llm_stream():
                candidates = get_failover_provider_candidates(model)

                if not candidates:
                    err_msg = f"⚠️ [System Alert]: All AI models are currently turned OFF (inactive) in System Configuration. Please enable at least one LLM Provider in Admin Settings to resume chat."
                    yield err_msg
                    if conversation_id:
                        try:
                            db.add_message(conversation_id, 'assistant', err_msg, workspace_id=workspace_id)
                        except Exception as dbe:
                            print(f"[DB] Note: {dbe}")
                    return

                success = False
                assistant_accumulated = []

                # Execute Failover Loop across candidates
                for idx, provider in enumerate(candidates):
                    api_url = provider.get('endpoint')
                    if not api_url:
                        continue
                    actual_model = provider.get('model_name') or model
                    timeout_val = provider.get('timeout') or 600

                    # Print clearly to terminal which model is being attempted
                    status_prefix = "PRIMARY" if idx == 0 else "FALLBACK"
                    print(f"\n[LLM Request] >>> Trying {status_prefix} Model [{idx+1}/{len(candidates)}]: '{actual_model}' ({provider.get('name')}) | Endpoint: {api_url}")

                    # Stream subtle markdown notice if failover happened
                    if idx > 0:
                        primary_label = candidates[0].get('name') or candidates[0].get('model_name')
                        fallback_label = provider.get('name') or actual_model
                        print(f"[LLM Failover] >>> Primary model ({primary_label}) failed. Switched to Fallback: '{fallback_label}'")
                        notice = f"> ℹ️ *Note: Primary model ({primary_label}) did not respond. Continuing with fallback model ({fallback_label})...*\n\n"
                        assistant_accumulated.append(notice)
                        yield notice

                    payload = {
                        "model": actual_model,
                        "messages": messages,
                        "stream": True
                    }

                    try:
                        # Connection timeout bounded by provider's configured timeout, read timeout from database
                        conn_timeout = min(20, timeout_val)
                        with requests.post(api_url, json=payload, stream=True, timeout=(conn_timeout, timeout_val)) as r:
                            if r.status_code >= 400:
                                print(f"[LLM Failover] >>> Model '{actual_model}' returned HTTP {r.status_code}. Trying next provider...")
                                continue

                            req_start = time.time()
                            stream_received = False
                            for line in r.iter_lines():
                                if not stream_received and (time.time() - req_start > timeout_val):
                                    raise requests.exceptions.Timeout(f"Initial token generation exceeded timeout ({timeout_val}s)")
                                if line:
                                    line = line.decode('utf-8')
                                    if line.startswith('data: '):
                                        data = line[6:]
                                        if data == '[DONE]':
                                            break
                                        try:
                                            chunk = json.loads(data)
                                            content = chunk['choices'][0].get('delta', {}).get('content', '')
                                            if content:
                                                stream_received = True
                                                assistant_accumulated.append(content)
                                                yield content
                                        except json.JSONDecodeError:
                                            pass

                            if stream_received:
                                print(f"[LLM Response] <<< Successfully generated response using model: '{actual_model}' ({provider.get('name')})\n")
                                success = True
                                break  # Successfully streamed from candidate provider

                    except Exception as req_err:
                        print(f"[LLM Failover] >>> Model '{actual_model}' failed ({req_err}). Trying next candidate...")
                        continue

                # If all candidate providers failed
                if not success:
                    err_msg = "⚠️ [LLM Service Error]: All configured AI models are currently unreachable. Please verify network connectivity or check provider endpoints."
                    assistant_accumulated.append(err_msg)
                    yield err_msg

                # Persist completed assistant message to database
                if assistant_accumulated and conversation_id:
                    try:
                        full_reply = "".join(assistant_accumulated)
                        db.add_message(conversation_id, 'assistant', full_reply, workspace_id=workspace_id)
                    except Exception as dbe:
                        print(f"[DB] Note: {dbe}")

            response = local_llm_stream()

            return Response(stream_with_context(generate_stream(response, jailbreak)), mimetype='text/event-stream')

        except Exception as e:
            print(e)
            print(e.__traceback__.tb_next)

            return {
                '_action': '_ask',
                'success': False,
                "error": f"an error occurred {str(e)}"
            }, 400


def build_messages(jailbreak):
    """  
    Build the messages for the conversation.  

    :param jailbreak: Jailbreak instruction string  
    :return: List of messages for the conversation  
    """
    _conversation = request.json['meta']['content']['conversation']
    internet_access = request.json['meta']['content']['internet_access']
    prompt = request.json['meta']['content']['parts'][0]
    user_prompt_text = prompt.get("content", "")

    conversation_id = request.json.get('conversation_id', '')
    payload_file_ids = request.json.get('meta', {}).get('file_ids', [])

    # Link payload file_ids to conversation if provided
    if conversation_id and payload_file_ids:
        try:
            for fid in payload_file_ids:
                db.link_file_to_conversation(conversation_id, fid)
        except Exception as e:
            print(f"[Backend] Warning linking files: {e}")

    # Fetch all files linked to this conversation
    active_file_ids = list(payload_file_ids)
    if conversation_id:
        try:
            conv_files = db.get_conversation_files(conversation_id)
            for cf in conv_files:
                if cf['id'] not in active_file_ids:
                    active_file_ids.append(cf['id'])
        except Exception as e:
            print(f"[Backend] Error fetching conversation files: {e}")

    # Retrieve RAG file context if any files are active
    file_context_msg = ""
    if active_file_ids:
        try:
            query_for_rag = user_prompt_text.strip() if user_prompt_text else "Summarize and analyze the attached document."
            results = rag_engine.retrieve_file_context(active_file_ids, query_for_rag)
            if results:
                file_context_msg = rag_engine.format_context_block(results)
        except Exception as rage:
            print(f"[RAG Engine Warning]: {rage}")

    # Slice past conversation history first (keep up to last 6 messages)
    history = list(_conversation)
    if len(history) > 6:
        history = history[-6:]

    conversation = []

    # Add jailbreak instructions if enabled
    if jailbreak_instructions := getJailbreak(jailbreak):
        conversation.extend(jailbreak_instructions)

    # Add past history
    conversation.extend(history)

    # Add web results if enabled
    if internet_access:
        current_date = datetime.now().strftime("%Y-%m-%d")
        query = f'Current date: {current_date}. ' + user_prompt_text
        search_results = fetch_search_results(query)
        conversation.extend(search_results)

    # Chart instruction prompt to ensure LLM generates structured ```chart JSON ONLY when requested
    chart_keywords = ['chart', 'pie', 'bar', 'graph', 'plot', 'visual', 'diagram', 'barchart', 'piechart', 'চার্ট', 'গ্রাফ', 'পাই চার্ট', 'বার চার্ট']
    user_wants_chart = any(kw in user_prompt_text.lower() for kw in chart_keywords)

    chart_prompt_guidelines = (
        "\n\n[USER EXPLICITLY REQUESTED A CHART]:\n"
        "The user wants a data chart/visualization (Pie chart, Bar chart, Line chart).\n"
        "You MUST output a valid ```chart codeblock containing JSON data extracted from the document/input so the frontend can render an interactive Pie/Bar chart.\n"
        "Required JSON Format:\n"
        "```chart\n"
        "{\n"
        '  "type": "pie",\n'
        '  "title": "Clear Title Describing The Chart Data",\n'
        '  "data": [\n'
        '    {"name": "Category Name 1", "value": 150},\n'
        '    {"name": "Category Name 2", "value": 230}\n'
        '  ]\n'
        "}\n"
        "```\n"
        "Use type 'pie' for proportional breakdown or 'bar' for comparison. Also provide a helpful textual explanation."
    )

    # Build final user prompt with file context embedded directly
    final_prompt_content = user_prompt_text
    if file_context_msg:
        if user_prompt_text and user_prompt_text.strip() != "Summarize and analyze the attached document.":
            final_prompt_content = f"{file_context_msg}\n\nUser Question/Instruction:\n{user_prompt_text}"
        else:
            final_prompt_content = f"{file_context_msg}\n\nPlease analyze and summarize the attached document."

    # If user wants a chart, append strict mandatory instruction at the very end of prompt asking to start with ```chart
    if user_wants_chart:
        chart_mandatory_rule = (
            "\n\n[MANDATORY SYSTEM DIRECTIVE FOR VISUAL CHART]:\n"
            "The user explicitly requested a visual Pie chart / Bar chart.\n"
            "YOU MUST START YOUR RESPONSE DIRECTLY WITH A ```chart JSON CODE BLOCK AS THE VERY FIRST ITEM IN YOUR OUTPUT.\n\n"
            "Required Format:\n"
            "```chart\n"
            "{\n"
            '  "type": "pie",\n'
            '  "title": "Category Sales / Market Share Breakdown",\n'
            '  "data": [\n'
            '    {"name": "Gold", "value": 22956},\n'
            '    {"name": "Silver", "value": 3308}\n'
            '  ]\n'
            "}\n"
            "```\n"
            "Calculate actual numerical totals/percentages from the file. DO NOT write text before the ```chart block. START WITH ```chart IMMEDIATELY."
        )
        final_prompt_content = final_prompt_content + chart_mandatory_rule

    conversation.append({'role': 'user', 'content': final_prompt_content})

    # Add system message at beginning of conversation array if user requested chart
    if user_wants_chart:
        conversation.insert(0, {
            'role': 'system',
            'content': 'You are an expert AI data assistant. Whenever the user requests a chart, pie chart, or bar chart, you MUST output a ```chart JSON code block as the very first output.'
        })

    return conversation



def fetch_search_results(query):
    """  
    Fetch search results for a given query.  

    :param query: Search query string  
    :return: List of search results  
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        # Fallback without date prefix if initial query returned nothing
        if not results:
            clean_q = re.sub(r'^Current date: \d{4}-\d{2}-\d{2}\.\s*', '', query)
            with DDGS() as ddgs:
                results = list(ddgs.text(clean_q, max_results=5))

        if not results:
            return []

        snippets = ""
        for index, result in enumerate(results):
            body = result.get("body", "")
            href = result.get("href", "")
            title = result.get("title", "")
            if body or href:
                snippets += f'[{index + 1}] "{body}" Source: {title} URL:{href}\n'

        if not snippets.strip():
            return []

        response = (
            "Here are the latest web search results for context:\n"
            f"{snippets}\n"
            "Instructions: Use the above search results along with your knowledge to give an up-to-date and accurate answer. "
            "Cite the source URLs at the end as clickable Markdown links, for example: [Website Name](URL)."
        )

        return [{'role': 'system', 'content': response}]
    except Exception as e:
        print(f"[Web Search Warning] {e}")
        return []


def generate_stream(response, jailbreak):
    """
    Generate the conversation stream.

    :param response: Response object from ChatCompletion.create
    :param jailbreak: Jailbreak instruction string
    :return: Generator object yielding messages in the conversation
    """
    if getJailbreak(jailbreak):
        response_jailbreak = ''
        jailbroken_checked = False
        for message in response:
            response_jailbreak += message
            if jailbroken_checked:
                yield message
            else:
                if response_jailbroken_success(response_jailbreak):
                    jailbroken_checked = True
                if response_jailbroken_failed(response_jailbreak):
                    yield response_jailbreak
                    jailbroken_checked = True
    else:
        yield from response


def response_jailbroken_success(response: str) -> bool:
    """Check if the response has been jailbroken.

    :param response: Response string
    :return: Boolean indicating if the response has been jailbroken
    """
    act_match = re.search(r'ACT:', response, flags=re.DOTALL)
    return bool(act_match)


def response_jailbroken_failed(response):
    """
    Check if the response has not been jailbroken.

    :param response: Response string
    :return: Boolean indicating if the response has not been jailbroken
    """
    return False if len(response) < 4 else not (response.startswith("GPT:") or response.startswith("ACT:"))


def getJailbreak(jailbreak):
    """  
    Check if jailbreak instructions are provided.  

    :param jailbreak: Jailbreak instruction string  
    :return: Jailbreak instructions if provided, otherwise None  
    """
    if jailbreak != "default":
        special_instructions[jailbreak][0]['content'] += special_instructions['two_responses_instruction']
        if jailbreak in special_instructions:
            special_instructions[jailbreak]
            return special_instructions[jailbreak]
        else:
            return None
    else:
        return None
