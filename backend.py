import re
from datetime import datetime
# from g4f import ChatCompletion
from flask import request, Response, stream_with_context
from requests import get
from config import special_instructions


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
            model = request.json.get('model', 'mistral:latest')
            messages = build_messages(jailbreak)

            # Persist user prompt to database
            try:
                import db
                user_msg_content = request.json.get('meta', {}).get('content', {}).get('parts', [{}])[0].get('content', '')
                if user_msg_content and conversation_id:
                    db.add_message(conversation_id, 'user', user_msg_content, workspace_id=workspace_id)
            except Exception as dbe:
                print(f"[DB] Note: {dbe}")

            # Generate response
            def local_llm_stream():
                import json, requests
                from config import api_url
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": True
                }
                assistant_accumulated = []
                try:
                    with requests.post(api_url, json=payload, stream=True, timeout=(20, 300)) as r:
                        for line in r.iter_lines():
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
                                            assistant_accumulated.append(content)
                                            yield content
                                    except json.JSONDecodeError:
                                        pass
                except Exception as req_err:
                    err_msg = f"\n[LLM Service Note: {str(req_err)}]"
                    assistant_accumulated.append(err_msg)
                    yield err_msg

                # Persist completed assistant message to database
                if assistant_accumulated and conversation_id:
                    try:
                        import db
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
            import db
            for fid in payload_file_ids:
                db.link_file_to_conversation(conversation_id, fid)
        except Exception as e:
            print(f"[Backend] Warning linking files: {e}")

    # Fetch all files linked to this conversation
    active_file_ids = list(payload_file_ids)
    if conversation_id:
        try:
            import db
            conv_files = db.get_conversation_files(conversation_id)
            for cf in conv_files:
                if cf['id'] not in active_file_ids:
                    active_file_ids.append(cf['id'])
        except Exception as e:
            print(f"[Backend] Error fetching conversation files: {e}")

    # Add the existing conversation
    conversation = list(_conversation)

    # RAG File Retrieval if files attached
    if active_file_ids and user_prompt_text:
        try:
            import rag_engine
            results, max_sim = rag_engine.retrieve_relevant_chunks(active_file_ids, user_prompt_text, top_k=4)
            if results:
                file_context_msg = rag_engine.format_context_block(results)
                conversation.insert(0, {'role': 'system', 'content': file_context_msg})
        except Exception as rage:
            print(f"[RAG Engine Warning]: {rage}")

    # Add web results if enabled
    if internet_access:
        current_date = datetime.now().strftime("%Y-%m-%d")
        query = f'Current date: {current_date}. ' + user_prompt_text
        search_results = fetch_search_results(query)
        conversation.extend(search_results)

    # Add jailbreak instructions if enabled
    if jailbreak_instructions := getJailbreak(jailbreak):
        conversation.extend(jailbreak_instructions)

    # Add the prompt
    conversation.append(prompt)

    # Reduce conversation size to avoid API Token quantity error
    if len(conversation) > 5:
        conversation = conversation[-5:]

    return conversation



def fetch_search_results(query):
    """  
    Fetch search results for a given query.  

    :param query: Search query string  
    :return: List of search results  
    """
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        
        snippets = ""
        for index, result in enumerate(results):
            snippet = f'[{index + 1}] "{result.get("body", "")}" URL:{result.get("href", "")}\n'
            snippets += snippet

        response = (
            "Here are the latest web search results:\n"
            f"{snippets}\n"
            "Instructions: Use ONLY the information provided in these search results to answer the user's query. "
            "Do not hallucinate or make up dates/information that are not explicitly stated in the snippets. "
            "If the snippets do not contain the complete answer, say so. "
            "IMPORTANT: At the end of your answer, you MUST list the source URLs from the snippets as References. "
            "You MUST format these references as Markdown clickable links, like this: [Website Name](URL)."
        )

        return [{'role': 'system', 'content': response}]
    except Exception as e:
        return [{'role': 'system', 'content': 'Note: Real-time web search is currently broken because the search API is offline. Answer using your existing knowledge only.'}]


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
