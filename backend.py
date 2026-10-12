import json
import random
import re
import time
from datetime import datetime

import requests
from flask import Response, request, stream_with_context
from requests import get

import config
from config import special_instructions
import db
import guardrails
import rag_engine


def get_dynamic_status(stage: str, prompt_text: str = "", context: dict = None) -> str:
    """
    Returns a dynamic, context-aware, and varied status text for each actual stage.
    """
    text_lower = (prompt_text or "").lower().strip()

    is_code = any(k in text_lower for k in ['code', 'python', 'javascript', 'typescript', 'ts', 'react', 'function', 'class', 'bug', 'error', 'script', 'java', 'c++', 'html', 'css', 'sql'])
    try:
        import guardrails
        chart_eval = guardrails.evaluate_chart_intent(prompt_text)
        is_chart = chart_eval["decision"] in (guardrails.ChartIntentDecision.GENERATE, guardrails.ChartIntentDecision.CLARIFY)
    except Exception:
        is_chart = any(k in text_lower for k in ['chart', 'pie', 'bar', 'graph', 'plot', 'visual', 'diagram', 'barchart', 'piechart'])
    is_summary = any(k in text_lower for k in ['summary', 'summarize', 'overview', 'brief', 'gist'])
    is_math = any(k in text_lower for k in ['calculate', 'math', 'solve', 'equation', 'sum', 'percentage', '%', 'formula'])

    if stage == 'analyzing_query':
        if is_code:
            options = [
                "Analyzing code context...",
                "Reviewing technical requirements...",
                "Inspecting code structure...",
                "Evaluating technical query...",
                "Parsing programming logic..."
            ]
        elif is_chart:
            options = [
                "Analyzing data visualization request...",
                "Structuring chart parameters...",
                "Evaluating data metrics...",
                "Interpreting chart requirements...",
                "Reviewing chart specifications..."
            ]
        elif is_summary:
            options = [
                "Analyzing summary request...",
                "Reviewing document scope...",
                "Structuring key highlights...",
                "Evaluating summary objectives...",
                "Reading key points..."
            ]
        elif is_math:
            options = [
                "Analyzing mathematical logic...",
                "Parsing calculation steps...",
                "Reviewing problem formula...",
                "Evaluating numerical query...",
                "Computing mathematical terms..."
            ]
        else:
            options = [
                "Analyzing your question...",
                "Understanding your intent...",
                "Interpreting your question...",
                "Reviewing your prompt...",
                "Processing your inquiry...",
                "Examining your question...",
                "Exploring your request...",
                "Evaluating prompt context..."
            ]
        return random.choice(options)

    elif stage == 'retrieving_context':
        if is_summary:
            options = [
                "Scanning full document content...",
                "Extracting document overview...",
                "Reviewing document sections...",
                "Reading file contents...",
                "Gathering document overview..."
            ]
        elif is_chart:
            options = [
                "Extracting numerical data from document...",
                "Retrieving chart data points...",
                "Parsing tabular document data...",
                "Scanning file metrics...",
                "Reading quantitative data..."
            ]
        else:
            options = [
                "Retrieving relevant context...",
                "Searching attached documents...",
                "Scanning document passages...",
                "Extracting relevant information...",
                "Cross-referencing file data...",
                "Querying document knowledge base...",
                "Gathering relevant citations..."
            ]
        return random.choice(options)

    elif stage == 'searching_web':
        options = [
            "Searching for relevant information...",
            "Searching the web for latest info...",
            "Browsing current online sources...",
            "Gathering real-time web data...",
            "Fetching up-to-date web results...",
            "Scanning live online sources...",
            "Querying real-time web knowledge..."
        ]
        return random.choice(options)

    elif stage == 'preparing_response':
        if is_code:
            options = [
                "Drafting code solution...",
                "Synthesizing technical response...",
                "Formulating optimized solution...",
                "Preparing code explanation...",
                "Generating technical answer..."
            ]
        elif is_chart:
            options = [
                "Generating chart visualization...",
                "Structuring visual data response...",
                "Finalizing chart breakdown...",
                "Preparing visual analytics...",
                "Formulating chart dataset..."
            ]
        else:
            options = [
                "Preparing your response...",
                "Drafting your answer...",
                "Synthesizing insights...",
                "Crafting your response...",
                "Formulating final answer...",
                "Composing the response...",
                "Structuring comprehensive reply..."
            ]
        return random.choice(options)

    return "Thinking..."


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
        req_json = request.get_json(silent=True) or {}
        conversation_id = req_json.get('conversation_id', '')
        workspace_id = req_json.get('workspace_id')
        default_model = db.get_system_config('default_model') or ''
        if not default_model:
            first_active = next((p for p in db.get_all_llm_providers() if p.get('is_active')), None)
            default_model = first_active.get('model_name') if first_active else ''
        model = req_json.get('model') or default_model or 'mistral:latest'

        def pipeline_stream():
            meta = req_json.get('meta', {})
            content_data = meta.get('content', {})
            _conv = content_data.get('conversation', [])
            internet_access = content_data.get('internet_access', False)
            parts = content_data.get('parts', [{}])
            user_prompt_text = parts[0].get("content", "") if parts else ""
            payload_file_ids = meta.get('file_ids', [])

            # Stage 1: Analyzing query (Dynamic)
            status_1 = get_dynamic_status('analyzing_query', user_prompt_text)
            yield f"data: {json.dumps({'type': 'status', 'stage': 'analyzing_query', 'text': status_1})}\n\n"

            # Persist user prompt to database
            if user_prompt_text and conversation_id:
                try:
                    db.add_message(conversation_id, 'user', user_prompt_text, workspace_id=workspace_id)
                except Exception as dbe:
                    print(f"[DB] Note: {dbe}")

            candidates = get_failover_provider_candidates(model)
            primary_candidate = candidates[0] if candidates else None
            actual_model = primary_candidate.get('model_name', model) if primary_candidate else model

            # Standard Mode Guardrails - Deterministic Identity Handler (Fast Path)
            if guardrails.is_standard_mode(jailbreak):
                identity_match = guardrails.handle_identity_query(user_prompt_text, actual_model)
                if identity_match:
                    status_prep = get_dynamic_status('preparing_response', user_prompt_text)
                    yield f"data: {json.dumps({'type': 'status', 'stage': 'preparing_response', 'text': status_prep})}\n\n"
                    resp_content = identity_match['response']
                    yield f"data: {json.dumps({'type': 'content', 'delta': resp_content})}\n\n"
                    if conversation_id:
                        try:
                            db.add_message(conversation_id, 'assistant', resp_content, workspace_id=workspace_id)
                        except Exception as dbe:
                            print(f"[DB] Note: {dbe}")
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Standard Mode Guardrails - Phase 2 Core Content Safety Evaluation
                safety_eval = guardrails.evaluate_content_safety(user_prompt_text)
                if safety_eval['decision'] == guardrails.ContentDecision.REFUSE_OR_REDIRECT:
                    status_prep = get_dynamic_status('preparing_response', user_prompt_text)
                    yield f"data: {json.dumps({'type': 'status', 'stage': 'preparing_response', 'text': status_prep})}\n\n"
                    refusal_resp = safety_eval['response']
                    yield f"data: {json.dumps({'type': 'content', 'delta': refusal_resp})}\n\n"
                    if conversation_id:
                        try:
                            db.add_message(conversation_id, 'assistant', refusal_resp, workspace_id=workspace_id)
                        except Exception as dbe:
                            print(f"[DB] Note: {dbe}")
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Standard Mode Guardrails - Phase 1 Explicit Chart Intent Clarification Fast-Path
                chart_intent_eval = guardrails.evaluate_chart_intent(user_prompt_text)
                if chart_intent_eval['decision'] == guardrails.ChartIntentDecision.CLARIFY:
                    status_prep = get_dynamic_status('preparing_response', user_prompt_text)
                    yield f"data: {json.dumps({'type': 'status', 'stage': 'preparing_response', 'text': status_prep})}\n\n"
                    clarify_resp = chart_intent_eval['response']
                    yield f"data: {json.dumps({'type': 'content', 'delta': clarify_resp})}\n\n"
                    if conversation_id:
                        try:
                            db.add_message(conversation_id, 'assistant', clarify_resp, workspace_id=workspace_id)
                        except Exception as dbe:
                            print(f"[DB] Note: {dbe}")
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            # Link payload file_ids to conversation if provided
            if conversation_id and payload_file_ids:
                try:
                    for fid in payload_file_ids:
                        db.link_file_to_conversation(conversation_id, fid)
                except Exception as e:
                    print(f"[Backend] Warning linking files: {e}")

            conv_files = []
            if conversation_id:
                try:
                    conv_files = db.get_conversation_files(conversation_id) or []
                except Exception as e:
                    print(f"[Backend] Error fetching conversation files: {e}")
                    conv_files = []

            # Phase 2: Strict File Context Isolation
            is_chart = False
            if guardrails.is_standard_mode(jailbreak):
                chart_eval_peek = guardrails.evaluate_chart_intent(user_prompt_text)
                is_chart = (chart_eval_peek['decision'] == guardrails.ChartIntentDecision.GENERATE)
                file_context_eval = guardrails.resolve_file_context(
                    user_prompt=user_prompt_text,
                    payload_file_ids=payload_file_ids,
                    conversation_files=conv_files,
                    is_chart_request=is_chart
                )
                if file_context_eval['decision'] == guardrails.FileContextDecision.AMBIGUOUS:
                    status_prep = get_dynamic_status('preparing_response', user_prompt_text)
                    yield f"data: {json.dumps({'type': 'status', 'stage': 'preparing_response', 'text': status_prep})}\n\n"
                    clarify_resp = file_context_eval['response']
                    yield f"data: {json.dumps({'type': 'content', 'delta': clarify_resp})}\n\n"
                    if conversation_id:
                        try:
                            db.add_message(conversation_id, 'assistant', clarify_resp, workspace_id=workspace_id)
                        except Exception as dbe:
                            print(f"[DB] Note: {dbe}")
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                active_file_ids = file_context_eval['active_file_ids']
            else:
                # Non-standard modes (DAN / EvilBOT): preserve legacy file resolution
                if payload_file_ids:
                    active_file_ids = list(payload_file_ids)
                elif conv_files:
                    active_file_ids = [cf['id'] for cf in conv_files]
                else:
                    active_file_ids = []

            # Stage 2 (Conditional): If active files exist, retrieve RAG file context
            file_context_msg = ""
            if active_file_ids:
                status_2 = get_dynamic_status('retrieving_context', user_prompt_text)
                yield f"data: {json.dumps({'type': 'status', 'stage': 'retrieving_context', 'text': status_2})}\n\n"
                try:
                    import rag_engine
                    query_for_rag = user_prompt_text.strip() if user_prompt_text else "Summarize and analyze the attached document."
                    results = rag_engine.retrieve_file_context(active_file_ids, query_for_rag)
                    if results:
                        file_context_msg = rag_engine.format_context_block(results)
                except Exception as rage:
                    print(f"[RAG Engine Warning]: {rage}")

            # Missing or Insufficient Data Fast-Path for Chart Requests on Selected Files
            if guardrails.is_standard_mode(jailbreak) and is_chart and active_file_ids and not file_context_msg:
                matched_names = [f.get('original_name', f.get('id', 'file')) for f in file_context_eval.get('matched_files', [])]
                name_str = ", ".join(f"'{n}'" for n in matched_names) if matched_names else "the selected file"
                no_data_resp = (
                    f"The selected file ({name_str}) does not contain readable data or content to generate a chart. "
                    "Please ensure the file has valid tabular or numerical data."
                )
                status_prep = get_dynamic_status('preparing_response', user_prompt_text)
                yield f"data: {json.dumps({'type': 'status', 'stage': 'preparing_response', 'text': status_prep})}\n\n"
                yield f"data: {json.dumps({'type': 'content', 'delta': no_data_resp})}\n\n"
                if conversation_id:
                    try:
                        db.add_message(conversation_id, 'assistant', no_data_resp, workspace_id=workspace_id)
                    except Exception as dbe:
                        print(f"[DB] Note: {dbe}")
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Slice past conversation history
            # If user attached new file(s) in this message, start fresh without carrying over older document topics/charts
            if payload_file_ids:
                history = []
            else:
                history = list(_conv)
                if len(history) > 6:
                    history = history[-6:]

            if guardrails.is_standard_mode(jailbreak):
                history = guardrails.sanitize_conversation_history_for_charts(
                    history=history,
                    user_prompt=user_prompt_text,
                    is_chart_request=is_chart
                )

            conversation = []

            # Add jailbreak instructions if enabled
            if jailbreak_instructions := getJailbreak(jailbreak):
                conversation.extend(jailbreak_instructions)

            # Add past history
            conversation.extend(history)

            # Stage 3 (Conditional): If internet access is enabled, fetch web search results
            if internet_access:
                status_3 = get_dynamic_status('searching_web', user_prompt_text)
                yield f"data: {json.dumps({'type': 'status', 'stage': 'searching_web', 'text': status_3})}\n\n"
                current_date = datetime.now().strftime("%Y-%m-%d")
                query = f'Current date: {current_date}. ' + user_prompt_text
                search_results = fetch_search_results(query)
                conversation.extend(search_results)

            # Build final prompt content with file context embedded directly
            final_prompt_content = user_prompt_text
            if file_context_msg:
                if user_prompt_text and user_prompt_text.strip() != "Summarize and analyze the attached document.":
                    final_prompt_content = f"{file_context_msg}\n\nUser Question/Instruction:\n{user_prompt_text}"
                else:
                    final_prompt_content = f"{file_context_msg}\n\nPlease analyze and summarize the attached document."

            if guardrails.is_standard_mode(jailbreak):
                # Phase 1: Explicit Chart Intent Guardrails for Standard Mode
                chart_intent = guardrails.evaluate_chart_intent(user_prompt_text)
                if chart_intent['decision'] == guardrails.ChartIntentDecision.GENERATE:
                    chart_directive = guardrails.get_chart_directive(chart_intent['chart_type'])
                    final_prompt_content = final_prompt_content + chart_directive
                else:
                    # Consecutive-turn independence & false-positive prevention:
                    # Explicitly instruct the model NOT to generate any chart block for this non-chart request,
                    # preventing history bleeding from earlier conversation turns.
                    no_chart_instruction = (
                        "\n\n[NO CHART REQUESTED]:\n"
                        "The user did NOT request a chart in this message. Do NOT generate or include any ```chart code block in your response, "
                        "even if previous messages in this conversation contain charts. Respond strictly with clear text, explanations, or standard markdown formatting."
                    )
                    final_prompt_content = final_prompt_content + no_chart_instruction

                conversation.append({'role': 'user', 'content': final_prompt_content})

                context_guidance = safety_eval.get('guidance') if 'safety_eval' in locals() else None
                std_sys_prompt = guardrails.get_standard_system_prompt(actual_model, contextual_guidance=context_guidance)
                if chart_intent['decision'] == guardrails.ChartIntentDecision.GENERATE:
                    ctype_str = chart_intent['chart_type']
                    std_sys_prompt += (
                        f"\n\n[MANDATORY INSTRUCTION FOR {ctype_str.upper()} CHART]:\n"
                        f"The user explicitly requested a {ctype_str.capitalize()} Chart visualization. You MUST start your response directly with a ```chart JSON code block of type \"{ctype_str}\". "
                        "All data, titles, labels, and numbers MUST be strictly extracted from the provided document or current context. "
                        "NEVER output unrelated dummy categories like Gold, Silver, or Sales."
                    )
                conversation.insert(0, {'role': 'system', 'content': std_sys_prompt})
            else:
                # Non-standard modes (DAN Mode 'gpt-dan-11.0', EvilBOT Mode 'gpt-evil'):
                # Preserve existing legacy behavior completely untouched.
                chart_keywords = ['chart', 'pie', 'bar', 'graph', 'plot', 'visual', 'diagram', 'barchart', 'piechart']
                user_wants_chart = any(kw in user_prompt_text.lower() for kw in chart_keywords)

                if user_wants_chart:
                    chart_mandatory_rule = (
                        "\n\n[MANDATORY INSTRUCTION FOR VISUAL CHART]:\n"
                        "The user explicitly requested a visual chart / pie chart / bar chart.\n"
                        "CRITICAL RULES:\n"
                        "1. REAL DATA ONLY: You MUST extract real entity names/categories and numerical values STRICTLY from the currently attached document content or prompt above.\n"
                        "   - DO NOT hallucinate, invent placeholder data, or reuse topics/entities from earlier messages.\n"
                        "   - The chart categories ('name') and values ('value') must strictly be the actual items and quantities found in this current document.\n"
                        "2. START WITH ```chart: You MUST start your response immediately with the ```chart JSON code block as the very first characters of your output.\n"
                        "3. JSON Format:\n"
                        "```chart\n"
                        "{\n"
                        '  "type": "pie",\n'
                        '  "title": "<Descriptive Title Reflecting The Document Data>",\n'
                        '  "data": [\n'
                        '    {"name": "<Actual Item Name From Document>", "value": <Actual Number>}\n'
                        '  ]\n'
                        "}\n"
                        "```\n"
                        "4. After the ```chart block, provide a clear text breakdown and explanation of the data from the document.\n"
                        "START WITH ```chart DIRECTLY."
                    )
                    final_prompt_content = final_prompt_content + chart_mandatory_rule

                conversation.append({'role': 'user', 'content': final_prompt_content})

                if user_wants_chart:
                    conversation.insert(0, {
                        'role': 'system',
                        'content': (
                            'You are an expert AI data assistant. Whenever the user requests a chart or visualization, you MUST start your response directly with a ```chart JSON code block. '
                            'All data, titles, labels, and numbers MUST be strictly extracted from the provided document or conversation context. '
                            'NEVER output unrelated dummy categories like Gold, Silver, or Sales.'
                        )
                    })

            # Stage 4: Preparing response (connecting to LLM) (Dynamic)
            status_4 = get_dynamic_status('preparing_response', user_prompt_text)
            yield f"data: {json.dumps({'type': 'status', 'stage': 'preparing_response', 'text': status_4})}\n\n"

            if not candidates:
                err_msg = "⚠️ [System Alert]: All AI models are currently turned OFF (inactive) in System Configuration. Please enable at least one LLM Provider in Admin Settings to resume chat."
                yield f"data: {json.dumps({'type': 'error', 'error': err_msg})}\n\n"
                if conversation_id:
                    try:
                        db.add_message(conversation_id, 'assistant', err_msg, workspace_id=workspace_id)
                    except Exception as dbe:
                        print(f"[DB] Note: {dbe}")
                return

            success = False
            assistant_accumulated = []

            for idx, provider in enumerate(candidates):
                api_url = provider.get('endpoint')
                if not api_url:
                    continue
                actual_model = provider.get('model_name') or model
                timeout_val = provider.get('timeout') or 600

                status_prefix = "PRIMARY" if idx == 0 else "FALLBACK"
                print(f"\n[LLM Request] >>> Trying {status_prefix} Model [{idx+1}/{len(candidates)}]: '{actual_model}' ({provider.get('name')}) | Endpoint: {api_url}")

                if idx > 0:
                    primary_label = candidates[0].get('name') or candidates[0].get('model_name')
                    fallback_label = provider.get('name') or actual_model
                    print(f"[LLM Failover] >>> Primary model ({primary_label}) failed. Switched to Fallback: '{fallback_label}'")
                    notice = f"> ℹ️ *Note: Primary model ({primary_label}) did not respond. Continuing with fallback model ({fallback_label})...*\n\n"
                    assistant_accumulated.append(notice)
                    yield f"data: {json.dumps({'type': 'content', 'delta': notice})}\n\n"

                payload = {
                    "model": actual_model,
                    "messages": conversation,
                    "stream": True
                }

                try:
                    conn_timeout = min(20, timeout_val)
                    with requests.post(api_url, json=payload, stream=True, timeout=(conn_timeout, timeout_val)) as r:
                        if r.status_code >= 400:
                            print(f"[LLM Failover] >>> Model '{actual_model}' returned HTTP {r.status_code}. Trying next provider...")
                            continue

                        req_start = time.time()
                        stream_received = False
                        is_std_chart = guardrails.is_standard_mode(jailbreak) and is_chart
                        chart_buffer = []
                        chart_validated = False

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
                                        if not content:
                                            continue
                                        stream_received = True

                                        if is_std_chart and not chart_validated:
                                            chart_buffer.append(content)
                                            buf_str = "".join(chart_buffer)

                                            first_non_ws = buf_str.lstrip()
                                            if len(first_non_ws) >= 8 and not first_non_ws.startswith('```'):
                                                # Model did not start with ```chart, flush buffer
                                                chart_validated = True
                                                for c in chart_buffer:
                                                    assistant_accumulated.append(c)
                                                    yield f"data: {json.dumps({'type': 'content', 'delta': c})}\n\n"
                                                chart_buffer = []
                                            elif '```' in buf_str[3:]:
                                                # Chart code block is complete, validate it
                                                chart_validated = True
                                                expected_c_type = chart_intent.get('chart_type') if 'chart_intent' in locals() else None
                                                is_v, clean_text, payload_res, v_err = guardrails.validate_and_sanitize_chart_response(
                                                    buf_str,
                                                    expected_chart_type=expected_c_type,
                                                    file_context=file_context_msg
                                                )
                                                assistant_accumulated.append(clean_text)
                                                yield f"data: {json.dumps({'type': 'content', 'delta': clean_text})}\n\n"
                                                chart_buffer = []
                                        else:
                                            assistant_accumulated.append(content)
                                            yield f"data: {json.dumps({'type': 'content', 'delta': content})}\n\n"
                                    except json.JSONDecodeError:
                                        pass

                        # Flush remaining buffer if stream ended
                        if chart_buffer:
                            buf_str = "".join(chart_buffer)
                            if is_std_chart and not chart_validated:
                                expected_c_type = chart_intent.get('chart_type') if 'chart_intent' in locals() else None
                                is_v, clean_text, payload_res, v_err = guardrails.validate_and_sanitize_chart_response(
                                    buf_str,
                                    expected_chart_type=expected_c_type,
                                    file_context=file_context_msg
                                )
                                assistant_accumulated.append(clean_text)
                                yield f"data: {json.dumps({'type': 'content', 'delta': clean_text})}\n\n"
                            else:
                                for c in chart_buffer:
                                    assistant_accumulated.append(c)
                                    yield f"data: {json.dumps({'type': 'content', 'delta': c})}\n\n"

                        if stream_received:
                            print(f"[LLM Response] <<< Successfully generated response using model: '{actual_model}' ({provider.get('name')})\n")
                            success = True
                            break

                except Exception as req_err:
                    print(f"[LLM Failover] >>> Model '{actual_model}' failed ({req_err}). Trying next candidate...")
                    continue

            if not success:
                err_msg = "⚠️ [LLM Service Error]: All configured AI models are currently unreachable. Please verify network connectivity or check provider endpoints."
                assistant_accumulated.append(err_msg)
                yield f"data: {json.dumps({'type': 'error', 'error': err_msg})}\n\n"

            # Persist completed assistant message to database
            if assistant_accumulated and conversation_id:
                try:
                    full_reply = "".join(assistant_accumulated)
                    if guardrails.is_standard_mode(jailbreak) and is_chart:
                        expected_c_type = chart_intent.get('chart_type') if 'chart_intent' in locals() else None
                        _, full_reply, _, _ = guardrails.validate_and_sanitize_chart_response(
                            full_reply,
                            expected_chart_type=expected_c_type,
                            file_context=file_context_msg
                        )
                    db.add_message(conversation_id, 'assistant', full_reply, workspace_id=workspace_id)
                except Exception as dbe:
                    print(f"[DB] Note: {dbe}")

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(pipeline_stream()),
            mimetype='text/event-stream',
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive"
            }
        )


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

    conv_files = []
    if conversation_id:
        try:
            conv_files = db.get_conversation_files(conversation_id) or []
        except Exception as e:
            print(f"[Backend] Error fetching conversation files: {e}")
            conv_files = []

    # Phase 2: Strict File Context Isolation
    is_chart = False
    if guardrails.is_standard_mode(jailbreak):
        chart_eval_peek = guardrails.evaluate_chart_intent(user_prompt_text)
        is_chart = (chart_eval_peek['decision'] == guardrails.ChartIntentDecision.GENERATE)
        file_context_eval = guardrails.resolve_file_context(
            user_prompt=user_prompt_text,
            payload_file_ids=payload_file_ids,
            conversation_files=conv_files,
            is_chart_request=is_chart
        )
        active_file_ids = file_context_eval['active_file_ids']
    else:
        if payload_file_ids:
            active_file_ids = list(payload_file_ids)
        elif conv_files:
            active_file_ids = [cf['id'] for cf in conv_files]
        else:
            active_file_ids = []

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

    # Slice past conversation history
    # If user attached new file(s) in this message, start fresh without carrying over older document topics/charts
    if payload_file_ids:
        history = []
    else:
        history = list(_conversation)
        if len(history) > 6:
            history = history[-6:]

    if guardrails.is_standard_mode(jailbreak):
        history = guardrails.sanitize_conversation_history_for_charts(
            history=history,
            user_prompt=user_prompt_text,
            is_chart_request=is_chart
        )

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

    # Build final user prompt with file context embedded directly
    final_prompt_content = user_prompt_text
    if file_context_msg:
        if user_prompt_text and user_prompt_text.strip() != "Summarize and analyze the attached document.":
            final_prompt_content = f"{file_context_msg}\n\nUser Question/Instruction:\n{user_prompt_text}"
        else:
            final_prompt_content = f"{file_context_msg}\n\nPlease analyze and summarize the attached document."

    if guardrails.is_standard_mode(jailbreak):
        chart_intent = guardrails.evaluate_chart_intent(user_prompt_text)
        if chart_intent['decision'] == guardrails.ChartIntentDecision.GENERATE:
            chart_directive = guardrails.get_chart_directive(chart_intent['chart_type'])
            final_prompt_content = final_prompt_content + chart_directive
        else:
            no_chart_instruction = (
                "\n\n[NO CHART REQUESTED]:\n"
                "The user did NOT request a chart in this message. Do NOT generate or include any ```chart code block in your response, "
                "even if previous messages in this conversation contain charts. Respond strictly with clear text, explanations, or standard markdown formatting."
            )
            final_prompt_content = final_prompt_content + no_chart_instruction

        conversation.append({'role': 'user', 'content': final_prompt_content})

        if chart_intent['decision'] == guardrails.ChartIntentDecision.GENERATE:
            ctype_str = chart_intent['chart_type']
            conversation.insert(0, {
                'role': 'system',
                'content': (
                    f"You are an expert AI data assistant. The user explicitly requested a {ctype_str.capitalize()} Chart visualization. "
                    f"You MUST start your response directly with a ```chart JSON code block of type \"{ctype_str}\". "
                    "All data, titles, labels, and numbers MUST be strictly extracted from the provided document or conversation context. "
                    "NEVER output unrelated dummy categories like Gold, Silver, or Sales."
                )
            })
    else:
        # Non-standard modes (DAN Mode / EvilBOT Mode)
        chart_keywords = ['chart', 'pie', 'bar', 'graph', 'plot', 'visual', 'diagram', 'barchart', 'piechart']
        user_wants_chart = any(kw in user_prompt_text.lower() for kw in chart_keywords)

        if user_wants_chart:
            chart_mandatory_rule = (
                "\n\n[MANDATORY INSTRUCTION FOR VISUAL CHART]:\n"
                "The user explicitly requested a visual chart / pie chart / bar chart.\n"
                "CRITICAL RULES:\n"
                "1. REAL DATA ONLY: You MUST extract real entity names/categories and numerical values STRICTLY from the currently attached document content or prompt above.\n"
                "   - DO NOT hallucinate, invent placeholder data, or reuse topics/entities from earlier messages.\n"
                "   - The chart categories ('name') and values ('value') must strictly be the actual items and quantities found in this current document.\n"
                "2. START WITH ```chart: You MUST start your response immediately with the ```chart JSON code block as the very first characters of your output.\n"
                "3. JSON Format:\n"
                "```chart\n"
                "{\n"
                '  "type": "pie",\n'
                '  "title": "<Descriptive Title Reflecting The Document Data>",\n'
                '  "data": [\n'
                '    {"name": "<Actual Item Name From Document>", "value": <Actual Number>}\n'
                '  ]\n'
                "}\n"
                "```\n"
                "4. After the ```chart block, provide a clear text breakdown and explanation of the data from the document.\n"
                "START WITH ```chart DIRECTLY."
            )
            final_prompt_content = final_prompt_content + chart_mandatory_rule

        conversation.append({'role': 'user', 'content': final_prompt_content})

        if user_wants_chart:
            conversation.insert(0, {
                'role': 'system',
                'content': (
                    'You are an expert AI data assistant. Whenever the user requests a chart or visualization, you MUST start your response directly with a ```chart JSON code block. '
                    'All data, titles, labels, and numbers MUST be strictly extracted from the provided document or conversation context. '
                    'NEVER output unrelated dummy categories like Gold, Silver, or Sales.'
                )
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
