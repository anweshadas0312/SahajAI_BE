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
import settings
from context_utils import (count_tokens, count_messages_tokens, get_budget, pack_chunks, map_chunks,
                           prepare_messages, context_overflows, generate_text,
                           strip_reasoning, SEARCH_MARK, SEARCH_ID, USER_SEP)



_provider_cooldowns = {}  # provider_id -> cooldown_expiry_timestamp


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

    elif stage == 'evaluating_quality':
        return random.choice(settings.get_list('status_evaluating_quality'))

    elif stage == 'researched_fallback':
        return random.choice(settings.get_list('status_researched_fallback'))

    elif stage == 'chunking_context':
        return random.choice(settings.get_list('status_chunking_context'))

    return "Thinking..."


def _snippets_only(context_text: str, limit: int = 10**7) -> str:
    return (context_text or '').split(SEARCH_MARK)[0][:limit].strip()


def _judge_chunk(user_query, answer_text, chunk, provider, now):
    prompt = settings.render(
        'prompt_judge',
        date=f"{now:%Y-%m-%d}",
        year=now.year,
        question=user_query,
        reference=(f"Reference material (partial):\n{chunk}\n\n" if chunk else ""),
        answer=answer_text[:settings.get('judge_answer_char_limit')]
    )
    timeout_val = min(int(provider.get('timeout') or settings.get('judge_timeout_default')),
                      settings.get('judge_timeout_cap'))
    try:
        resp = requests.post(
            provider['endpoint'],
            json={"model": provider['model_name'], "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0, "stream": False},
            timeout=(10, timeout_val))
        if resp.status_code != 200:
            return None
        raw = strip_reasoning(resp.json()['choices'][0]['message'].get('content', ''))
        print(f"[Judge raw]: {raw[:200]}")
        s = re.search(r'"score"\s*:\s*(\d{1,3})', raw)
        if not s:
            return None
        rel = re.search(r'"relevant"\s*:\s*(true|false)', raw, re.IGNORECASE)
        return {'score': max(0, min(100, int(s.group(1)))),
                'relevant': (rel.group(1).lower() == 'true') if rel else True}
    except Exception as e:
        print(f"[Judge chunk] failed: {e}")
        return None


def evaluate_answer_accuracy(user_query, answer_text, context_text, judge_provider) -> dict:
    if not judge_provider or not judge_provider.get('endpoint'):
        return {'score': 100, 'reason': 'No evaluator'}

    now = datetime.now()
    snippets = _snippets_only(context_text)

    budget = get_budget(judge_provider, reserve_output=settings.get('judge_reserve_output_tokens'))
    fixed = (count_tokens(user_query)
             + count_tokens(answer_text[:settings.get('judge_answer_char_limit')])
             + settings.get('judge_prompt_overhead_tokens'))
    chunk_budget = max(500, budget - fixed)
    max_chunks = settings.get('judge_max_chunks')
    chunks = pack_chunks(snippets, chunk_budget)[:max_chunks] if snippets else [""]
    print(f"\n⚖️ [FALLBACK JUDGE REVIEW] Model: '{judge_provider.get('model_name')}' | Evaluating against {len(chunks)} search reference chunk(s) (chunk_budget={chunk_budget} tokens)...")

    results = [r for r in map_chunks(
        chunks, lambda c: _judge_chunk(user_query, answer_text, c, judge_provider, now)) if r]
    if not results:
        print(f"❌ [FALLBACK JUDGE FAILED] Judge model did not return valid JSON evaluation.")
        return {'score': 100, 'reason': 'Judge failed'}

    relevant = [r for r in results if r['relevant']]
    if not relevant:
        print(f"🚨 [FALLBACK JUDGE] No chunks marked relevant by {judge_provider.get('model_name')}. Grounding failed (score=0).")
        return {'score': 0, 'reason': 'Evaluated'}

    mean = sum(r['score'] for r in relevant) / len(relevant)
    cut = settings.get('judge_contradiction_cutoff')
    cap = settings.get('judge_contradiction_cap')
    score = int(min(mean, cap)) if min(r['score'] for r in relevant) < cut else int(mean)

    kw = '|'.join(re.escape(k) for k in settings.get_list('judge_current_keywords'))
    if re.search(rf'\b({kw})\b', user_query, re.IGNORECASE):
        for ym in re.finditer(r'\bas of (?:[A-Za-z]+\s+)?(20\d{2})\b', answer_text, re.IGNORECASE):
            if int(ym.group(1)) < now.year:
                print(f"⚠️ [JUDGE OUTDATED PENALTY] Detected outdated year {ym.group(1)} < {now.year}. Capping score.")
                score = min(score, settings.get('judge_outdated_cap'))
                break

    # Parroting check: if the answer merely echoed raw search snippet metadata format
    if re.search(r'\[\d+\]\s*(?:Title|Summary|Source)\s*:', answer_text, re.IGNORECASE) or 'Summary: Read full articles' in answer_text:
        print("⚠️ [JUDGE PARROTING PENALTY] Answer merely echoed raw search snippet metadata format. Capping score.")
        score = min(score, settings.get('judge_parroting_cap'))

    # Check if numbers cited in the answer exist in reference snippets
    if snippets:
        nums = {n.replace(',', '') for n in re.findall(r'\b\d[\d,]{2,}\b', answer_text or '')}
        if nums:
            ctx_clean = snippets.replace(',', '')
            found_ratio = sum(n in ctx_clean for n in nums) / len(nums)
            if found_ratio == 0.0:
                print(f"⚠️ [JUDGE NUMBER CHECK] None of the {len(nums)} numbers {nums} found in snippets. Capping score.")
                score = min(score, settings.get('judge_contradiction_cap'))

    print(f"📊 [FALLBACK JUDGE ACCURACY] Evaluated by '{judge_provider['model_name']}' -> Final Quality Score: {score}/100 ({len(relevant)}/{len(chunks)} chunk(s) relevant)")
    return {'score': score, 'reason': 'Evaluated'}


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
    # Check cooldown for primary candidate
    now = time.time()
    primary_is_cooling_down = False
    if primary_provider and primary_provider.get('id') in _provider_cooldowns:
        if now < _provider_cooldowns[primary_provider['id']]:
            primary_is_cooling_down = True
            remaining = int(_provider_cooldowns[primary_provider['id']] - now)
            print(f"[LLM Cooldown] Primary model '{primary_provider.get('model_name')}' is on cooldown ({remaining}s remaining). Skipping directly to Fallback.")
        else:
            _provider_cooldowns.pop(primary_provider['id'], None)

    # 2. Designated Single Fallback candidate: match fallback_model from database and must be active
    fallback_provider = None
    if fallback_model:
        fallback_provider = next(
            (p for p in providers if p.get('model_name') == fallback_model and p.get('is_active') and p.get('id') != (primary_provider.get('id') if primary_provider else None)),
            None
        )

    if primary_is_cooling_down and fallback_provider:
        # Step [2]: Primary down/cooldown -> directly route to [7] Fallback path
        candidates.append(fallback_provider)
    else:
        if primary_provider:
            candidates.append(primary_provider)
            seen_ids.add(primary_provider.get('id'))
        if fallback_provider and fallback_provider.get('id') not in seen_ids:
            candidates.append(fallback_provider)
            seen_ids.add(fallback_provider.get('id'))

    # Strictly limit to Primary + 1 Fallback model only (no endless looping across other models)
    return candidates


def build_fallback_messages(conversation, new_search_results):
    """Purano search system msg bad, notun search ta LAST user message e embed (history er shuru e na)."""
    msgs = []
    for m in conversation:
        c = m.get('content', '') or ''
        if m.get('role') == 'system' and SEARCH_ID in c:
            continue
        msgs.append(dict(m))

    if new_search_results:
        raw = "\n\n".join(s['content'] for s in new_search_results)
        snippets, _, instr = raw.partition(SEARCH_MARK)
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get('role') == 'user':
                msgs[i]['content'] = (f"{snippets.strip()}{USER_SEP}{msgs[i]['content']}"
                                      f"\n\n{SEARCH_MARK}{instr}")
                break
    return msgs


def build_sources_list(sd):
    if not sd:
        return []
    out, seen = [], set()
    for key in ('wikipedia', 'duckduckgo'):
        s = sd.get(key)
        if s and s.get('url') and s['url'] not in seen:
            out.append({'title': s['title'], 'url': s['url']}); seen.add(s['url'])
    limit = settings.get('search_sources_limit')
    for s in sd.get('all', [])[:limit]:
        if s['url'] not in seen:
            out.append(s); seen.add(s['url'])
    return out


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
        jailbreak = req_json.get('jailbreak', 'default')
        default_model = db.get_system_config('default_model') or ''
        if not default_model:
            first_active = next((p for p in db.get_all_llm_providers() if p.get('is_active')), None)
            default_model = first_active.get('model_name') if first_active else ''
        model = req_json.get('model') or default_model or settings.get('fallback_default_model')

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
                max_hist = settings.get('history_max_messages')
                if len(history) > max_hist:
                    history = history[-max_hist:]

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
            search_sources_dict = None
            if internet_access:
                status_3 = get_dynamic_status('searching_web', user_prompt_text)
                yield f"data: {json.dumps({'type': 'status', 'stage': 'searching_web', 'text': status_3})}\n\n"
                current_date = datetime.now().strftime("%Y-%m-%d")
                query = f'Current date: {current_date}. ' + user_prompt_text
                print(f"\n🌐 [SEARCH 1] Searching web via DuckDuckGo + Wikipedia for: '{user_prompt_text}'")
                search_results, search_sources_dict = fetch_search_results(query, return_sources=True)
                links_found = len(search_sources_dict.get('all', [])) if search_sources_dict else 0
                print(f"✅ [SEARCH 1 DONE] Retrieved {len(search_results)} search context block(s) with {links_found} source link(s).")
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
                derived_chart = None
                if chart_intent['decision'] == guardrails.ChartIntentDecision.GENERATE:
                    if active_file_ids:
                        for fid in active_file_ids:
                            sp = None
                            if conv_files:
                                cf = next((f for f in conv_files if str(f.get('id', '')) == str(fid)), None)
                                if cf and cf.get('storage_path'):
                                    sp = guardrails.get_local_storage_path(cf.get('storage_path', ''))
                            if not sp and 'file_context_eval' in locals() and file_context_eval.get('matched_files'):
                                mf = next((f for f in file_context_eval['matched_files'] if str(f.get('id', '')) == str(fid)), None)
                                if mf and mf.get('storage_path'):
                                    sp = guardrails.get_local_storage_path(mf.get('storage_path', ''))
                            if not sp:
                                try:
                                    f_rec = db.get_file_by_id(fid)
                                    if f_rec and f_rec.get('storage_path'):
                                        sp = guardrails.get_local_storage_path(f_rec.get('storage_path', ''))
                                except Exception:
                                    pass
                            if sp:
                                derived_chart = guardrails.derive_tabular_chart_data(sp, user_prompt_text, chart_intent['chart_type'])
                                if derived_chart:
                                    break
                    chart_directive = guardrails.get_chart_directive(chart_intent['chart_type'], derived_metrics=derived_chart)
                    final_prompt_content = final_prompt_content + chart_directive

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
            primary_provider = candidates[0] if candidates else None
            fallback_provider_candidate = candidates[1] if len(candidates) > 1 else None
            judge_model_name = settings.get('judge_model')
            judge_provider = None
            if judge_model_name:
                judge_provider = next((p for p in (db.get_all_llm_providers() or []) if p.get('model_name') == judge_model_name and p.get('is_active')), None)
            if not judge_provider:
                judge_provider = fallback_provider_candidate
            final_confidence = None
            final_mode = None
            min_accuracy_threshold = settings.get('min_answer_accuracy_threshold')

            for idx, provider in enumerate(candidates):
                api_url = provider.get('endpoint')
                if not api_url:
                    continue
                actual_model = provider.get('model_name') or model
                timeout_val = provider.get('timeout') or settings.get('llm_stream_timeout_default')

                status_prefix = "PRIMARY" if idx == 0 else "FALLBACK"
                print(f"\n🤖 [LLM REQUEST] >>> Trying {status_prefix} Model [{idx+1}/{len(candidates)}]: '{actual_model}' ({provider.get('name')}) | Endpoint: {api_url}")

                if idx > 0:
                    primary_label = candidates[0].get('name') or candidates[0].get('model_name')
                    fallback_label = provider.get('name') or actual_model
                    print(f"🔄 [FAILOVER SWITCH] >>> Primary model ({primary_label}) unreachable. Switched to Fallback: '{fallback_label}'")
                    notice = f"> ℹ️ *Note: Primary model ({primary_label}) did not respond. Continuing with fallback model ({fallback_label})...*\n\n"
                    assistant_accumulated.append(notice)
                    yield f"data: {json.dumps({'type': 'content', 'delta': notice})}\n\n"

                eval_gate_active = (
                    settings.get('judge_enabled')
                    and idx == 0
                    and judge_provider is not None
                    and (internet_access or bool(file_context_msg))
                    and not (guardrails.is_standard_mode(jailbreak) and is_chart)
                )

                tot_tokens = count_messages_tokens(conversation)
                bud_tokens = get_budget(provider)
                print(f"📏 [CONTEXT SIZE CHECK] Candidate: '{actual_model}' ({status_prefix}) | Total Context Tokens: {tot_tokens} | Model Budget Window: {bud_tokens}")
                if context_overflows(conversation, provider):
                    print(f"⚠️ [CONTEXT OVERFLOW] Context size ({tot_tokens} tokens) > Model Budget ({bud_tokens} tokens)! Chunking & condensing...")
                    yield f"data: {json.dumps({'type': 'status', 'stage': 'chunking_context', 'text': get_dynamic_status('chunking_context', user_prompt_text)})}\n\n"
                else:
                    print(f"✨ [CONTEXT OK] Context size ({tot_tokens} tokens) fits inside model budget ({bud_tokens} tokens). No chunking needed.")
                llm_messages = prepare_messages(conversation, provider, user_prompt_text)

                if idx == 0 and eval_gate_active:
                    print(f"⏳ [PRIMARY GENERATION] Calling Primary Model '{actual_model}' | Buffering initial answer in memory for Quality Review...")
                else:
                    print(f"🚀 [STREAMING GENERATION] Calling Model '{actual_model}' ({status_prefix}) | Streaming response directly to UI...")

                payload = {
                    "model": actual_model,
                    "messages": llm_messages,
                    "stream": True
                }

                try:
                    conn_timeout = int(settings.get('llm_connect_timeout') or 10)
                    first_token_timeout = min(
                        int(timeout_val or settings.get('llm_first_token_timeout') or 30),
                        int(settings.get('llm_stream_read_timeout_cap') or 30)
                    )
                    with requests.post(api_url, json=payload, stream=True, timeout=(conn_timeout, first_token_timeout)) as r:
                        if r.status_code >= 400:
                            print(f"❌ [LLM HTTP ERROR] Model '{actual_model}' returned HTTP {r.status_code}. Trying next provider...")
                            continue

                        req_start = time.time()
                        stream_received = False
                        is_std_chart = guardrails.is_standard_mode(jailbreak) and is_chart
                        chart_buffer = []
                        chart_validated = False
                        primary_buffer = []

                        for line in r.iter_lines():
                            if not stream_received and (time.time() - req_start > first_token_timeout):
                                raise requests.exceptions.Timeout(f"Initial token generation exceeded timeout ({first_token_timeout}s)")
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

                                            open_idx = buf_str.find('```')
                                            if open_idx != -1:
                                                if open_idx > 0:
                                                    # Flush introductory prose before the chart block immediately
                                                    intro_chunk = buf_str[:open_idx]
                                                    assistant_accumulated.append(intro_chunk)
                                                    yield f"data: {json.dumps({'type': 'content', 'delta': intro_chunk})}\n\n"
                                                    buf_str = buf_str[open_idx:]
                                                    chart_buffer = [buf_str]

                                                close_idx = buf_str.find('```', 3)
                                                if close_idx != -1:
                                                    # Chart code block is complete, validate and sanitize it
                                                    chart_validated = True
                                                    chart_block_text = buf_str[:close_idx + 3]
                                                    trailing_text = buf_str[close_idx + 3:]

                                                    expected_c_type = chart_intent.get('chart_type') if 'chart_intent' in locals() else None
                                                    is_v, clean_text, payload_res, v_err = guardrails.validate_and_sanitize_chart_response(
                                                        chart_block_text,
                                                        expected_chart_type=expected_c_type,
                                                        file_context=file_context_msg,
                                                        derived_chart=derived_chart if 'derived_chart' in locals() else None
                                                    )
                                                    assistant_accumulated.append(clean_text)
                                                    yield f"data: {json.dumps({'type': 'content', 'delta': clean_text})}\n\n"
                                                    if trailing_text:
                                                        assistant_accumulated.append(trailing_text)
                                                        yield f"data: {json.dumps({'type': 'content', 'delta': trailing_text})}\n\n"
                                                    chart_buffer = []
                                            else:
                                                # No ``` backticks found yet
                                                if len(buf_str) > 300:
                                                    # Safely stream leading prose without losing any forming backticks
                                                    flush_part = buf_str[:-20]
                                                    assistant_accumulated.append(flush_part)
                                                    yield f"data: {json.dumps({'type': 'content', 'delta': flush_part})}\n\n"
                                                    chart_buffer = [buf_str[-20:]]
                                        elif eval_gate_active:
                                            filtered_content = re.sub(r'\[NO CHART REQUESTED\]:?', '', content)
                                            if filtered_content:
                                                primary_buffer.append(filtered_content)
                                        else:
                                            # Strip accidental prompt leakages such as [NO CHART REQUESTED] from streaming text
                                            filtered_content = re.sub(r'\[NO CHART REQUESTED\]:?', '', content)
                                            if filtered_content:
                                                assistant_accumulated.append(filtered_content)
                                                yield f"data: {json.dumps({'type': 'content', 'delta': filtered_content})}\n\n"
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
                                    file_context=file_context_msg,
                                    derived_chart=derived_chart if 'derived_chart' in locals() else None
                                )
                                assistant_accumulated.append(clean_text)
                                yield f"data: {json.dumps({'type': 'content', 'delta': clean_text})}\n\n"
                            else:
                                for c in chart_buffer:
                                    clean_c = re.sub(r'\[NO CHART REQUESTED\]:?', '', c)
                                    if clean_c:
                                        assistant_accumulated.append(clean_c)
                                        yield f"data: {json.dumps({'type': 'content', 'delta': clean_c})}\n\n"

                        if stream_received:
                            if eval_gate_active:
                                full_primary_text = "".join(primary_buffer)
                                print(f"📝 [PRIMARY ANSWER COMPLETE] Primary generated {len(full_primary_text)} chars. Sending to Fallback Judge for review...")
                                yield f"data: {json.dumps({'type': 'status', 'stage': 'evaluating_quality', 'text': get_dynamic_status('evaluating_quality', user_prompt_text)})}\n\n"

                                ref_context = file_context_msg or ""
                                for m in conversation:
                                    if m.get('role') == 'system' and SEARCH_ID in m.get('content', ''):
                                        ref_context += "\n" + m['content']

                                eval_result = evaluate_answer_accuracy(user_prompt_text, full_primary_text,
                                                                       ref_context, judge_provider)
                                accuracy_score = eval_result['score']
                                judged = eval_result['reason'] == 'Evaluated'

                                if (eval_result['reason'] == 'Judge failed') or (judged and accuracy_score >= min_accuracy_threshold):
                                    print(f"✅ [QUALITY GATE PASSED] Accuracy: {accuracy_score}% >= Threshold: {min_accuracy_threshold}%. Directly releasing buffered Primary answer to UI!")
                                    if judged:
                                        final_confidence = accuracy_score
                                        final_mode = 'verified'
                                    for c in primary_buffer:
                                        assistant_accumulated.append(c)
                                        yield f"data: {json.dumps({'type': 'content', 'delta': c})}\n\n"
                                    success = True
                                    if provider.get('id'):
                                        _provider_cooldowns.pop(provider['id'], None)
                                    break

                                # [5]-[7] score < threshold
                                print(f"🚨 [QUALITY GATE FAILED] Score: {accuracy_score}% < Threshold: {min_accuracy_threshold}%. Primary answer rejected!")
                                print(f"🔄 [RE-SEARCH TRIGGERED] Fallback initiated! Re-searching web with '{settings.get('fallback_search_engine')}'...")
                                yield f"data: {json.dumps({'type': 'status', 'stage': 'researched_fallback', 'text': get_dynamic_status('researched_fallback', user_prompt_text)})}\n\n"

                                re_query = f"{user_prompt_text} {datetime.now():%B %d %Y}"
                                print(f"🔍 [RE-SEARCH QUERY] '{re_query}'")
                                new_results, new_sources = fetch_search_results(
                                    re_query, return_sources=True, engine=settings.get('fallback_search_engine')
                                )
                                print(f"✅ [RE-SEARCH DONE] Retrieved {len(new_results)} fresh context block(s).")
                                if new_sources:
                                    search_sources_dict = new_sources

                                fb_conv = build_fallback_messages(conversation, new_results)
                                fb_tokens = count_messages_tokens(fb_conv)
                                fb_budget = get_budget(fallback_provider_candidate)
                                print(f"📏 [FALLBACK CONTEXT CHECK] Fallback Model: '{fallback_provider_candidate.get('model_name')}' | Re-search Tokens: {fb_tokens} | Budget Window: {fb_budget}")
                                if context_overflows(fb_conv, fallback_provider_candidate):
                                    print(f"⚠️ [FALLBACK CHUNKING] Re-search context ({fb_tokens} tokens) > Fallback Budget ({fb_budget} tokens)! Chunking for Fallback...")
                                    yield f"data: {json.dumps({'type': 'status', 'stage': 'chunking_context', 'text': get_dynamic_status('chunking_context', user_prompt_text)})}\n\n"
                                else:
                                    print(f"✨ [FALLBACK CONTEXT OK] Re-search context fits inside Fallback window. No chunking needed.")
                                fb_msgs = prepare_messages(fb_conv, fallback_provider_candidate, user_prompt_text)

                                print(f"🚀 [FALLBACK GENERATION] Calling Fallback LLM '{fallback_provider_candidate.get('model_name')}' to generate final accurate answer...")
                                try:
                                    fb_answer = generate_text(fallback_provider_candidate, fb_msgs)
                                    print(f"✅ [FALLBACK GENERATION DONE] Generated {len(fb_answer)} chars. Streaming to UI with verified confidence score...")
                                except Exception as e:
                                    print(f"❌ [FALLBACK GENERATION FAILED] {e}")
                                    fb_answer = ""

                                if fb_answer.strip():
                                    final_text = fb_answer
                                    verifying_text = random.choice(settings.get_list('status_verifying'))
                                    yield f"data: {json.dumps({'type': 'status', 'stage': 'evaluating_quality', 'text': verifying_text})}\n\n"
                                    new_ctx = "\n\n".join(s['content'] for s in new_results)
                                    verify = evaluate_answer_accuracy(user_prompt_text, fb_answer, new_ctx, judge_provider)
                                    if verify.get('reason') == 'Evaluated':
                                        final_confidence = verify['score']
                                        final_mode = 'verified' if verify['score'] >= min_accuracy_threshold else 'low_confidence'
                                    else:
                                        final_confidence = int(settings.get('fallback_confidence_default') or 85)
                                        final_mode = 'verified'
                                else:
                                    final_text = full_primary_text
                                    final_confidence = accuracy_score
                                    final_mode = 'low_confidence'

                                step = settings.get('stream_piece_chars')
                                for i in range(0, len(final_text), step):
                                    piece = final_text[i:i + step]
                                    assistant_accumulated.append(piece)
                                    yield f"data: {json.dumps({'type': 'content', 'delta': piece})}\n\n"
                                success = True
                                if provider.get('id'):
                                    _provider_cooldowns.pop(provider['id'], None)
                                break
                            else:
                                print(f"[LLM Response] <<< Successfully generated response using model: '{actual_model}' ({provider.get('name')})\n")
                                success = True
                                if provider.get('id'):
                                    _provider_cooldowns.pop(provider['id'], None)
                                break

                except Exception as req_err:
                    print(f"❌ [LLM FAILED] Model '{actual_model}' failed with error: {req_err}")
                    if provider.get('id'):
                        cooldown_secs = int(settings.get('provider_cooldown_seconds') or 60)
                        _provider_cooldowns[provider['id']] = time.time() + cooldown_secs
                        print(f"⏳ [COOLDOWN] Model '{actual_model}' placed on {cooldown_secs}s cooldown. Failing over to next candidate...")
                    continue

            if not success:
                err_msg = "⚠️ [LLM Service Error]: All configured AI models are currently unreachable. Please verify network connectivity or check provider endpoints."
                assistant_accumulated.append(err_msg)
                yield f"data: {json.dumps({'type': 'error', 'error': err_msg})}\n\n"
            else:
                current_output = "".join(assistant_accumulated)
                appended_text = ""

                # Strip model-invented sources block to prevent fake links / duplication
                current_output_clean = re.sub(r'\n+\**(?:Sources|References|Citations)\**\s*:.*$', '', current_output,
                                              flags=re.IGNORECASE | re.DOTALL)
                if current_output_clean != current_output:
                    assistant_accumulated = [current_output_clean]
                    current_output = current_output_clean

                # Append sources as plain markdown links if internet access was enabled
                if internet_access and search_sources_dict:
                    src_list = build_sources_list(search_sources_dict)
                    if src_list:
                        missing_sources = []
                        for s in src_list:
                            s_url = s.get('url')
                            s_title = s.get('title') or 'Source'
                            if s_url:
                                missing_sources.append(f"[{s_title}]({s_url})")

                        if missing_sources:
                            sources_header = settings.get('sources_header')
                            appended_text += f"\n\n{sources_header}\n- " + "\n- ".join(missing_sources)

                # Append confidence badge as plain markdown text if evaluated
                if final_confidence is not None:
                    mode_label = final_mode or 'verified'
                    appended_text += "\n\n" + settings.get('confidence_template').format(score=final_confidence, mode=mode_label)
                elif final_mode == 'unverified':
                    appended_text += "\n\n" + settings.get('confidence_unverified_template')

                if appended_text:
                    assistant_accumulated.append(appended_text)
                    yield f"data: {json.dumps({'type': 'content', 'delta': appended_text})}\n\n"

            # Persist completed assistant message to database
            if assistant_accumulated and conversation_id:
                try:
                    full_reply = "".join(assistant_accumulated)
                    if guardrails.is_standard_mode(jailbreak):
                        if is_chart:
                            expected_c_type = chart_intent.get('chart_type') if 'chart_intent' in locals() else None
                            _, full_reply, _, _ = guardrails.validate_and_sanitize_chart_response(
                                full_reply,
                                expected_chart_type=expected_c_type,
                                file_context=file_context_msg,
                                derived_chart=derived_chart if 'derived_chart' in locals() else None
                            )
                        else:
                            # Standard non-chart turn: ensure no stray delimiter leaks or unsolicited chart blocks
                            full_reply = re.sub(r'```(?:chart|pie|bar|line)\b', '```text', full_reply, flags=re.IGNORECASE)
                            full_reply = re.sub(r'\s*\[(?:END OF CHART REQUEST|NO CHART REQUESTED)\]:?', '', full_reply, flags=re.IGNORECASE).strip()
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
        max_hist = settings.get('history_max_messages')
        if len(history) > max_hist:
            history = history[-max_hist:]

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



def fetch_wikipedia_summary(query: str):
    """
    Fetches encyclopedic summary from Wikipedia OpenSearch and REST API.
    """
    try:
        import urllib.parse
        clean_q = re.sub(r'^Current date: \d{4}-\d{2}-\d{2}\.\s*', '', query).strip()
        # Clean question prefixes like 'who is', 'what is', 'tell me about'
        clean_q = re.sub(r'^(?:who|what|where|tell me about|explain)\s+(?:is|was|are|the|a)?\s*', '', clean_q, flags=re.IGNORECASE).strip()
        if not clean_q or len(clean_q) < 3:
            return None

        wiki_timeout = settings.get('wiki_timeout')
        wiki_ua = settings.get('wiki_user_agent')
        headers = {'User-Agent': f'{wiki_ua} (contact@sahaj.ai; https://sahaj.ai)'}
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(clean_q)}&limit=1&namespace=0&format=json"
        res = requests.get(search_url, headers=headers, timeout=wiki_timeout)
        if res.status_code == 200:
            data = res.json()
            if len(data) >= 4 and data[1] and data[3]:
                title = data[1][0]
                url = data[3][0]
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                sum_res = requests.get(summary_url, headers=headers, timeout=wiki_timeout)
                if sum_res.status_code == 200:
                    extract = sum_res.json().get('extract', '')
                    if extract:
                        return {
                            'title': f"Wikipedia: {title}",
                            'body': extract,
                            'url': url,
                            'source': 'Wikipedia'
                        }
    except Exception as e:
        print(f"[Wikipedia Search Note] {e}")
    return None


def fetch_search_results(query, return_sources=False, engine='auto'):
    """  
    Fetch search results for a given query combining DuckDuckGo and Wikipedia.  

    :param query: Search query string  
    :param return_sources: If True, returns a tuple of (messages_list, sources_dict)
    :param engine: Search engine/backend ('auto', 'bing', etc.)
    :return: List of search results (or tuple if return_sources=True)  
    """
    sources_collected = {'wikipedia': None, 'duckduckgo': None, 'all': []}
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = []
        clean_q = re.sub(r'^Current date: \d{4}-\d{2}-\d{2}\.\s*', '', query).strip()
        q_lower = query.lower()
        is_news_query = any(k in q_lower for k in settings.get_list('news_keywords'))
        is_weather_or_finance = any(k in q_lower for k in settings.get_list('live_data_keywords'))
        kwargs = {} if engine == 'auto' else {'backend': engine}

        # 1. Fetch DuckDuckGo results (news if requested, else general text)
        with DDGS() as ddgs:
            if is_news_query:
                news_q = re.sub(r'\b(headlines?|today)\b', '', clean_q, flags=re.I).strip() or "breaking news"
                try:
                    results = list(ddgs.news(news_q, max_results=settings.get('search_news_max_results')))
                except Exception:
                    try:
                        results = list(ddgs.news(clean_q, max_results=settings.get('search_news_max_results')))
                    except Exception:
                        results = []
            if not results:
                try:
                    results = list(ddgs.text(clean_q, max_results=settings.get('search_text_max_results'), **kwargs))
                except TypeError:
                    try:
                        results = list(ddgs.text(clean_q, max_results=settings.get('search_text_max_results')))
                    except Exception:
                        results = []
                except Exception:
                    results = []

        # Filter out generic portal directory descriptions if substantive results exist
        def _is_generic_portal(body_text):
            b_low = (body_text or '').lower()
            return ('read full articles, watch videos' in b_low or
                    'browse thousands of titles' in b_low or
                    'get breaking news stories and in-depth coverage' in b_low)

        filtered = [r for r in results if not _is_generic_portal(r.get('body') or r.get('snippet'))]
        if filtered:
            results = filtered

        # Find first valid DuckDuckGo web result
        for r in results:
            r_url = r.get("url") or r.get("href")
            r_title = r.get("title") or r.get("source") or "DuckDuckGo Web Search"
            if r_url and r_url.startswith("http"):
                sources_collected['duckduckgo'] = {'title': r_title, 'url': r_url}
                break

        # If no direct DDG search link found, construct dedicated fallback search link
        if not sources_collected['duckduckgo']:
            import urllib.parse
            encoded_q = urllib.parse.quote(clean_q)
            sources_collected['duckduckgo'] = {
                'title': f"DuckDuckGo Search: {clean_q[:40]}",
                'url': f"https://duckduckgo.com/?q={encoded_q}"
            }

        # 2. Fetch Wikipedia encyclopedic context
        wiki_data = None
        if not (is_news_query or is_weather_or_finance):
            wiki_data = fetch_wikipedia_summary(query)
        if wiki_data:
            # Place Wikipedia article at the top of context for strong factual anchor
            results.insert(0, wiki_data)
            sources_collected['wikipedia'] = {'title': wiki_data['title'], 'url': wiki_data['url']}
        elif not (is_news_query or is_weather_or_finance):
            # Dedicated Wikipedia search fallback link
            import urllib.parse
            clean_term = re.sub(r'^(?:who|what|where|tell me about|explain)\s+(?:is|was|are|the|a)?\s*', '', clean_q, flags=re.IGNORECASE).strip()
            encoded_wiki = urllib.parse.quote(clean_term or clean_q)
            sources_collected['wikipedia'] = {
                'title': f"Wikipedia: {(clean_term or clean_q)[:40]}",
                'url': f"https://en.wikipedia.org/w/index.php?search={encoded_wiki}"
            }

        if not results:
            if return_sources:
                return [], sources_collected
            return []

        snippets = ""
        citation_examples = []
        for index, result in enumerate(results):
            body = result.get("body", "") or result.get("snippet", "")
            href = result.get("url", "") or result.get("href", "")
            title = result.get("title", "")
            source = result.get("source", "")
            if body or title:
                snippets += f"[{index + 1}] Title: {title}\nSummary: {body}\nSource: {source}\nURL: {href}\n\n"
                clean_src = source if source else title
                if href:
                    citation_examples.append(f"[{clean_src}]({href})")
                    sources_collected['all'].append({'title': clean_src, 'url': href})

        if not snippets.strip():
            if return_sources:
                return [], sources_collected
            return []

        current_today = datetime.now().strftime("%Y-%m-%d")
        response = (
            f"Here are the latest live web & Wikipedia search results retrieved today ({current_today}):\n\n"
            f"{snippets}\n"
            f"{SEARCH_MARK}:\n"
            + settings.render('prompt_search_rules', date=current_today)
        )

        messages = [{'role': 'system', 'content': response}]
        if return_sources:
            return messages, sources_collected
        return messages
    except Exception as e:
        print(f"[Web Search Warning] {e}")
        if return_sources:
            return [], sources_collected
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
