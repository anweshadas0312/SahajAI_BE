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
                           prepare_messages, context_overflows, generate_text, fit_to_budget,
                           strip_reasoning, SEARCH_MARK, SEARCH_ID, USER_SEP,
                           clean_search_query, verify_number_consistency, verify_temporal_consistency,
                           determine_query_evidence_need, resolve_search_coreference,
                           rank_and_filter_search_results, is_time_sensitive_query)



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


def _parse_judge_json(raw: str) -> dict:
    """Safely extracts and parses JSON evaluation response from LLM output."""
    if not raw:
        return None
    # 1. Strip markdown code fence if present
    cleaned = raw.strip()
    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # Find outer braces
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1].strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2. Regex fallback extraction for malformed JSON
    out = {}
    f_match = re.search(r'"factual(?:_score)?"\s*:\s*(\d{1,3})', raw, re.I)
    r_match = re.search(r'"relevance(?:_score)?"\s*:\s*(\d{1,3})', raw, re.I)
    c_match = re.search(r'"completeness(?:_score)?"\s*:\s*(\d{1,3})', raw, re.I)
    s_match = re.search(r'"source_reliability(?:_score)?"\s*:\s*(\d{1,3})', raw, re.I)
    cl_match = re.search(r'"clarity(?:_score)?"\s*:\s*(\d{1,3})', raw, re.I)
    legacy_score = re.search(r'"score"\s*:\s*(\d{1,3})', raw, re.I)

    if not any([f_match, r_match, c_match, s_match, cl_match, legacy_score]):
        return None

    if legacy_score and not f_match:
        base_val = int(legacy_score.group(1))
        out['factual_score'] = base_val
        out['relevance_score'] = base_val
        out['completeness_score'] = base_val
        out['source_reliability_score'] = base_val
        out['clarity_score'] = base_val
    else:
        out['factual_score'] = int(f_match.group(1)) if f_match else 50
        out['relevance_score'] = int(r_match.group(1)) if r_match else 70
        out['completeness_score'] = int(c_match.group(1)) if c_match else 60
        out['source_reliability_score'] = int(s_match.group(1)) if s_match else 60
        out['clarity_score'] = int(cl_match.group(1)) if cl_match else 70

    crit_match = re.search(r'"has_critical_error"\s*:\s*(true|false)', raw, re.I)
    out['has_critical_error'] = (crit_match.group(1).lower() == 'true') if crit_match else False

    exp_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', raw, re.I)
    out['explanation'] = exp_match.group(1) if exp_match else "Extracted via pattern match"
    out['issues'] = []
    return out


def _judge_chunk(user_query, answer_text, chunk, provider, now):
    if chunk:
        ref_text = f"Reference material (partial):\n{chunk}\n\n"
    else:
        ref_text = (
            "Reference material:\n"
            "[Self-contained / General reasoning query — no external search evidence was required. "
            "Evaluate factual accuracy based on established consensus, mathematical/algorithmic correctness, and logical soundness. "
            "For source_reliability_score, evaluate whether the reasoning is methodologically sound and internally grounded "
            "rather than penalizing the absence of external citations.]\n\n"
        )
    prompt = settings.render(
        'prompt_judge',
        date=f"{now:%Y-%m-%d}",
        year=now.year,
        question=user_query,
        reference=ref_text,
        answer=answer_text[:settings.get('judge_answer_char_limit')]
    )
    timeout_val = min(int(provider.get('timeout') or settings.get('judge_timeout_default')),
                      settings.get('judge_timeout_cap'))
    try:
        resp = requests.post(
            provider['endpoint'],
            json={"model": provider['model_name'], "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0, "stream": False, "max_tokens": 350},
            timeout=(10, timeout_val))

        if resp.status_code != 200:
            return None
        raw = strip_reasoning(resp.json()['choices'][0]['message'].get('content', ''))
        parsed = _parse_judge_json(raw)
        if not parsed:
            return None

        # Extract criterion scores (0-100 clamped)
        factual = max(0, min(100, int(parsed.get('factual_score', 50))))
        relevance = max(0, min(100, int(parsed.get('relevance_score', 50))))
        completeness = max(0, min(100, int(parsed.get('completeness_score', 50))))
        source_rel = max(0, min(100, int(parsed.get('source_reliability_score', 50))))
        clarity = max(0, min(100, int(parsed.get('clarity_score', 50))))

        # Compute weighted overall score
        w_f = settings.get('weight_factual_accuracy') or 0.40
        w_r = settings.get('weight_relevance') or 0.20
        w_c = settings.get('weight_completeness') or 0.15
        w_s = settings.get('weight_source_reliability') or 0.15
        w_cl = settings.get('weight_clarity_consistency') or 0.10
        chunk_score = int(round(w_f * factual + w_r * relevance + w_c * completeness + w_s * source_rel + w_cl * clarity))

        raw_issues = parsed.get('issues', [])
        if isinstance(raw_issues, str):
            issues_list = [raw_issues.strip()] if raw_issues.strip() else []
        elif isinstance(raw_issues, list):
            issues_list = [str(x).strip() for x in raw_issues if str(x).strip()]
        else:
            issues_list = []

        return {
            'score': max(0, min(100, chunk_score)),
            'factual_score': factual,
            'relevance_score': relevance,
            'completeness_score': completeness,
            'source_reliability_score': source_rel,
            'clarity_score': clarity,
            'has_critical_error': bool(parsed.get('has_critical_error', False)),
            'issues': issues_list,
            'explanation': str(parsed.get('explanation', '')),
            'relevant': (relevance >= 25)
        }
    except Exception as e:
        print(f"[Judge chunk] failed: {e}")
        return None


def evaluate_answer_accuracy(user_query, answer_text, context_text, judge_provider) -> dict:
    if not judge_provider or not judge_provider.get('endpoint'):
        return {
            'score': 100,
            'factual_score': 100,
            'relevance_score': 100,
            'completeness_score': 100,
            'source_reliability_score': 100,
            'clarity_score': 100,
            'has_critical_error': False,
            'issues': [],
            'explanation': 'No evaluator configured',
            'reason': 'No evaluator'
        }

    now = datetime.now()
    snippets = _snippets_only(context_text)

    budget = get_budget(judge_provider, reserve_output=settings.get('judge_reserve_output_tokens'))
    fixed = (count_tokens(user_query)
             + count_tokens(answer_text[:settings.get('judge_answer_char_limit')])
             + settings.get('judge_prompt_overhead_tokens'))
    chunk_budget = max(500, budget - fixed)
    evidence_block = fit_to_budget(snippets, chunk_budget) if snippets else ""
    print(f"\n⚖️ [FALLBACK JUDGE REVIEW] Model: '{judge_provider.get('model_name')}' | Evaluating against focused evidence block ({count_tokens(evidence_block)} tokens, budget={chunk_budget})...")

    judge_res = _judge_chunk(user_query, answer_text, evidence_block, judge_provider, now)
    if not judge_res and evidence_block:
        print(f"⚠️ [JUDGE RETRY] Initial evaluation call failed or timed out. Retrying with compact evidence slice...")
        compact_block = fit_to_budget(snippets, 400)
        judge_res = _judge_chunk(user_query, answer_text, compact_block, judge_provider, now)

    results = [judge_res] if judge_res else []
    
    # Safe failure handling: Never auto-approve on judge failure
    if not results:
        print(f"❌ [FALLBACK JUDGE FAILED] Judge model did not return valid JSON evaluation.")
        return {
            'score': 0,
            'factual_score': 0,
            'relevance_score': 0,
            'completeness_score': 0,
            'source_reliability_score': 0,
            'clarity_score': 0,
            'has_critical_error': True,
            'issues': ['Evaluator call timed out or failed to return a valid structured quality evaluation'],
            'explanation': 'Judge evaluation failed',
            'reason': 'Judge failed'
        }

    relevant = [r for r in results if r['relevant']]
    if not relevant:
        relevant = results  # Fall back to all results if chunk relevance was ambiguous

    # 1. Compute criterion averages across evaluated chunks
    avg_factual = int(sum(r['factual_score'] for r in relevant) / len(relevant))
    avg_relevance = int(sum(r['relevance_score'] for r in relevant) / len(relevant))
    avg_completeness = int(sum(r['completeness_score'] for r in relevant) / len(relevant))
    avg_source_rel = int(sum(r['source_reliability_score'] for r in relevant) / len(relevant))
    avg_clarity = int(sum(r['clarity_score'] for r in relevant) / len(relevant))
    has_critical_error = any(r.get('has_critical_error') for r in relevant)

    collected_issues = []
    for r in relevant:
        collected_issues.extend(r.get('issues', []))

    # 2. Objective Fact Verification Checks
    # (a) Temporal / Outdated Year check
    is_outdated, out_msg = verify_temporal_consistency(user_query, answer_text, now.year)
    if is_outdated:
        print(f"⚠️ [JUDGE OUTDATED PENALTY] {out_msg}")
        avg_factual = min(avg_factual, settings.get('judge_outdated_cap'))
        has_critical_error = True
        collected_issues.append(out_msg)

    # (b) Number / Metric Verification check against snippets
    if snippets:
        match_ratio, unverified_nums = verify_number_consistency(answer_text, snippets, user_query=user_query)
        if unverified_nums and match_ratio == 0.0:
            print(f"⚠️ [JUDGE NUMBER CHECK] None of the cited numbers {unverified_nums} found in snippets. Capping score.")
            avg_factual = min(avg_factual, settings.get('judge_contradiction_cap'))
            has_critical_error = True
            collected_issues.append(f"Unverified figures not found in evidence: {', '.join(unverified_nums[:5])}")
        elif unverified_nums and match_ratio < 0.5:
            print(f"⚠️ [JUDGE NUMBER CHECK] Low number grounding ratio ({match_ratio:.2f}).")
            avg_factual = min(avg_factual, 50)
            collected_issues.append(f"Partially unverified figures: {', '.join(unverified_nums[:3])}")
    else:
        # For self-contained queries without external snippets, align source reliability with factual correctness
        avg_source_rel = max(avg_source_rel, avg_factual)

    # (c) Parroting check
    if re.search(r'\[\d+\]\s*(?:Title|Summary|Source)\s*:', answer_text, re.IGNORECASE) or 'Summary: Read full articles' in answer_text:
        print("⚠️ [JUDGE PARROTING PENALTY] Answer merely echoed raw search snippet metadata format.")
        avg_clarity = min(avg_clarity, settings.get('judge_parroting_cap'))
        avg_factual = min(avg_factual, 35)
        has_critical_error = True
        collected_issues.append("Answer merely echoed raw search snippet metadata format without synthesis")

    # If factual grounding is low (< 40%), flag critical error
    if avg_factual < 40:
        has_critical_error = True

    # 3. Calculate Final Overall Score consistently from weighted criteria
    w_f = settings.get('weight_factual_accuracy') or 0.40
    w_r = settings.get('weight_relevance') or 0.20
    w_c = settings.get('weight_completeness') or 0.15
    w_s = settings.get('weight_source_reliability') or 0.15
    w_cl = settings.get('weight_clarity_consistency') or 0.10
    final_score = int(round(
        w_f * avg_factual +
        w_r * avg_relevance +
        w_c * avg_completeness +
        w_s * avg_source_rel +
        w_cl * avg_clarity
    ))
    final_score = max(0, min(100, final_score))

    # Deduplicate issues preserving order
    dedup_issues = list(dict.fromkeys(collected_issues))
    explanation = relevant[0].get('explanation', '') if relevant else 'Evaluation complete'

    print(f"📊 [FALLBACK JUDGE ACCURACY] Evaluated by '{judge_provider['model_name']}' -> Final Quality Score: {final_score}/100 | Factual: {avg_factual}% | Relevance: {avg_relevance}% | Completeness: {avg_completeness}% | Sources: {avg_source_rel}% | Clarity: {avg_clarity}% | Critical Error: {has_critical_error}")
    return {
        'score': final_score,
        'factual_score': avg_factual,
        'relevance_score': avg_relevance,
        'completeness_score': avg_completeness,
        'source_reliability_score': avg_source_rel,
        'clarity_score': avg_clarity,
        'has_critical_error': has_critical_error,
        'issues': dedup_issues,
        'explanation': explanation,
        'reason': 'Evaluated'
    }


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

    # 2. Designated Single Fallback candidate: match fallback_model or secondary_model or next active provider
    fallback_provider = None
    if not fallback_model:
        fallback_model = db.get_system_config('secondary_model') or ''
    if fallback_model:
        fallback_provider = next(
            (p for p in providers if p.get('model_name') == fallback_model and p.get('is_active') and p.get('id') != (primary_provider.get('id') if primary_provider else None)),
            None
        )
    if not fallback_provider and primary_provider:
        # Fall back to any active provider different from primary to ensure two-tier fallback pipeline
        fallback_provider = next(
            (p for p in providers if p.get('is_active') and p.get('id') != primary_provider.get('id')),
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


def build_fallback_messages(conversation, new_search_results, deficiencies=None):
    """Purano search system msg bad, notun search ta LAST user message e embed (history er shuru e na)."""
    msgs = []
    original_search_content = ""
    for m in conversation:
        c = m.get('content', '') or ''
        if m.get('role') == 'system' and SEARCH_ID in c:
            original_search_content = c
            continue
        msgs.append(dict(m))

    correction_block = ""
    if deficiencies:
        issues_text = "\n".join(f"- {issue}" for issue in deficiencies if issue)
        if issues_text:
            correction_block = (
                f"\n\n[MANDATORY CORRECTION INSTRUCTIONS FOR FALLBACK ANSWER]:\n"
                f"An initial draft answer was rejected due to the following specific deficiencies:\n"
                f"{issues_text}\n"
                f"Your task: Correct every one of these deficiencies. Discard any ungrounded assertions. "
                f"Ground all facts strictly in the verified context. If facts are absent, clearly acknowledge the uncertainty."
            )

    evidence_to_embed = ""
    if new_search_results:
        evidence_to_embed = "\n\n".join(s['content'] for s in new_search_results)
    elif original_search_content:
        evidence_to_embed = original_search_content

    if evidence_to_embed:
        snippets, _, instr = evidence_to_embed.partition(SEARCH_MARK)
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get('role') == 'user':
                msgs[i]['content'] = (f"{snippets.strip()}{USER_SEP}{msgs[i]['content']}"
                                      f"{correction_block}"
                                      f"\n\n{SEARCH_MARK}{instr}")
                break
    elif correction_block:
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get('role') == 'user':
                msgs[i]['content'] = f"{msgs[i]['content']}{correction_block}"
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


def is_candidate_better(cand, base):
    """
    Compares candidate evaluation dictionaries or candidate objects.
    Returns True if cand is strictly better than base.
    Comparison criteria (in priority order):
    1. Critical error elimination (critical error in base, none in cand).
    2. Critical error introduction (critical error in cand, none in base -> False).
    3. Factual correctness score.
    4. Overall evaluation score.
    """
    if not cand:
        return False
    if not base:
        return True

    c_eval = cand.get('eval') if isinstance(cand, dict) and 'eval' in cand and isinstance(cand.get('eval'), dict) else cand
    b_eval = base.get('eval') if isinstance(base, dict) and 'eval' in base and isinstance(base.get('eval'), dict) else base

    c_crit = c_eval.get('has_critical_error', False)
    b_crit = b_eval.get('has_critical_error', False)
    if b_crit and not c_crit:
        return True
    if not b_crit and c_crit:
        return False

    c_fact = c_eval.get('factual_score', 0)
    b_fact = b_eval.get('factual_score', 0)
    if c_fact > b_fact:
        return True
    if c_fact < b_fact:
        return False

    return c_eval.get('score', 0) > b_eval.get('score', 0)


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

            # Stage 3 (Conditional): If internet access is enabled, dynamically evaluate retrieval need
            search_sources_dict = None
            if internet_access:
                needs_web, strat_reason = determine_query_evidence_need(user_prompt_text, has_active_files=bool(active_file_ids))
                if needs_web:
                    status_3 = get_dynamic_status('searching_web', user_prompt_text)
                    yield f"data: {json.dumps({'type': 'status', 'stage': 'searching_web', 'text': status_3})}\n\n"
                    # Resolve multi-turn coreference if this is a follow-up
                    search_prompt = resolve_search_coreference(user_prompt_text, history)
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    query = f'Current date: {current_date}. ' + search_prompt
                    print(f"\n🌐 [SEARCH 1] Searching web via DuckDuckGo + Wikipedia for: '{search_prompt}' (strategy: {strat_reason})")
                    search_results, search_sources_dict = fetch_search_results(query, return_sources=True)
                    links_found = len(search_sources_dict.get('all', [])) if search_sources_dict else 0
                    print(f"✅ [SEARCH 1 DONE] Retrieved {len(search_results)} search context block(s) with {links_found} source link(s).")
                    conversation.extend(search_results)
                else:
                    print(f"\n⚡ [DYNAMIC EVIDENCE STRATEGY] Query identified as '{strat_reason}'. External search bypassed to preserve context purity & speed.")

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
                                judged = (eval_result['reason'] == 'Evaluated')
                                has_critical_error = eval_result.get('has_critical_error', False)
                                min_accuracy_threshold = settings.get('min_answer_accuracy_threshold')  # 60%

                                # Primary acceptable condition:
                                # Must be judged, score >= 60, no critical errors, and factual score >= 50.
                                # Never auto-approve on judge failure!
                                is_primary_acceptable = (
                                    judged
                                    and accuracy_score >= min_accuracy_threshold
                                    and not has_critical_error
                                    and eval_result.get('factual_score', 100) >= 50
                                )

                                if is_primary_acceptable:
                                    print(f"✅ [QUALITY GATE PASSED] Accuracy: {accuracy_score}% >= Threshold: {min_accuracy_threshold}%. Directly releasing buffered Primary answer to UI!")
                                    final_confidence = accuracy_score
                                    final_mode = 'verified'
                                    for c in primary_buffer:
                                        assistant_accumulated.append(c)
                                        yield f"data: {json.dumps({'type': 'content', 'delta': c})}\n\n"
                                    success = True
                                    if provider.get('id'):
                                        _provider_cooldowns.pop(provider['id'], None)
                                    break

                                # Trigger Fallback path
                                deficiencies = eval_result.get('issues', [])
                                print(f"🚨 [QUALITY GATE FAILED] Score: {accuracy_score}% (Threshold: {min_accuracy_threshold}% | Critical Error: {has_critical_error}). Deficiencies: {deficiencies}")
                                print(f"🔄 [RE-SEARCH & FALLBACK TRIGGERED] Re-verifying evidence and generating corrected answer...")
                                yield f"data: {json.dumps({'type': 'status', 'stage': 'researched_fallback', 'text': get_dynamic_status('researched_fallback', user_prompt_text)})}\n\n"

                                # Check evidence sufficiency: if factual score was low or no search context existed, do targeted re-search
                                new_results = []
                                if internet_access:
                                    clean_q = clean_search_query(user_prompt_text)
                                    if is_time_sensitive_query(user_prompt_text):
                                        re_query = f"{clean_q} {datetime.now():%Y}"
                                    else:
                                        re_query = clean_q
                                    print(f"🔍 [RE-SEARCH QUERY] '{re_query}'")
                                    new_results, new_sources = fetch_search_results(
                                        re_query, return_sources=True, engine=settings.get('fallback_search_engine')
                                    )
                                    print(f"✅ [RE-SEARCH DONE] Retrieved {len(new_results)} fresh context block(s).")
                                    if new_sources and (new_sources.get('all') or new_sources.get('wikipedia')):
                                        if search_sources_dict:
                                            existing_urls = {s.get('url') for s in search_sources_dict.get('all', []) if s.get('url')}
                                            for ns in new_sources.get('all', []):
                                                if ns.get('url') and ns.get('url') not in existing_urls:
                                                    search_sources_dict['all'].append(ns)
                                                    existing_urls.add(ns.get('url'))
                                            if new_sources.get('wikipedia'):
                                                search_sources_dict['wikipedia'] = new_sources['wikipedia']
                                        else:
                                            search_sources_dict = new_sources

                                # Build fallback messages containing question, evidence, and specific deficiencies to fix
                                fb_conv = build_fallback_messages(conversation, new_results, deficiencies=deficiencies)
                                fb_tokens = count_messages_tokens(fb_conv)
                                fb_budget = get_budget(fallback_provider_candidate)
                                print(f"📏 [FALLBACK CONTEXT CHECK] Fallback Model: '{fallback_provider_candidate.get('model_name') if fallback_provider_candidate else 'None'}' | Re-search Tokens: {fb_tokens} | Budget Window: {fb_budget}")
                                if context_overflows(fb_conv, fallback_provider_candidate):
                                    print(f"⚠️ [FALLBACK CHUNKING] Re-search context overflows Fallback budget. Chunking...")
                                    yield f"data: {json.dumps({'type': 'status', 'stage': 'chunking_context', 'text': get_dynamic_status('chunking_context', user_prompt_text)})}\n\n"
                                fb_msgs = prepare_messages(fb_conv, fallback_provider_candidate, user_prompt_text)

                                print(f"🚀 [FALLBACK GENERATION] Calling Fallback LLM '{fallback_provider_candidate.get('model_name') if fallback_provider_candidate else 'None'}' to generate corrected answer...")
                                try:
                                    fb_answer = generate_text(fallback_provider_candidate, fb_msgs)
                                    print(f"✅ [FALLBACK GENERATION DONE] Generated {len(fb_answer)} chars. Evaluating fallback quality...")
                                except Exception as e:
                                    print(f"❌ [FALLBACK GENERATION FAILED] {e}")
                                    fb_answer = ""

                                # Evaluate the Fallback answer using the same 5-criterion rubric
                                fb_eval = None
                                fb_score = 0
                                if fb_answer.strip():
                                    verifying_text = random.choice(settings.get_list('status_verifying'))
                                    yield f"data: {json.dumps({'type': 'status', 'stage': 'evaluating_quality', 'text': verifying_text})}\n\n"
                                    fallback_ref_context = ref_context
                                    if new_results:
                                        fallback_ref_context += "\n\n" + "\n\n".join(s['content'] for s in new_results)

                                    fb_eval = evaluate_answer_accuracy(user_prompt_text, fb_answer, fallback_ref_context, judge_provider)
                                    fb_score = fb_eval.get('score', 0)
                                    print(f"📊 [FALLBACK EVALUATION RESULT] Score: {fb_score}/100 (Target: 75%) | Issues: {fb_eval.get('issues', [])}")

                                    # Bounded Improvement Cycle: If score < 75% and deficiencies exist, run up to 1 targeted refinement round
                                    target_accuracy = settings.get('target_final_accuracy') or 75
                                    max_retries = settings.get('max_fallback_retries') or 1
                                    if fb_score < target_accuracy and fb_eval.get('issues') and max_retries > 0:
                                        print(f"🔄 [BOUNDED REFINEMENT] Score {fb_score}% < {target_accuracy}%. Executing 1 bounded refinement round targeting: {fb_eval['issues']}")
                                        refine_msgs = build_fallback_messages(conversation, new_results, deficiencies=fb_eval.get('issues', []))
                                        refine_msgs.append({'role': 'assistant', 'content': fb_answer})
                                        refine_msgs.append({
                                            'role': 'user',
                                            'content': (
                                                f"Your previous response had the following identified deficiencies:\n"
                                                + "\n".join(f"- {i}" for i in fb_eval.get('issues', []))
                                                + "\nPlease provide a fully factual, corrected final answer adhering strictly to the verified context."
                                            )
                                        })
                                        refine_prep = prepare_messages(refine_msgs, fallback_provider_candidate, user_prompt_text)
                                        try:
                                            refined_answer = generate_text(fallback_provider_candidate, refine_prep)
                                            if refined_answer.strip():
                                                refined_eval = evaluate_answer_accuracy(user_prompt_text, refined_answer, fallback_ref_context, judge_provider)
                                                if is_candidate_better(refined_eval, fb_eval):
                                                    print(f"✅ [REFINEMENT SUCCEEDED] Improved quality: {fb_score}% -> {refined_eval['score']}% (Factual: {fb_eval.get('factual_score')}% -> {refined_eval.get('factual_score')}%)")
                                                    fb_answer = refined_answer
                                                    fb_score = refined_eval['score']
                                                    fb_eval = refined_eval
                                                else:
                                                    print(f"⚠️ [REFINEMENT NO GAIN] Refined answer did not resolve deficiencies or improve accuracy. Retaining best previous candidate.")
                                        except Exception as r_err:
                                            print(f"[Refinement note]: {r_err}")

                                # Build candidates for primary and fallback
                                candidate_primary = {
                                    'text': full_primary_text,
                                    'eval': eval_result,
                                    'score': accuracy_score,
                                    'factual_score': eval_result.get('factual_score', 0),
                                    'has_critical_error': has_critical_error,
                                    'is_evaluated': judged
                                }
                                candidate_fallback = {
                                    'text': fb_answer,
                                    'eval': fb_eval,
                                    'score': fb_score,
                                    'factual_score': fb_eval.get('factual_score', 0) if fb_eval else 0,
                                    'has_critical_error': fb_eval.get('has_critical_error', True) if fb_eval else True,
                                    'is_evaluated': bool(fb_eval and fb_eval.get('reason') == 'Evaluated')
                                }

                                # Select best candidate using is_candidate_better
                                if candidate_fallback['text'].strip() and is_candidate_better(candidate_fallback, candidate_primary):
                                    best_candidate = candidate_fallback
                                    print(f"🏆 [FINAL CANDIDATE] Selected Fallback answer (Score: {best_candidate['score']}%, Critical: {best_candidate['has_critical_error']})")
                                else:
                                    best_candidate = candidate_primary
                                    print(f"🏆 [FINAL CANDIDATE] Selected Primary answer (Score: {best_candidate['score']}%, Critical: {best_candidate['has_critical_error']})")

                                # Verify if best candidate meets acceptance threshold
                                is_acceptable = (
                                    best_candidate.get('is_evaluated', False)
                                    and best_candidate.get('score', 0) >= min_accuracy_threshold
                                    and not best_candidate.get('has_critical_error', False)
                                    and best_candidate.get('factual_score', 0) >= 50
                                )

                                raw_text = best_candidate['text']
                                if is_acceptable:
                                    final_text = raw_text
                                    final_confidence = best_candidate['score']
                                    final_mode = 'verified'
                                else:
                                    # Fallback could not fully resolve issues or score < threshold
                                    disclaimer = (
                                        "> ⚠️ **Notice: Insufficient Verification** — "
                                        "This response could not be fully verified against reliable sources and may contain inaccuracies. "
                                        "Please cross-check critical information.\n\n"
                                    )
                                    if not raw_text.startswith("> ⚠️ **Notice: Insufficient Verification**"):
                                        final_text = disclaimer + raw_text
                                    else:
                                        final_text = raw_text

                                    if best_candidate.get('is_evaluated', False):
                                        final_confidence = best_candidate['score']
                                        final_mode = 'low_confidence'
                                    else:
                                        final_confidence = None
                                        final_mode = 'unverified'

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
    Fetches encyclopedic summary from Wikipedia OpenSearch, Search API, and REST API.
    """
    try:
        import urllib.parse
        clean_q = clean_search_query(query)
        if not clean_q or len(clean_q) < 3:
            return None

        wiki_timeout = settings.get('wiki_timeout') or 4
        wiki_ua = settings.get('wiki_user_agent') or 'SahajAI/1.0'
        headers = {'User-Agent': f'{wiki_ua} (contact@sahaj.ai; https://sahaj.ai)'}

        title = None
        url = None

        # 1. Try OpenSearch first
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(clean_q)}&limit=2&namespace=0&format=json"
        res = requests.get(search_url, headers=headers, timeout=wiki_timeout)
        if res.status_code == 200:
            data = res.json()
            if len(data) >= 4 and data[1] and data[3]:
                title = data[1][0]
                url = data[3][0]

        # 2. If OpenSearch did not return a match (e.g. for multi-word or complex queries), try Search API
        if not title:
            sr_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_q)}&format=json&utf8=1&srlimit=1"
            res_sr = requests.get(sr_url, headers=headers, timeout=wiki_timeout)
            if res_sr.status_code == 200:
                sr_data = res_sr.json()
                search_hits = sr_data.get('query', {}).get('search', [])
                if search_hits:
                    title = search_hits[0].get('title')
                    url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

        if title:
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
    :param engine: Search engine/backend ('auto', 'duckduckgo', etc.)
    :return: List of search results (or tuple if return_sources=True)  
    """
    sources_collected = {'wikipedia': None, 'duckduckgo': None, 'all': []}
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = []
        clean_q = clean_search_query(query)
        q_lower = query.lower()
        is_news_query = any(k in q_lower for k in settings.get_list('news_keywords'))
        is_weather_or_finance = any(k in q_lower for k in settings.get_list('live_data_keywords'))

        # If engine was set to 'bing', handle honestly without claiming fake provider
        if engine and engine.lower() == 'bing':
            print("[Search Provider Notice] Bing API backend not configured in environment. Using DuckDuckGo Search provider.")

        # 1. Fetch DuckDuckGo results (news if requested or live events, else general text)
        with DDGS() as ddgs:
            if is_news_query:
                news_q = re.sub(r'\b(headlines?|today)\b', '', clean_q, flags=re.I).strip() or clean_q
                try:
                    results = list(ddgs.news(news_q, max_results=settings.get('search_news_max_results')))
                except Exception:
                    try:
                        results = list(ddgs.news(clean_q, max_results=settings.get('search_news_max_results')))
                    except Exception:
                        results = []
            if not results:
                try:
                    results = list(ddgs.text(clean_q, max_results=settings.get('search_text_max_results')))
                except Exception as de:
                    print(f"[DDG text search note]: {de}")
                    results = []

        # 2. Fetch Wikipedia encyclopedic context
        wiki_data = None
        if not (is_news_query or is_weather_or_finance):
            wiki_data = fetch_wikipedia_summary(query)
        if wiki_data:
            # Place Wikipedia article at the top of context for strong factual anchor
            results.insert(0, wiki_data)
            sources_collected['wikipedia'] = {'title': wiki_data['title'], 'url': wiki_data['url']}

        # Rank and filter results by query relevance and noise/boilerplate removal
        results = rank_and_filter_search_results(clean_q, results)

        # Record first genuine DuckDuckGo web result
        for r in results:
            r_url = r.get("url") or r.get("href")
            r_title = r.get("title") or r.get("source") or "DuckDuckGo Web Search"
            if r_url and r_url.startswith("http") and "wikipedia.org" not in r_url:
                sources_collected['duckduckgo'] = {'title': r_title, 'url': r_url}
                break

        if not results:
            if return_sources:
                return [], sources_collected
            return []

        snippets = ""
        seen_urls = set()
        for index, result in enumerate(results):
            body = (result.get("body") or result.get("snippet") or "").strip()
            href = result.get("url") or result.get("href") or ""
            title = (result.get("title") or "").strip()
            source = result.get("source") or ("Wikipedia" if "wikipedia.org" in href else "")
            date_info = result.get("date") or ""
            if href in seen_urls:
                continue
            if href:
                seen_urls.add(href)
            if body or title:
                date_line = f"Date: {date_info}\n" if date_info else ""
                snippets += f"[{index + 1}] Title: {title}\n{date_line}Summary: {body}\nSource: {source}\nURL: {href}\n\n"
                clean_src = source if source else title
                if href:
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
