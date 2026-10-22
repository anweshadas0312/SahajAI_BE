import os
import time

_TTL = 30          # seconds, cache duration before re-reading from database
_cache = {}

DEFAULTS = {
    # --- quality gate / judge ---
    'min_answer_accuracy_threshold': 60,
    'target_final_accuracy': 75,
    'max_fallback_retries': 1,
    'weight_factual_accuracy': 0.40,
    'weight_relevance': 0.20,
    'weight_completeness': 0.15,
    'weight_source_reliability': 0.15,
    'weight_clarity_consistency': 0.10,
    'judge_model': '',
    'judge_max_chunks': 4,
    'judge_timeout_default': 30,
    'judge_timeout_cap': 30,
    'judge_answer_char_limit': 4000,
    'judge_prompt_overhead_tokens': 350,
    'judge_reserve_output_tokens': 200,
    'judge_contradiction_cutoff': 30,
    'judge_contradiction_cap': 40,
    'judge_outdated_cap': 20,
    'judge_parroting_cap': 15,
    'judge_enabled': True,
    'judge_current_keywords': 'today,current,currently,latest,now,weather,news,recent,2026',

    # --- search ---
    'fallback_search_engine': 'duckduckgo',
    'search_text_max_results': 6,
    'search_news_max_results': 8,
    'search_sources_limit': 6,
    'wiki_timeout': 4,
    'wiki_user_agent': 'SahajAI/1.0',
    'news_keywords': 'news,headline,headlines,cricket,match,score,today,latest,election,result,update',
    'live_data_keywords': 'weather,temperature,forecast,gold price,stock,crypto,price,rate,inflation',

    # --- history ---
    'history_max_messages': 6,
    'history_keep_when_condensing': 3,

    # --- chunking / context ---
    'default_context_window': 8192,
    'reserve_output_tokens': 1024,
    'context_safety_factor': 0.88,
    'condense_reserve_output_tokens': 600,
    'condense_prompt_overhead_tokens': 250,
    'condense_min_chunk_tokens': 400,
    'min_context_tokens': 300,
    'max_chunks': 8,
    'max_rounds': 3,
    'chunk_workers': 2,
    'llm_call_timeout_cap': 60,
    'llm_call_timeout_default': 30,
    'llm_connect_timeout': 10,
    'llm_first_token_timeout': 30,
    'llm_stream_timeout_default': 30,
    'llm_stream_read_timeout_cap': 30,
    'provider_cooldown_seconds': 60,
    'token_encoding': 'cl100k_base',

    # --- streaming / output ---
    'stream_piece_chars': 80,
    'fallback_confidence_default': 85,
    'fallback_default_model': 'mistral:latest',
    'confidence_template': '_Confidence: {score}% ({mode})_',
    'confidence_unverified_template': '_Confidence: unverified_',
    'sources_header': '**Sources:**',

    # --- prompts (placeholders: <<DATE>> <<YEAR>> <<QUESTION>> <<REFERENCE>> <<ANSWER>> <<TEXT>> <<ISSUES>>) ---
    'prompt_judge': (
        "You are an expert fact-checking and answer quality evaluator. Today's date is <<DATE>> (year <<YEAR>>).\n\n"
        "Question:\n<<QUESTION>>\n\n<<REFERENCE>>"
        "Answer to evaluate:\n<<ANSWER>>\n\n"
        "Evaluate the answer strictly according to the following 5 weighted criteria (0 to 100 each):\n"
        "1. factual_score (40% weight): Are factual claims strictly accurate and supported by the reference context? Heavily penalize any invented facts, conflicting numbers, false dates, or hallucinations. (0-30 = false/hallucinated, 31-59 = weak/unsupported, 60-84 = mostly factual, 85-100 = completely verified).\n"
        "2. relevance_score (20% weight): Does the answer directly and specifically answer the user's question without extraneous filler?\n"
        "3. completeness_score (15% weight): Are all key aspects of the question addressed? Does it acknowledge gaps honestly without guessing?\n"
        "4. source_reliability_score (15% weight): Are the stated facts grounded in credible evidence? Is temporal freshness respected? (If external research is not required for a self-contained question, rate based on logical consistency).\n"
        "5. clarity_score (10% weight): Is the answer internally consistent, well-structured, coherent, and free of contradictions?\n\n"
        "Critical Errors:\n"
        "Set has_critical_error to true if the answer makes demonstrably false claims, uses an outdated past year as the current date, invents unsupported numbers/metrics, or hallucinates citations.\n\n"
        "Reply ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "factual_score": <int 0-100>,\n'
        '  "relevance_score": <int 0-100>,\n'
        '  "completeness_score": <int 0-100>,\n'
        '  "source_reliability_score": <int 0-100>,\n'
        '  "clarity_score": <int 0-100>,\n'
        '  "has_critical_error": <true|false>,\n'
        '  "issues": [<string descriptions of specific errors or unsupported claims>],\n'
        '  "explanation": "<concise evaluation summary>"\n'
        "}"
    ),
    'prompt_fallback_correction': (
        "You are SahajAI's specialized fact-checking and accuracy improvement assistant. Today is <<DATE>> (year <<YEAR>>).\n\n"
        "User Question:\n<<QUESTION>>\n\n"
        "<<REFERENCE>>"
        "The previous draft had the following identified deficiencies and was rejected:\n"
        "<<ISSUES>>\n\n"
        "MANDATORY INSTRUCTIONS FOR IMPROVED ANSWER:\n"
        "1. STRICT FACTUAL GROUNDING: State facts ONLY that are supported by the provided reference evidence above.\n"
        "2. FIX DEFICIENCIES: Directly resolve and remove all identified errors, outdated claims, or unsupported statements.\n"
        "3. HONEST UNCERTAINTY: If specific details (exact numbers, dates, or results) are not present in the reference evidence, explicitly state that rather than inventing or guessing.\n"
        "4. DIRECT & COMPLETE: Answer the core question directly, clearly, and concisely in clean markdown.\n"
        "5. DO NOT cite URLs or write a Sources section; sources are attached automatically."
    ),
    'prompt_extract': (
        "Question: <<QUESTION>>\n\nText:\n<<TEXT>>\n\n"
        "Extract ONLY the facts from the text that help answer the question. "
        "Keep numbers, dates, names, units and [n]/URL markers exactly as written. "
        "Do not add outside knowledge. If nothing is relevant, reply exactly: NONE"
    ),
    'prompt_search_rules': (
        "1. TODAY'S DATE is <<DATE>>. Never present an older year as the current date.\n"
        "2. STRICT FACTUAL GROUNDING: Prioritize factual correctness above all else. State facts ONLY from the verified snippets above. Do NOT invent numbers, dates, statistics, or news.\n"
        "3. DISTINGUISH FACTS FROM UNCERTAINTY: If the snippets lack sufficient information to fully answer the question, clearly and honestly state what is known and what cannot be confirmed.\n"
        "4. SYNTHESIZE NATURALLY: Directly answer the question in clear, professional markdown. Use concise bullet points and headings.\n"
        "5. NEVER ECHO SEARCH METADATA: Do NOT write raw snippet blocks like '[1] Title: ... Summary: ...' or list generic portal homepages.\n"
        "6. DO NOT write a Sources section or URLs. Sources are added automatically by the system.\n"
        "7. NUMERICAL ACCURACY: Ensure all numbers, currency units, percentages, and dates match the reference snippets exactly."
    ),

    # --- status texts (comma separated variants) ---
    'status_evaluating_quality': 'Evaluating answer quality...,Fact-checking against reference material...,Verifying data points and citations...,Assessing answer completeness and accuracy...',
    'status_verifying': 'Verifying the revised answer...,Cross-checking with primary model...,Validating updated response accuracy...',
    'status_researched_fallback': 'Re-verifying with search and fallback model...,Consulting fallback model for improved accuracy...,Gathering fresh search results for cross-verification...',
    'status_chunking_context': 'Context is large, extracting key facts across chunks...,Processing extensive context into concise factual notes...,Condensing large documents for context fit...',
}


def _cast(val, default):
    t = type(default)
    if t is bool:
        return str(val).strip().lower() in ('1', 'true', 'yes', 'on')
    if t is int:
        return int(float(val))
    if t is float:
        return float(val)
    return str(val)


def get(key):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < _TTL:
        return hit[1]

    default = DEFAULTS[key]            # KeyError on typo for early detection
    val = None
    try:
        import db
        val = db.get_system_config(key)
    except Exception:
        pass
    if val in (None, ''):
        val = os.getenv(key.upper())
    if val in (None, ''):
        val = default
    try:
        val = _cast(val, default)
    except Exception:
        val = default

    _cache[key] = (now, val)
    return val


def get_list(key):
    return [x.strip() for x in str(get(key)).split(',') if x.strip()]


def get_for_model(key, model_name):
    """Per-model override: system_config e '<key>.<model_name>' thakle seta, na hole global."""
    if model_name:
        try:
            import db
            v = db.get_system_config(f"{key}.{model_name}")
            if v not in (None, ''):
                return _cast(v, DEFAULTS[key])
        except Exception:
            pass
    return get(key)


def render(template_key, **vars):
    """<<NAME>> placeholder replace (JSON er { } er sathe conflict hoy na)."""
    out = get(template_key)
    for k, v in vars.items():
        out = out.replace(f"<<{k.upper()}>>", str(v))
    return out


def clear_cache():
    _cache.clear()
