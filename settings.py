import os
import time

_TTL = 30          # seconds, cache duration before re-reading from database
_cache = {}

DEFAULTS = {
    # --- quality gate / judge ---
    'min_answer_accuracy_threshold': 60,
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
    'judge_current_keywords': 'today,current,currently,latest,now,weather,news',

    # --- search ---
    'fallback_search_engine': 'bing',
    'search_text_max_results': 6,
    'search_news_max_results': 8,
    'search_sources_limit': 6,
    'wiki_timeout': 4,
    'wiki_user_agent': 'SahajAI/1.0',
    'news_keywords': 'news,headline,headlines,cricket,match,score,today,latest',
    'live_data_keywords': 'weather,temperature,forecast,gold price,stock,crypto,price',

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

    # --- prompts (placeholders: <<DATE>> <<YEAR>> <<QUESTION>> <<REFERENCE>> <<ANSWER>> <<TEXT>>) ---
    'prompt_judge': (
        "You are a strict fact-checking judge. Today is <<DATE>> (year <<YEAR>>).\n\n"
        "Question:\n<<QUESTION>>\n\n<<REFERENCE>>"
        "Answer to evaluate:\n<<ANSWER>>\n\n"
        "Rules:\n"
        "- relevant=false if this reference material says nothing about the question.\n"
        "- score 0-30: contradicts the reference, uses an old year as current, or FAILS to answer the question (e.g. merely lists search directory websites like 'Google News', website homepages, or echoes raw snippet markers '[n] Title:' without providing substantive facts/news).\n"
        "- score 31-59: partially correct, vague, or incomplete answer.\n"
        "- score 60-100: informative, accurate, substantive, and directly answers the question consistent with the reference.\n"
        'Reply ONLY with JSON: {"relevant": true|false, "score": <0-100>}'
    ),
    'prompt_extract': (
        "Question: <<QUESTION>>\n\nText:\n<<TEXT>>\n\n"
        "Extract ONLY the facts from the text that help answer the question. "
        "Keep numbers, dates, names, units and [n]/URL markers exactly as written. "
        "Do not add outside knowledge. If nothing is relevant, reply exactly: NONE"
    ),
    'prompt_search_rules': (
        "1. TODAY'S DATE is <<DATE>>. Never present an older year as the current date.\n"
        "2. STRICT GROUNDING: State facts ONLY from the snippets above. Do NOT invent numbers, weather data or news. "
        "If the snippets lack actual facts, figures, or events, say so honestly.\n"
        "3. SYNTHESIZE NATURALLY: Write like a professional journalist or analyst. Use clear markdown headings (###) and bullet points.\n"
        "4. NEVER ECHO SEARCH METADATA: Do NOT write raw snippet blocks like '[1] Title: ... Summary: ...' or list generic website homepages. Explain the actual events and news stories.\n"
        "5. DO NOT write a Sources section or URLs. Sources are added automatically.\n"
        "6. Always state the carat, city, unit, and date if the snippets mention it (e.g. 22K vs 24K, city name)."
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
