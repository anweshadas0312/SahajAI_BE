import os
import re
import json
import requests
from concurrent.futures import ThreadPoolExecutor

import settings

try:
    import importlib
    tiktoken = importlib.import_module("tiktoken")
    _enc = tiktoken.get_encoding(settings.get('token_encoding'))
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text or "")) + 1
except Exception:
    def count_tokens(text: str) -> int:
        text = text or ""
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        return int(len(text) / (2.0 if non_ascii > len(text) * 0.3 else 3.5)) + 1

SEARCH_MARK = "MANDATORY INSTRUCTIONS FOR ANSWERING"
SEARCH_ID = "live web & Wikipedia search results"
FILE_SEP = "\n\nUser Question/Instruction:\n"
USER_SEP = "\n\n---\nUser Question:\n"


def strip_reasoning(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text or '', flags=re.DOTALL | re.IGNORECASE).strip()


def get_budget(provider: dict, reserve_output=None, safety=None) -> int:
    reserve_output = settings.get('reserve_output_tokens') if reserve_output is None else reserve_output
    safety = settings.get('context_safety_factor') if safety is None else safety
    # priority: provider row -> per-model system_config -> global default
    ctx = (provider or {}).get('context_window') \
          or settings.get_for_model('default_context_window', (provider or {}).get('model_name'))
    return max(512, int((int(ctx) - reserve_output) * safety))


def split_question(content: str):
    """(context_part, separator, question_part). Last separator dhore bhage."""
    idx, sep = -1, ""
    for s in (FILE_SEP, USER_SEP):
        i = content.rfind(s)
        if i > idx:
            idx, sep = i, s
    if idx < 0:
        return content, "", ""
    return content[:idx], sep, content[idx + len(sep):]


def _split_oversized(block: str, max_tokens: int) -> list:
    parts, cur = [], ""
    for sent in re.split(r'(?<=[.!?।])\s+', block):
        if cur and count_tokens(cur + " " + sent) > max_tokens:
            parts.append(cur)
            cur = sent
        else:
            cur = (cur + " " + sent).strip()
    if cur:
        parts.append(cur)
    out = []
    for p in parts:
        while count_tokens(p) > max_tokens:
            out.append(p[:max_tokens * 2])
            p = p[max_tokens * 2:]
        if p:
            out.append(p)
    return out


def pack_chunks(text: str, max_tokens: int) -> list:
    blocks = []
    for b in re.split(r'\n{2,}', text or ""):
        b = b.strip()
        if not b:
            continue
        blocks.extend([b] if count_tokens(b) <= max_tokens else _split_oversized(b, max_tokens))
    chunks, cur, cur_t = [], [], 0
    for b in blocks:
        t = count_tokens(b)
        if cur and cur_t + t > max_tokens:
            chunks.append("\n\n".join(cur))
            cur, cur_t = [], 0
        cur.append(b)
        cur_t += t
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def fit_to_budget(text: str, max_tokens: int) -> str:
    chunks = pack_chunks(text, max_tokens)
    return chunks[0] if chunks else ""


def map_chunks(chunks: list, fn, max_workers: int = None) -> list:
    if max_workers is None:
        max_workers = settings.get('chunk_workers')
    if len(chunks) <= 1:
        return [fn(c) for c in chunks]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, chunks))


def _total(msgs):
    return sum(count_tokens(m.get('content', '')) + 8 for m in msgs)


def count_messages_tokens(msgs):
    return _total(msgs)


def context_overflows(messages: list, provider: dict, reserve_output: int = None) -> bool:
    reserve_output = settings.get('reserve_output_tokens') if reserve_output is None else reserve_output
    return _total(messages) > get_budget(provider, reserve_output)


def fit_messages_to_window(messages: list, provider: dict, reserve_output: int = None) -> list:
    """Last resort: shudhu kete chhoto kore."""
    reserve_output = settings.get('reserve_output_tokens') if reserve_output is None else reserve_output
    budget = get_budget(provider, reserve_output)
    min_tokens = settings.get('min_context_tokens')
    msgs = [dict(m) for m in messages]
    before = _total(msgs)
    if before <= budget:
        return msgs

    for m in msgs:
        c = m.get('content', '')
        if m.get('role') == 'system' and SEARCH_ID in c:
            head, _, tail = c.partition(SEARCH_MARK)
            others = _total(msgs) - count_tokens(c)
            allow = max(min_tokens, budget - others - count_tokens(tail) - 50)
            m['content'] = fit_to_budget(head, allow) + "\n\n" + SEARCH_MARK + tail

    while _total(msgs) > budget and len(msgs) > 2:
        idx = next((i for i, m in enumerate(msgs[:-1]) if m.get('role') != 'system'), None)
        if idx is None:
            break
        msgs.pop(idx)

    if _total(msgs) > budget:
        last = msgs[-1]
        head, sep, tail = split_question(last['content'])
        if sep:
            allow = max(min_tokens, budget - (_total(msgs) - count_tokens(head)) - 50)
            last['content'] = fit_to_budget(head, allow) + sep + tail

    print(f"[Context Fit] {provider.get('model_name')}: {before} -> {_total(msgs)} tokens (budget {budget})")
    return msgs


def call_llm(provider: dict, messages: list, timeout_cap: int = None) -> str:
    timeout_cap = settings.get('llm_call_timeout_cap') if timeout_cap is None else timeout_cap
    default_timeout = settings.get('llm_call_timeout_default')
    t = min(int(provider.get('timeout') or default_timeout), timeout_cap)
    r = requests.post(provider['endpoint'],
                      json={"model": provider['model_name'], "messages": messages,
                            "temperature": 0.0, "stream": False},
                      timeout=(10, t))
    r.raise_for_status()
    return strip_reasoning(r.json()['choices'][0]['message'].get('content', ''))


def generate_text(provider: dict, messages: list) -> str:
    """Stream kore pura answer string return kore (UI te pathay na)."""
    default_stream_timeout = settings.get('llm_stream_timeout_default')
    t = int(provider.get('timeout') or default_stream_timeout)
    out = []
    with requests.post(provider['endpoint'],
                       json={"model": provider['model_name'], "messages": messages, "stream": True},
                       stream=True, timeout=(min(20, t), t)) as r:
        if r.status_code != 200:
            print(f"[generate_text] HTTP {r.status_code}")
            return ""
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8')
            if not line.startswith('data: '):
                continue
            data = line[6:]
            if data == '[DONE]':
                break
            try:
                out.append(json.loads(data)['choices'][0].get('delta', {}).get('content', '') or '')
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    return strip_reasoning("".join(out))


def condense_context(text: str, question: str, provider: dict, target_tokens: int) -> str:
    """Map-reduce: chunk e bhenge protita chunk theke relevant fact tole."""
    cur = text
    max_rounds = settings.get('max_rounds')
    max_chunks = settings.get('max_chunks')
    condense_reserve = settings.get('condense_reserve_output_tokens')
    condense_overhead = settings.get('condense_prompt_overhead_tokens')
    condense_min_chunk = settings.get('condense_min_chunk_tokens')

    for rnd in range(max_rounds):
        if count_tokens(cur) <= target_tokens:
            return cur
        chunk_budget = max(condense_min_chunk, get_budget(provider, reserve_output=condense_reserve) - count_tokens(question) - condense_overhead)
        chunks = pack_chunks(cur, chunk_budget)[:max_chunks]
        print(f"✂️ [CHUNKING & CONDENSING] Round {rnd+1}: Context has {count_tokens(cur)} tokens -> Split into {len(chunks)} chunk(s) (Target: {target_tokens} tokens)")

        def extract(c):
            prompt = settings.render('prompt_extract', question=question, text=c)
            try:
                return call_llm(provider, [{"role": "user", "content": prompt}])
            except Exception as e:
                print(f"[Condense] chunk failed: {e}")
                return ""

        notes = [n for n in map_chunks(chunks, extract) if n and not n.strip().upper().startswith("NONE")]
        if not notes:
            return ""
        cur = "\n\n".join(notes)
        print(f"✅ [CHUNKING ROUND {rnd+1} DONE] Condensed to {count_tokens(cur)} tokens.")
    return fit_to_budget(cur, target_tokens)


def prepare_messages(messages: list, provider: dict, question: str, reserve_output: int = None) -> list:
    """Fit hole kichu kore na. Na hole: history kom -> search condense -> file condense -> trim."""
    reserve_output = settings.get('reserve_output_tokens') if reserve_output is None else reserve_output
    budget = get_budget(provider, reserve_output)
    min_tokens = settings.get('min_context_tokens')
    keep_history = settings.get('history_keep_when_condensing')
    msgs = [dict(m) for m in messages]
    if _total(msgs) <= budget:
        return msgs

    non_sys = [i for i, m in enumerate(msgs) if m.get('role') != 'system']
    if len(non_sys) > keep_history:
        keep = set(non_sys[-keep_history:])
        msgs = [m for i, m in enumerate(msgs) if m.get('role') == 'system' or i in keep]

    if _total(msgs) > budget:
        for m in msgs:
            c = m.get('content', '')
            if m.get('role') == 'system' and SEARCH_ID in c:
                head, _, tail = c.partition(SEARCH_MARK)
                others = _total(msgs) - count_tokens(c)
                allow = max(min_tokens, budget - others - count_tokens(tail) - 50)
                m['content'] = condense_context(head, question, provider, allow) + "\n\n" + SEARCH_MARK + tail

    if _total(msgs) > budget:
        last = msgs[-1]
        head, sep, tail = split_question(last['content'])
        if sep:
            allow = max(min_tokens, budget - (_total(msgs) - count_tokens(head)) - 50)
            last['content'] = condense_context(head, question, provider, allow) + sep + tail

    return fit_messages_to_window(msgs, provider, reserve_output)


def clean_search_query(raw_query: str) -> str:
    """
    Cleans conversational boilerplate, instructions, and dates from a raw user prompt
    to construct a crisp, high-precision search engine query.
    """
    if not raw_query:
        return ""
    q = re.sub(r'^Current date:\s*\d{4}-\d{2}-\d{2}\.\s*', '', raw_query, flags=re.IGNORECASE).strip()

    prefixes = [
        r'^(?:please\s+)?(?:can|could)\s+you\s+(?:please\s+)?(?:search\s+(?:the\s+web\s+)?(?:for\s+|and\s+)?)?',
        r'^please\s+(?:search\s+(?:the\s+web\s+)?(?:for\s+|and\s+)?)?',
        r'^search\s+(?:the\s+web\s+)?(?:for\s+|and\s+)?',
        r'^(?:tell\s+me\s+(?:about\s+)?|find\s+(?:out\s+)?|give\s+me\s+|explain\s+)',
        r'^(?:what|who|where|when|why|how)\s+(?:is|are|was|were)\s+(?:the\s+)?'
    ]
    cleaned = q
    changed = True
    while changed:
        before = cleaned
        for p in prefixes:
            candidate = re.sub(p, '', cleaned, flags=re.IGNORECASE).strip()
            if candidate and len(candidate) >= 3:
                cleaned = candidate
        changed = (cleaned != before)

    # Remove trailing question marks or punctuation
    cleaned = re.sub(r'[\?\.!]+$', '', cleaned).strip()
    return cleaned if len(cleaned) >= 3 else q



def determine_query_evidence_need(user_prompt: str, has_active_files: bool = False) -> tuple:
    """
    Dynamically determines if external web search retrieval is beneficial for the query.
    Returns (needs_search: bool, strategy_reason: str).
    Ensures search is not forced for self-contained math, pure code syntax, greetings, or file-only tasks.
    """
    if not user_prompt:
        return False, "empty_query"

    q = user_prompt.strip().lower()

    # 1. Simple conversational greetings / thanks / identity
    conversational_patterns = [
        r'^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening))\b',
        r'^(?:thank\s+you|thanks|thx)\b',
        r'^(?:who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do)\b'
    ]
    if any(re.match(p, q) for p in conversational_patterns):
        return False, "conversational_greeting"

    # 2. Pure mathematical calculations, arithmetic, or numerical formulas
    math_patterns = [
        r'^(?:what\s+is\s+)?(?:\d+\.?\d*\s*[\+\-\*\/\^\%]\s*\d+\.?\d*)',
        r'^(?:what\s+is\s+|calculate\s+)?\d+(?:\.\d+)?\s*(?:percent|\%)\s*of\s*\d+',
        r'^(?:calculate|compute|solve|simplify)\s+[\d\s\+\-\*\/\%\^\(\)\.\=a-zA-Z]+$',
        r'^(?:what\s+is\s+)?the\s+square\s+root\s+of\s+\d+'
    ]
    if any(re.search(p, q) for p in math_patterns):
        return False, "self_contained_math"

    # 3. Pure programming language syntax / algorithm without external entity lookup
    code_syntax_patterns = [
        r'^(?:write|implement|create)\s+(?:a\s+)?(?:python|javascript|typescript|c\+\+|java|rust|go|c#)?\s*(?:function|script|code|algorithm|regex)\s+(?:to|for)\s+[a-zA-Z\s_]+$',
        r'^(?:how\s+to\s+)?(?:reverse|sort|traverse|flatten|filter|map)\s+(?:a\s+)?(?:string|array|list|linked\s+list|tree|dict|dictionary)\s+in\s+[a-zA-Z\+\#]+'
    ]
    if any(re.search(p, q) for p in code_syntax_patterns):
        return False, "self_contained_code"

    # 4. Explicit file-only request when files are attached
    if has_active_files:
        file_only_indicators = [
            'this file', 'the file', 'attached file', 'this document', 'the document',
            'this pdf', 'the pdf', 'this csv', 'the sheet', 'summarize this', 'analyze this'
        ]
        if any(ind in q for ind in file_only_indicators) and 'search the web' not in q:
            return False, "file_context_primary"

    # 5. Default for real-world knowledge, external facts, current events, research
    return True, "external_evidence_needed"


def resolve_search_coreference(user_prompt: str, history: list) -> str:
    """
    Enhances a follow-up query with contextual nouns from previous conversation turns
    when the user query uses pronouns ('it', 'he', 'she', 'they', 'this', 'that', 'its')
    or elided phrases ('when was it built', 'second term', 'how many goals').
    """
    if not user_prompt or not history:
        return user_prompt

    q = user_prompt.strip()
    q_low = q.lower()

    coreference_markers = [
        r'\b(?:it|its|he|his|him|she|her|they|their|them|this|that|these|those)\b',
        r'^(?:when|where|who|why|how\s+many|how\s+much)\s+(?:was|did|is|are|were)\s+(?:it|he|she|they|this)\b',
        r'^(?:and\s+)?(?:what\s+about|how\s+about)\s+'
    ]

    has_marker = any(re.search(m, q_low) for m in coreference_markers)
    is_short_followup = len(q.split()) <= 4 and re.match(r'^(?:when|where|who|how|why|what|which)\b', q_low)

    if not (has_marker or is_short_followup):
        return user_prompt

    # Extract the most recent user turn from history
    last_user_turn = None
    for msg in reversed(history):
        if msg.get('role') == 'user' and msg.get('content'):
            last_user_turn = msg['content'].strip()
            break

    if not last_user_turn:
        return user_prompt

    # Clean conversational lead-ins from the previous user turn to find the main subject
    prev_cleaned = clean_search_query(last_user_turn)
    words = prev_cleaned.split()
    if not words:
        return user_prompt

    subject = " ".join(words[:4])
    if subject.lower() in q_low:
        return user_prompt

    return f"{subject} {user_prompt}"


def verify_number_consistency(answer_text: str, reference_text: str, user_query: str = "") -> tuple:
    """
    Checks whether specific numbers, percentages, or metrics cited in the answer
    exist in the provided reference snippets or the user query itself.
    Returns (found_ratio: float, unverified_nums: list).
    """
    if not answer_text or not reference_text:
        return 1.0, []
    
    # Match numbers with 2+ digits or percentages (e.g., 2024, 6.5%, 1500)
    raw_nums = re.findall(r'\b\d[\d,.]*(?:%|\b)', answer_text)
    # Filter out trivial single digits (0-9) or standard markdown bullet indices
    meaningful = [n.replace(',', '').strip() for n in raw_nums if re.search(r'\d{2,}|%', n)]
    if not meaningful:
        return 1.0, []

    combined_ref = (reference_text + " " + (user_query or "")).replace(',', '')
    unverified = [n for n in set(meaningful) if n not in combined_ref]
    ratio = (len(meaningful) - len(unverified)) / len(meaningful)
    return ratio, unverified


def verify_temporal_consistency(query: str, answer_text: str, current_year: int) -> tuple:
    """
    Detects if the user asked a current/latest question but the answer asserts an outdated past year as current.
    Returns (is_outdated: bool, issue_description: str).
    """
    if not query or not answer_text:
        return False, ""
    
    q_low = query.lower()
    time_keywords = ['today', 'current', 'currently', 'latest', 'now', 'recent', str(current_year)]
    if any(k in q_low for k in time_keywords):
        patterns = [
            r'\bas of (?:[A-Za-z]+\s+)?(20\d{2})\b',
            r'\bcurrently in (20\d{2})\b',
            r'\bstatus in (20\d{2})\b',
            r'\bthe current year is (20\d{2})\b'
        ]
        for pat in patterns:
            for ym in re.finditer(pat, answer_text, re.IGNORECASE):
                cited_year = int(ym.group(1))
                if cited_year < current_year:
                    return True, f"Answer cites outdated year {cited_year} as current context (current year: {current_year})"
    return False, ""


def is_time_sensitive_query(user_prompt: str) -> bool:
    """
    Returns True if the query explicitly asks for current, recent, live, or time-sensitive data.
    """
    if not user_prompt:
        return False
    q_low = user_prompt.lower()
    time_keywords = [
        'today', 'current', 'currently', 'latest', 'now', 'recent',
        'live', 'stock', 'weather', 'score', 'news', 'price',
        'this week', 'this month', 'this year'
    ]
    return any(re.search(rf'\b{re.escape(k)}\b', q_low) for k in time_keywords)


def rank_and_filter_search_results(query: str, raw_results: list, max_results: int = 6) -> list:
    """
    Ranks and filters raw search engine results to discard low-quality, boilerplate,
    or error snippets (e.g., 'Access Denied', 'Cloudflare', 'Just a moment...', empty text)
    and sorts by query keyword relevance density.
    """
    if not raw_results:
        return []

    junk_patterns = [
        r'access denied', r'attention required', r'cloudflare', r'verify you are human',
        r'just a moment', r'enable javascript', r'sign in', r'robot or human',
        r'403 forbidden', r'404 not found', r'browse thousands of titles',
        r'read full articles, watch videos', r'cookies to continue'
    ]

    clean_q = clean_search_query(query).lower()
    q_words = [w for w in re.findall(r'\b[a-zA-Z0-9]{3,}\b', clean_q) if w not in (
        'what', 'when', 'where', 'which', 'who', 'how', 'the', 'and', 'for', 'about'
    )]

    scored = []
    seen_urls = set()

    for r in raw_results:
        body = (r.get("body") or r.get("snippet") or "").strip()
        title = (r.get("title") or "").strip()
        url = (r.get("url") or r.get("href") or "").strip()

        if not url or url in seen_urls:
            continue
        if len(body) < 25 and len(title) < 5:
            continue

        b_low = body.lower()
        t_low = title.lower()
        comb_low = f"{t_low} {b_low}"
        if any(re.search(p, comb_low) for p in junk_patterns):
            continue

        score = 0.0
        for w in q_words:
            if w in t_low:
                score += 2.0
            if w in b_low:
                score += 1.0

        score += min(len(body) / 200.0, 1.0)
        if "wikipedia.org" in url:
            score += 2.5

        seen_urls.add(url)
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:max_results]]



