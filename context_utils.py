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
