# gemini_client.py
from __future__ import annotations
import os, json, re, time
from typing import List, Dict, Tuple, Optional
import requests

class GeminiNotAvailable(Exception): ...
class GeminiBadResponse(Exception): ...

# --------------------- Validators / helpers ---------------------
_FUNCTION_WORDS = {
    # articles/determiners
    "the","a","an","this","that","these","those","some","any","no","each","every","either","neither",
    # pronouns (personal/possessive/demonstrative/relative/indefinite)
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","her","its","our","their","mine","yours","hers","ours","theirs",
    "who","whom","whose","which","that","someone","something","anyone","anything","everyone","everything",
    # auxiliaries/modals/copulas
    "am","is","are","was","were","be","been","being",
    "do","does","did","have","has","had",
    "will","would","shall","should","can","could","may","might","must",
    # negation/particles
    "not","n't","to",
    # prepositions
    "of","in","on","at","from","to","with","for","by","about","as","into","through",
    "during","before","after","above","below","over","under","between","among","around",
    "near","inside","outside","without","within","across","behind","beyond","upon","off",
    # conjunctions
    "and","or","but","so","because","although","though","if","while","when","than",
    # minor fillers
    "please",
}
_WORD_RE = re.compile(r"[A-Za-z’']+")

def _canon_token(s: str) -> str:
    """Lowercase, strip brackets/possessive to compare content tokens consistently."""
    s = (s or "").strip().lower().strip("[]")
    s = re.sub(r"(’s|'s)$", "", s)
    return s

def _extract_words(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())

def _covers_nonspelled(sentence: str, tokens: List[str], spelled_idx: set[int]) -> bool:
    """All non-spelled tokens must appear in the sentence (case-insens., allow possessive)."""
    low = (sentence or "").lower()
    for i, t in enumerate(tokens):
        if i in spelled_idx:  # may be corrected; skip exact coverage check
            continue
        t_low = _canon_token(t)
        if not t_low:
            continue
        if (t_low in low) or (t_low + "’s" in low) or (t_low + "'s" in low):
            continue
        return False
    return True

def _violates_connector_only(sentence: str,
                             tokens: List[str],
                             spelled_idx: set[int],
                             corrected_tokens: Optional[List[str]]) -> bool:
    """
    True if the sentence adds content words beyond:
      - all non-spelled tokens as-is,
      - corrected versions of spelled tokens (if provided; else original spelled tokens),
      - allowed function words and contraction shards.
    """
    sent_words = _extract_words(sentence)
    allowed: set[str] = set()

    # non-spelled tokens (verbatim)
    for i, t in enumerate(tokens):
        if i in spelled_idx: continue
        ct = _canon_token(t)
        if ct: allowed.add(ct)

    # spelled tokens → allow corrected values if available, else original spelled token
    if corrected_tokens and len(corrected_tokens) == len(tokens):
        for i, t in enumerate(corrected_tokens):
            if i in spelled_idx:
                ct = _canon_token(t)
                if ct: allowed.add(ct)
    else:
        for i, t in enumerate(tokens):
            if i in spelled_idx:
                ct = _canon_token(t)
                if ct: allowed.add(ct)

    for w in sent_words:
        cw = _canon_token(w)
        if not cw: 
            continue
        if cw in allowed: 
            continue
        if cw in _FUNCTION_WORDS: 
            continue
        if cw in {"m","re","ve","ll","d","s"}:  # contraction shards
            continue
        # any other content word is a violation
        return True
    return False

# --------------------- Gemini Client ---------------------
class GeminiFormatter:
    """
    Sentence former for ISL:
      • INPUT: tokens + spelled_indices (indices in tokens that came from fingerspelling).
      • Gemini may correct spelling ONLY for tokens at spelled_indices; all others are fixed.
      • Gemini may add ONLY function/connecting words; MUST NOT add new content words.
      • Returns JSON with 'sentence', 'used_tokens', 'corrected_tokens' (same length as tokens).
      • Client validates and retries (strict mode, endpoint variants) if constraints are violated.
    """

    # -------- Project description + rules sent verbatim in the request --------
    _BASE_RULES = (
        "PROJECT CONTEXT:\n"
        "You assist an Indian Sign Language (ISL) recognition system that converts recognized signs into English.\n"
        "There are two token sources:\n"
        " • Dynamic signs: common words from a fixed ISL vocabulary (e.g., pronouns, actions, kinship terms).\n"
        " • Static fingerspelling: letter-by-letter sequences used for names and rare words; the system composes these into a token.\n\n"
        "TASK:\n"
        "Given the recognized tokens, produce exactly ONE fluent, grammatical English sentence that uses ALL content tokens.\n"
        "CRITICAL RULES:\n"
        " 1) Do NOT change the meaning of any non-spelled token.\n"
        " 2) You MAY perform spelling correction ONLY for tokens whose indices are listed in 'spelled_indices'.\n"
        " 3) You MAY add ONLY connecting/function words (articles, auxiliaries, copulas, pronouns, prepositions, conjunctions, negation)\n"
        "    as needed for grammar. Do NOT introduce any new content nouns/verbs/adjectives/adverbs beyond the tokens and any corrected\n"
        "    form of a spelled token. Prefer the SHORTEST grammatical, syntactic, and semantic sentence that makes sense overall.\n"
        " 4) Treat bracketed tokens like [Aditya] as proper names; use normal capitalization for sentence start and proper names.\n\n"
        "OUTPUT (JSON ONLY, no prose):\n"
        "{\n"
        '  "sentence": str,                          # final sentence (single-best, properly capitalized, ends with . ? or !) \n'
        '  "used_tokens": [str, ...],                # echo which tokens you used (copy or corrected), same order as input\n'
        '  "corrected_tokens": [str, ...]            # same length as tokens; identity for non-spelled indices; corrected for spelled indices\n'
        "}\n"
    )

    _STRICT_APPEND = (
        "\nSTRICT MODE:\n"
        "- Never add any new content words not present in the input tokens (except corrected forms of spelled tokens).\n"
        "- If in doubt, choose the shortest grammatical structure using only function words to connect the tokens.\n"
        "- Keep non-spelled tokens unchanged."
    )

    def __init__(self, model_name: str="gemini-1.5-flash", api_key: Optional[str]=None, temperature: float=0.2):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise GeminiNotAvailable("GEMINI_API_KEY/GOOGLE_API_KEY not found; set env var or pass --gemini_key.")
        self.model = model_name
        self.temperature = float(temperature)
        self.base_v1  = "https://generativelanguage.googleapis.com/v1/models"
        self.base_v1b = "https://generativelanguage.googleapis.com/v1beta/models"

    # ---------------- internals ----------------
    def _parse_json_text(self, text: str) -> Dict:
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
            if m:
                return json.loads(m.group(0))
            raise GeminiBadResponse("Gemini did not return valid JSON text.")

    def _post(self, base: str, use_gen_cfg: bool, payload_text: str, timeout_s: float) -> Dict:
        url = f"{base}/{self.model}:generateContent?key={self.api_key}"
        payload = { "contents": [{ "role": "user", "parts": [{ "text": payload_text }] }] }
        if use_gen_cfg:
            payload["generationConfig"] = { "temperature": self.temperature, "maxOutputTokens": 60 }
        r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            try: err = r.json()
            except Exception: err = r.text
            raise GeminiBadResponse(f"HTTP {r.status_code} @ {url} :: {err}")
        return r.json()

    # ---------------- public API ----------------
    def build_prompt_text(self, tokens: List[str], spelled_indices: List[int], strict: bool=False) -> str:
        sys_prompt = self._BASE_RULES + (self._STRICT_APPEND if strict else "")
        user_payload = { "tokens": tokens, "spelled_indices": spelled_indices }
        return sys_prompt + "\n" + json.dumps(user_payload, ensure_ascii=False)

    def format_tokens(self,
                      tokens: List[str],
                      spelled_indices: Optional[List[int]] = None,
                      timeout_s: float = 12.0) -> Tuple[str, Dict]:
        """
        Returns (sentence, meta) where meta includes:
          - used_tokens
          - corrected_tokens
          - endpoint, strict, latency
        """
        if not tokens:
            raise GeminiBadResponse("No tokens provided.")
        spelled_indices = spelled_indices or []
        spelled_idx_set = set(int(i) for i in spelled_indices if 0 <= int(i) < len(tokens))

        def call_once(base: str, use_cfg: bool, prompt_text: str) -> Tuple[str, Dict]:
            data = self._post(base, use_cfg, prompt_text, timeout_s)
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            obj = self._parse_json_text(raw)
            if "sentence" not in obj or not isinstance(obj["sentence"], str):
                raise GeminiBadResponse("JSON missing 'sentence'.")
            sent = obj["sentence"].strip()
            used = obj.get("used_tokens", [])
            corrected = obj.get("corrected_tokens", None)

            # Normalize sentence
            if sent and sent[-1] not in ".?!": sent += "."
            sent = sent[0:1].upper() + sent[1:]
            sent = sent.replace(" ’s", "’s").replace(" 's", "'s")

            # Basic corrected_tokens shape requirement
            if corrected is not None and len(corrected) != len(tokens):
                raise GeminiBadResponse("'corrected_tokens' length must match the input tokens length.")

            return sent, {"used_tokens": used, "corrected_tokens": corrected}

        t0 = time.time()
        tries = [
            (False, self.base_v1,  True),
            (False, self.base_v1,  False),
            (True,  self.base_v1,  True),
            (True,  self.base_v1,  False),
            (False, self.base_v1b, True),
            (False, self.base_v1b, False),
            (True,  self.base_v1b, True),
            (True,  self.base_v1b, False),
        ]
        last_err: Optional[Exception] = None

        for strict, base, use_cfg in tries:
            try:
                prompt = self.build_prompt_text(tokens, list(spelled_idx_set), strict=strict)
                sentence, meta = call_once(base, use_cfg, prompt)

                # Hard checks:
                if not _covers_nonspelled(sentence, tokens, spelled_idx_set):
                    raise GeminiBadResponse("Output does not include all non-spelled tokens.")

                if _violates_connector_only(sentence, tokens, spelled_idx_set, meta.get("corrected_tokens")):
                    raise GeminiBadResponse("Output added new content words (beyond corrected spelled tokens).")

                meta.update({"endpoint": base, "strict": strict, "latency": time.time() - t0})
                return sentence, meta

            except Exception as e:
                last_err = e
                # try next strategy

        raise GeminiBadResponse(f"All attempts failed. Last error: {last_err}")
