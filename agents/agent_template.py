from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
from collections import Counter, defaultdict

from openai import AsyncOpenAI
import tiktoken

from model import ModelProvider

# -----------------------------
# Helpers
# -----------------------------

_WORD_RE = re.compile(r"[0-9A-Za-z\u0370-\u03FF\u1F00-\u1FFF]+")
# "Anchor" = likely-unique identifier token patterns (IDs, codes, refs)
_ANCHOR_RE = re.compile(
    r"""
    (?:
        # Things like ABC-123, P-8813-Zeta, ID_42X, ref:XYZ-9A, Ω-42, ΦΔ_9A
        [0-9A-Za-z\u0370-\u03FF\u1F00-\u1FFF]{2,}[\-_][0-9A-Za-z\u0370-\u03FF\u1F00-\u1FFF]{2,} |
        [0-9A-Za-z\u0370-\u03FF\u1F00-\u1FFF]{2,}[\-_][0-9A-Za-z\u0370-\u03FF\u1F00-\u1FFF]{2,} |
        [A-Za-z\u0370-\u03FF\u1F00-\u1FFF]{2,}\d{2,}[0-9A-Za-z\u0370-\u03FF\u1F00-\u1FFF]* |
        \d{2,}[A-Za-z\u0370-\u03FF\u1F00-\u1FFF]{2,}[0-9A-Za-z\u0370-\u03FF\u1F00-\u1FFF]* |
        [0-9A-Za-z\u0370-\u03FF\u1F00-\u1FFF]{6,}  # long-ish alnum token
    )
    """,
    re.VERBOSE,
)

def _word_tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())

def _safe_lower(s: str) -> str:
    try:
        return s.lower()
    except Exception:
        return s

def _iter_char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> Iterable[str]:
    """
    Light char-n-gram generator (whitespace-normalized).
    Designed to be computed only on a small candidate set.
    """
    t = re.sub(r"\s+", " ", text.lower()).strip()
    if not t:
        return
    # Add boundaries to improve matching of short tokens
    t = f" {t} "
    L = len(t)
    for n in range(n_min, n_max + 1):
        if L < n:
            continue
        for i in range(0, L - n + 1):
            yield t[i : i + n]

def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    # sparse cosine similarity
    if not a or not b:
        return 0.0
    # iterate over smaller dict
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            dot += va * vb
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class _Chunk:
    filename: str
    chunk_id: int
    text: str
    word_tokens: Tuple[str, ...]
    source: str  # "chunk" or "anchor_window"


# -----------------------------
# Main Agent
# -----------------------------

class HybridRetrieveAgent(ModelProvider):
    """
    Enhanced retrieval agent (2nd version):

    Retrieval = Anchor-first + BM25 (multi-query RRF) + Char-n-gram rerank (+ optional dense rerank)
    Generation = Send top snippets to LLM, instruct "output only the final answer".

    ✅ Compliance:
      - ONLY uses prompt['context_data']['files'][*]['modified_content'] for retrieval.
      - Does NOT read PaulGrahamEssays from disk.
      - Does NOT read any test case files.
      - FIXED: Uses 'async with' for OpenAI client to prevent EventLoopClosedError.

    Notes:
      - Works out-of-the-box with pure Python + tiktoken + openai client.
      - If sentence_transformers is installed and a model is available locally, can optionally rerank with dense embeddings.
        (Auto-disabled if import/model load fails.)
    """

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)

        self.model_name = "ecnu-max"
        
        # --- FIX START: 保存参数，而不是直接创建 client ---
        # 避免在 __init__ 中创建长连接，改为在 evaluate_model 中按需创建
        self.client_args = {"api_key": api_key, "base_url": base_url}
        # --- FIX END ---

        # Tokenizer only for chunking + budget control
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        # Chunking
        self.chunk_size_tokens = 750
        self.chunk_overlap_tokens = 120

        # Candidate sizes (tune)
        self.bm25_top_n = 40
        self.final_top_k = 8
        self.per_file_cap = 3

        # Packing budget (leave headroom for instructions + answer)
        self.max_context_tokens = 5000
        self.max_answer_tokens = 64

        # Anchor windows (chars)
        self.anchor_window_chars = 1200  # total window approx ~2400 chars (±1200)

        # Optional dense rerank
        self.enable_dense_rerank = False
        self._dense_model = None  # lazy init

    # ---------- required interface ----------

    async def evaluate_model(self, prompt: Dict) -> str:
        context_data = prompt.get("context_data") or {}
        question = (prompt.get("question") or "").strip()

        files = context_data.get("files") or []
        if not files or not question:
            return ""

        # 1) Build token chunks for BM25
        chunks = self._build_chunks(files)

        # 2) Anchor-first windows
        anchors = self._extract_anchors(question)
        anchor_windows = self._anchor_windows(files, anchors)

        # 3) BM25 multi-query RRF over chunks
        bm25_candidates = self._bm25_rrf(chunks, question, top_n=self.bm25_top_n)

        # 4) Fuse + diversify
        candidates = self._fuse_candidates(anchor_windows, bm25_candidates)
        candidates = self._diversify_by_file(candidates, per_file_cap=self.per_file_cap)

        # 5) Char-n-gram rerank (cheap & strong for IDs/codes/encoded strings)
        reranked = self._char_ngram_rerank(candidates, question, top_k=self.final_top_k)

        # 6) Optional dense rerank (only if available; otherwise skipped)
        reranked = self._dense_rerank_if_available(reranked, question)

        # 7) Pack snippets
        packed_context = self._pack_chunks(reranked, token_budget=self.max_context_tokens)

        # 8) Ask LLM to output only ONE clean answer line
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precision extraction agent. "
                    "Use ONLY the snippets provided. "
                    "Output ONLY the final answer, as a single line. "
                    "Do NOT output any reasoning, labels (like 'Answer:'), quotes, punctuation, or extra words. "
                    "If the answer is a number, output digits only. "
                    "If the answer is a weekday, output exactly one of: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday. "
                    "If the answer is a code/string, copy it EXACTLY from the snippets."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Snippets:\n{packed_context}\n\n"
                    "Return ONLY the final answer on one line:"
                ),
            },
        ]

        try:
            # --- FIX START: 使用 async with 上下文管理器 ---
            # 这确保了请求完成后连接会被正确关闭，不会残留到 event loop 关闭之后
            async with AsyncOpenAI(**self.client_args) as client:
                resp = await client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=self.max_answer_tokens,
                    stop=["\n"],
                )
            print(question)
            print((resp.choices[0].message.content or "").strip())
            return (resp.choices[0].message.content or "").strip()
            # --- FIX END ---
        except Exception:
            # Fallback: return best snippet prefix to help local debug
            return (reranked[0].text[:300] if reranked else "").strip()

    def generate_prompt(self, **kwargs) -> Dict:
        return {"context_data": kwargs.get("context_data"), "question": kwargs.get("question")}

    def encode_text_to_tokens(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        if context_length is not None:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)

    # ---------- Indexing / chunking ----------

    def _build_chunks(self, files: List[Dict]) -> List[_Chunk]:
        chunks: List[_Chunk] = []
        step = max(1, self.chunk_size_tokens - self.chunk_overlap_tokens)

        for f in files:
            filename = str(f.get("filename", "unknown"))
            content = f.get("modified_content") or ""
            if not content.strip():
                continue

            toks = self.encode_text_to_tokens(content)
            if not toks:
                continue

            chunk_id = 0
            for start in range(0, len(toks), step):
                end = min(len(toks), start + self.chunk_size_tokens)
                text = self.decode_tokens(toks[start:end])

                # Safety cap by chars to avoid huge blocks
                if len(text) > 20000:
                    text = text[:20000]

                wtoks = tuple(_word_tokenize(text))
                if wtoks:
                    chunks.append(
                        _Chunk(
                            filename=filename,
                            chunk_id=chunk_id,
                            text=text,
                            word_tokens=wtoks,
                            source="chunk",
                        )
                    )
                    chunk_id += 1

                if end >= len(toks):
                    break

        return chunks

    # ---------- Anchor-first ----------

    def _extract_anchors(self, question: str) -> List[str]:
        """
        Extract likely-unique identifiers from the question.
        We keep anchors that are:
          - alnum and length>=6, or contain '-'/'_'
          - not purely digits
        """
        q = question.strip()
        raw = _ANCHOR_RE.findall(q)
        anchors = []
        seen = set()
        for a in raw:
            a = a.strip()
            if len(a) < 6 and ("-" not in a and "_" not in a):
                continue
            if a.isdigit():
                continue
            # drop very common words
            if a.lower() in {"because", "between", "without", "should", "answer", "needle", "context", "snippets"}:
                continue
            if a not in seen:
                anchors.append(a)
                seen.add(a)
        # prefer longer / more specific anchors first
        anchors.sort(key=lambda s: (-len(s), s))
        return anchors[:8]

    def _anchor_windows(self, files: List[Dict], anchors: List[str]) -> List[_Chunk]:
        """
        For each anchor, do a fast substring search in each file's modified_content.
        Extract a window around each hit as a candidate snippet.
        """
        if not anchors:
            return []

        out: List[_Chunk] = []
        for f in files:
            filename = str(f.get("filename", "unknown"))
            content = f.get("modified_content") or ""
            if not content:
                continue

            lower = content.lower()
            for a in anchors:
                al = a.lower()
                idx = lower.find(al)
                # also try exact-case find if lower find fails (rare)
                if idx < 0:
                    idx = content.find(a)
                if idx < 0:
                    continue

                start = max(0, idx - self.anchor_window_chars)
                end = min(len(content), idx + len(a) + self.anchor_window_chars)
                text = content[start:end]

                wtoks = tuple(_word_tokenize(text))
                if not wtoks:
                    continue

                out.append(
                    _Chunk(
                        filename=filename,
                        chunk_id=-len(out) - 1,  # negative ids for anchor windows
                        text=text,
                        word_tokens=wtoks,
                        source="anchor_window",
                    )
                )

        return out

    # ---------- BM25 + RRF fusion ----------

    def _bm25_rrf(self, chunks: List[_Chunk], question: str, top_n: int) -> List[_Chunk]:
        """
        Multi-query BM25 using Reciprocal Rank Fusion (RRF).
        Query set:
          - original question
          - quoted phrases
          - anchor tokens
          - keyword-only query (drop stop-ish tokens)
        """
        if not chunks:
            return []

        queries = self._make_subqueries(question)
        # Score accumulator
        rrf_scores: defaultdict[int, float] = defaultdict(float)
        k = 60.0  # RRF constant

        for q in queries:
            ranked = self._bm25_retrieve(chunks, q, top_k=min(top_n, 80))
            for rank, ch in enumerate(ranked, 1):
                # use id by index in original list: we need stable id
                # fallback: hash of (filename, chunk_id, source)
                idx = self._chunk_key_hash(ch)
                rrf_scores[idx] += 1.0 / (k + rank)

        # Map hash back to chunk objects (best one if collisions, very unlikely)
        best_by_hash: Dict[int, _Chunk] = {}
        for ch in chunks:
            best_by_hash[self._chunk_key_hash(ch)] = ch

        scored = [(score, best_by_hash[h]) for h, score in rrf_scores.items() if h in best_by_hash]
        scored.sort(key=lambda x: x[0], reverse=True)

        return [ch for _, ch in scored[:top_n]]

    def _make_subqueries(self, question: str) -> List[str]:
        qs = []
        q = question.strip()
        if q:
            qs.append(q)

        # quoted phrases
        qs.extend(self._extract_quoted_phrases(q))

        # anchors
        qs.extend(self._extract_anchors(q))

        # keyword-only query: keep alnum tokens length>=3 and numbers
        toks = _word_tokenize(q)
        keep = []
        for t in toks:
            if len(t) >= 3 or t.isdigit():
                keep.append(t)
        if keep:
            qs.append(" ".join(keep[:40]))

        # de-dup while preserving order
        out = []
        seen = set()
        for s in qs:
            s2 = " ".join(s.split())
            if not s2:
                continue
            key = s2.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s2)
        return out[:8]

    def _extract_quoted_phrases(self, text: str) -> List[str]:
        phrases = []
        for m in re.finditer(r'"([^"]+)"', text):
            phrases.append(m.group(1).strip())
        for m in re.finditer(r"'([^']+)'", text):
            phrases.append(m.group(1).strip())
        return [p for p in phrases if 2 <= len(p) <= 80]

    def _chunk_key_hash(self, ch: _Chunk) -> int:
        # stable-ish hash (avoid Python hash randomization across runs by using a manual hash)
        s = f"{ch.filename}||{ch.source}||{ch.chunk_id}"
        h = 2166136261
        for c in s:
            h ^= ord(c)
            h *= 16777619
            h &= 0xFFFFFFFF
        return int(h)

    def _bm25_retrieve(self, chunks: List[_Chunk], query: str, top_k: int) -> List[_Chunk]:
        q_tokens = _word_tokenize(query)
        if not q_tokens:
            return chunks[:top_k]

        # document frequencies
        df: defaultdict[str, int] = defaultdict(int)
        doc_lens: List[int] = []
        for ch in chunks:
            doc_lens.append(len(ch.word_tokens))
            for t in set(ch.word_tokens):
                df[t] += 1

        N = max(1, len(chunks))
        avgdl = (sum(doc_lens) / N) if doc_lens else 1.0

        # BM25 params
        k1, b = 1.5, 0.75

        q_unique = list(dict.fromkeys(q_tokens))
        idf: Dict[str, float] = {}
        for t in q_unique:
            n_q = df.get(t, 0)
            idf[t] = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))

        scored: List[Tuple[float, _Chunk]] = []
        for ch in chunks:
            tf = Counter(ch.word_tokens)
            dl = len(ch.word_tokens) or 1
            denom_const = k1 * (1.0 - b + b * dl / avgdl)

            score = 0.0
            for t in q_unique:
                f = tf.get(t, 0)
                if f <= 0:
                    continue
                score += idf[t] * (f * (k1 + 1.0)) / (f + denom_const)

            # phrase boost for short quoted phrases
            for phrase in self._extract_quoted_phrases(query):
                if phrase and phrase.lower() in ch.text.lower():
                    score += 2.0

            if score > 0:
                scored.append((score, ch))

        if not scored:
            return chunks[:top_k]

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ch for _, ch in scored[:top_k]]

    # ---------- Fusion / diversification ----------

    def _fuse_candidates(self, anchor_windows: List[_Chunk], bm25_candidates: List[_Chunk]) -> List[_Chunk]:
        """
        Combine anchor windows and bm25 candidates, then de-dup by (filename, normalized text hash).
        Anchor windows are kept first to prioritize exact-ID hits.
        """
        out: List[_Chunk] = []
        seen = set()

        def key(ch: _Chunk) -> Tuple[str, int]:
            # hash on first 400 chars to avoid heavy hashing and still de-dup near-identical windows
            s = (ch.text[:400] or "").strip()
            h = 2166136261
            for c in s:
                h ^= ord(c)
                h *= 16777619
                h &= 0xFFFFFFFF
            return (ch.filename, int(h))

        for ch in anchor_windows + bm25_candidates:
            k = key(ch)
            if k in seen:
                continue
            seen.add(k)
            out.append(ch)

        return out

    def _diversify_by_file(self, chunks: List[_Chunk], per_file_cap: int) -> List[_Chunk]:
        out: List[_Chunk] = []
        counts: defaultdict[str, int] = defaultdict(int)
        for ch in chunks:
            if counts[ch.filename] >= per_file_cap:
                continue
            out.append(ch)
            counts[ch.filename] += 1
        return out

    # ---------- Char-n-gram rerank ----------

    def _char_ngram_rerank(self, candidates: List[_Chunk], question: str, top_k: int) -> List[_Chunk]:
        """
        Build TF-IDF over char n-grams on the candidate set only,
        then cosine-similarity rerank. Very effective for:
          - IDs / codes
          - base64/hex-like strings
          - exact substrings
        """
        if not candidates:
            return []

        # Compute document frequencies over candidate set
        df: defaultdict[str, int] = defaultdict(int)
        doc_ngrams: List[Counter] = []

        for ch in candidates:
            c = Counter(_iter_char_ngrams(ch.text))
            doc_ngrams.append(c)
            for ng in c.keys():
                df[ng] += 1

        N = len(candidates)

        # Build query vector
        q_counts = Counter(_iter_char_ngrams(question))
        q_vec: Dict[str, float] = {}
        for ng, tf in q_counts.items():
            # tf-idf
            idf = math.log(1.0 + (N + 1.0) / (df.get(ng, 0) + 1.0))
            q_vec[ng] = (1.0 + math.log(tf)) * idf

        # Score each doc
        scored: List[Tuple[float, _Chunk]] = []
        for ch, counts in zip(candidates, doc_ngrams):
            d_vec: Dict[str, float] = {}
            for ng, tf in counts.items():
                idf = math.log(1.0 + (N + 1.0) / (df.get(ng, 0) + 1.0))
                d_vec[ng] = (1.0 + math.log(tf)) * idf
            sim = _cosine_sparse(q_vec, d_vec)

            # small boost: anchor_window source
            if ch.source == "anchor_window":
                sim += 0.03

            scored.append((sim, ch))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ch for _, ch in scored[:top_k]]

    # ---------- Optional dense rerank ----------

    def _dense_rerank_if_available(self, chunks: List[_Chunk], question: str) -> List[_Chunk]:
        """
        Optional dense rerank with sentence_transformers if installed and model is loadable.
        Auto-disabled if any failure occurs.
        """
        if not self.enable_dense_rerank or not chunks:
            return chunks

        model = self._get_dense_model()
        if model is None:
            return chunks

        try:
            import numpy as np  # optional
        except Exception:
            return chunks

        try:
            # Embed query and docs
            q_emb = model.encode([question], normalize_embeddings=True)[0]
            texts = [c.text[:4000] for c in chunks]  # cap to speed
            d_embs = model.encode(texts, normalize_embeddings=True)
            sims = (d_embs @ q_emb).tolist()
            ranked = sorted(zip(sims, chunks), key=lambda x: x[0], reverse=True)
            return [c for _, c in ranked]
        except Exception:
            # disable permanently for this run
            self.enable_dense_rerank = False
            return chunks

    def _get_dense_model(self):
        if self._dense_model is not None:
            return self._dense_model

        # Lazy import/load
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            self.enable_dense_rerank = False
            return None

        # Try a small model; if not present locally, this may fail (no internet on some graders).
        # Users can vendor/cache a model path and set ST_MODEL_PATH env.
        import os
        model_name_or_path = os.getenv("ST_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")

        try:
            self._dense_model = SentenceTransformer(model_name_or_path)
            return self._dense_model
        except Exception:
            self.enable_dense_rerank = False
            self._dense_model = None
            return None

    # ---------- Packing ----------

    def _pack_chunks(self, chunks: List[_Chunk], token_budget: int) -> str:
        """
        Pack only a few top snippets to keep context focused and reduce hallucinations.
        """
        parts: List[str] = []
        used = 0

        # Keep only top 5 snippets (already ranked)
        for rank, ch in enumerate(chunks[:5], 1):
            # Light metadata helps debugging but keep it short
            block = f"[#{rank} file={ch.filename} src={ch.source}]\n{ch.text}\n"
            bt = len(self.encode_text_to_tokens(block))
            if used + bt > token_budget:
                break
            parts.append(block)
            used += bt

        return "\n".join(parts).strip()

