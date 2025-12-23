import math
import re
from typing import Dict, List, Optional

import numpy as np
import tiktoken
from openai import AsyncOpenAI

from model import ModelProvider


class RetrievalAgent(ModelProvider):
    """A retrieval-focused agent for multi-document needle tasks.

    The agent performs lightweight keyword-based retrieval across all modified files,
    extracts the most relevant chunks within a token budget, and crafts a concise
    prompt that encourages grounded answers.
    """

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.model_name = "ecnu-max"
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        # Retrieval configuration
        self.chunk_size = 320
        self.chunk_overlap = 64
        self.max_context_tokens = 8000
        self.min_chunks = 6

    async def evaluate_model(self, prompt: Dict) -> str:
        context_data = prompt["context_data"]
        question = prompt["question"]

        keywords = self._extract_keywords(question)
        ranked_chunks = self._rank_chunks(context_data["files"], keywords)
        selected_chunks = self._select_chunks_within_budget(ranked_chunks)

        context_blocks = "\n\n".join(
            [f"File: {c['filename']}\nExcerpt:\n{c['text']}" for c in selected_chunks]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a meticulous research assistant. Base answers ONLY on the given excerpts. "
                    "Quote the key sentences verbatim when possible and avoid explanations, bullet points, "
                    "or prefixed phrases. Respond with the final answer text only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context excerpts (most relevant first):\n{context_blocks}\n\n"
                    f"Question: {question}\n\n"
                    "Return the direct answer using only the excerpts above. If multiple excerpts are relevant, "
                    "combine them into one concise statement without adding commentary."
                ),
            },
        ]

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=300,
        )

        return response.choices[0].message.content

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[\w']+", text.lower())
        stopwords = {
            "the",
            "a",
            "an",
            "of",
            "and",
            "to",
            "in",
            "on",
            "for",
            "with",
            "is",
            "are",
            "was",
            "were",
            "what",
            "which",
            "who",
            "whom",
            "when",
            "where",
            "why",
            "how",
        }
        return [t for t in tokens if t not in stopwords]

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks based on token count."""
        token_ids = self.encode_text_to_tokens(text)
        if len(token_ids) <= self.chunk_size:
            return [text]

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for start in range(0, len(token_ids), step):
            end = start + self.chunk_size
            chunk_tokens = token_ids[start:end]
            chunks.append(self.decode_tokens(chunk_tokens))
            if end >= len(token_ids):
                break
        return chunks

    def _sentence_windows(self, text: str, max_sentences: int = 3) -> List[str]:
        """Create multi-sentence windows to keep context intact."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        windows = []
        for i in range(len(sentences)):
            for span in range(1, max_sentences + 1):
                segment = " ".join(sentences[i : i + span]).strip()
                if segment:
                    windows.append(segment)
        return windows or [text]

    def _chunk_score(self, chunk: str, keywords: List[str]) -> float:
        if not keywords:
            return 0.0

        words = re.findall(r"[\w']+", chunk.lower())
        if not words:
            return 0.0

        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

        matched = [k for k in keywords if word_counts.get(k, 0) > 0]
        coverage = len(set(matched)) / max(1, len(set(keywords)))
        density = sum(word_counts.get(k, 0) for k in keywords) / max(1, len(words))
        return float(2.5 * coverage + density)

    def _rank_chunks(self, files: List[Dict], keywords: List[str]) -> List[Dict]:
        """Rank contextual windows from each file by keyword overlap."""
        ranked = []
        for file_data in files:
            windows = self._sentence_windows(file_data["modified_content"])
            if keywords:
                windows = [w for w in windows if any(k in w.lower() for k in keywords)] or windows

            for window in windows:
                score = self._chunk_score(window, keywords)
                ranked.append(
                    {
                        "filename": file_data["filename"],
                        "text": window,
                        "score": score,
                        "token_count": len(self.encode_text_to_tokens(window)),
                    }
                )

        ranked.sort(key=lambda c: (c["score"], -c["token_count"]), reverse=True)
        return ranked

    def _select_chunks_within_budget(self, ranked_chunks: List[Dict]) -> List[Dict]:
        """Keep high-value chunks while covering multiple files."""
        selected = []
        budget = self.max_context_tokens

        # Prefer the best chunk from each file first
        best_by_file = {}
        for chunk in ranked_chunks:
            if chunk["filename"] not in best_by_file:
                best_by_file[chunk["filename"]] = chunk

        prioritized = list(best_by_file.values()) + ranked_chunks
        seen = set()

        for chunk in prioritized:
            key = (chunk["filename"], chunk["text"])
            if key in seen:
                continue
            seen.add(key)

            if chunk["token_count"] <= budget or not selected:
                selected.append(chunk)
                budget -= chunk["token_count"]

            if len(selected) >= self.min_chunks and budget <= 0:
                break

        return selected

    def generate_prompt(self, **kwargs) -> Dict:
        return {
            "context_data": kwargs.get("context_data"),
            "question": kwargs.get("question"),
        }

    def encode_text_to_tokens(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        if context_length:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)
