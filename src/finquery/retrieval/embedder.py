"""Dense embeddings with sentence-transformers.

bge models are trained with an instruction in front of short queries
("Represent this sentence for searching relevant passages: ") and nothing in
front of passages. Forgetting the instruction is a classic silent drop in recall.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class Embedder:
    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 32,
                 device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> Embedder:
        cfg = settings["embedding"]
        return cls(cfg["model"], cfg.get("query_prefix", ""), cfg.get("batch_size", 32),
                   cfg.get("device", "cpu"))

    @property
    def dimension(self) -> int:
        # renamed in recent sentence-transformers versions; support both
        getter = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension
        return getter()

    @property
    def max_tokens(self) -> int:
        return self.model.max_seq_length

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Number of tokens of each text; above max_tokens the end is cut off."""
        return [len(ids) for ids in self.model.tokenizer(texts, add_special_tokens=True)["input_ids"]]

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=show_progress, convert_to_numpy=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(self.query_prefix + query, normalize_embeddings=True,
                                 convert_to_numpy=True)
