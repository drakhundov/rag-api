"""Splits text based on windowed (left margin + right margin) cosine similarity."""
import logging
from typing import List, Dict, Tuple
import re
import os

import numpy as np
from langchain_core.documents import Document

from ragsuite.core.types import SemanticTextSplitterConfig
from ragsuite.store import CacheStore
from ragsuite.utilities import docutils, err, vector

logger: logging.Logger = logging.getLogger(__name__)


# ! BUG: Caching does not account for configuration.


# Interface: ports/TextSplitter
class SemanticTextSplitter:
    _SPLIT_SENTENCES_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        bufsz: int = 2,
        breakpoint_percentile_threshold: float = .95,
    ):
        """
        Args:
            bufsz:
                how many neighboring sentences to use when looking for semantic difference.
            breakpoint_percentile_threshold:
                what percentage is the threshold, below which is put for split.
        """
        logger.debug("Starting SemanticTextSplitter initialization")

        self.bufsz = bufsz
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold

        self.config = SemanticTextSplitterConfig(
            bufsz=self.bufsz,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold
        )

        self.cached_splits_mng = CacheStore(svc_id="semantic-text-splitter")

        logger.debug("SemanticTextSplitter initialized")

    def split(self, docs: List[Document]) -> List[Document]:
        """
        Chunks each Document into semantically-cohesive pieces using sentence-level embedding distances.

        Args:
            docs:
                list of documents to split
        Returns:
                list of documents (len(ret) >= len(docs))
        """
        logger.debug(f"Splitting {len(docs)} documents")
        all_chunks: List[Document] = []
        for doc in docs:
            # * Try to retrieve chunks from cache in case they were split before.
            hit, cached_data = self.retrieve_split_from_cache(doc, self.config.to_dict())
            if hit:
                cached_splits = cached_data["splits"]
                all_chunks.extend(cached_splits)
                logger.debug("Hit: %s", doc.page_content[:40])
                continue
            logger.debug("No hit, splitting the doc: %s", doc.page_content[:40])
            sentences = self.split_into_sentences(doc.page_content)
            if len(sentences) <= 1:
                # Nothing to chunk; keep as-is.
                all_chunks.append(doc)
                continue

            # Build local context windows per sentence and embed.
            windowed = self.windowed_concat(sentences, self.bufsz)
            embs_lst = vector.embed_texts(windowed)  # TODO write an embedder

            # Calculate distances between adjacent windows.
            distances: List[float] = []
            for i in range(embs_lst.shape[0] - 1):
                d = 1.0 - self.calc_cosine_similarity(embs_lst[i], embs_lst[i + 1])
                distances.append(float(d))
            if not distances:
                all_chunks.append(doc)
                continue

            # Determine breakpoints by percentile threshold.
            # Basically, if the distance lies at the top N %
            # of the list, we consider it to be too large and
            # seperate the sentences at that particular point.
            threshold = float(
                np.percentile(distances, self.breakpoint_percentile_threshold)
            )
            breakpoints = [i for i, d in enumerate(distances) if d > threshold]

            # Create chunk strings and wrap as Documents, inheriting metadata
            chunk_texts = SemanticTextSplitter.break_sentences_at_breakpoints(
                sentences, breakpoints
            )
            doc_splits: List[Document] = []
            for idx, chunk_text in enumerate(chunk_texts):
                doc_splits.append(
                    Document(
                        page_content=chunk_text,
                        metadata=SemanticTextSplitter.inherit_metadata(doc, idx),
                    )
                )
            all_chunks.extend(doc_splits)
            doc_hash = docutils.compute_doc_hash(doc)
            payload = {"conf": self.get_conf(), "doc": self._serialize_docs(doc_splits)}
            self.cached_splits_mng.set(
                cache_key=os.path.join("cached_splits", doc_hash),
                data={"splitter": payload}
            )
        self.session_store.dump(session_data={"splits": all_chunks})
        return all_chunks

    def retrieve_split_from_cache(
        self, doc: Document, conf: Dict
    ) -> Tuple[bool, List[Document]]:
        doc_hash = docutils.compute_doc_hash(doc)
        try:
            cached_splits = self.cached_splits_mng.get(
                cache_key=os.path.join("cached_splits", doc_hash)
            )
            for k, v in conf.items():
                if cached_splits.get(k) != v:
                    return False, []
            return True, {
                **cached_splits,
                "splits": self._deserialize_docs(cached_splits.get("doc", [])),
            }
        except FileNotFoundError:
            return False, []

    @staticmethod
    def _serialize_docs(docs: List[Document]) -> List[Dict]:
        return [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata or {},
            }
            for doc in docs
        ]

    @staticmethod
    def _deserialize_docs(docs: List[Dict]) -> List[Document]:
        return [
            Document(page_content=doc["page_content"], metadata=doc.get("metadata", {}))
            for doc in docs
        ]

    @staticmethod
    def inherit_metadata(parent: Document, chunk_index: int) -> Dict:
        essential_metadata_keys = ("id", "source", "page", "doc_id")
        meta = dict(parent.metadata or {})
        if "parent_id" not in meta:
            for k in essential_metadata_keys:
                if k in meta:
                    meta["parent_id"] = meta[k]
                    break
            else:
                meta["parent_id"] = id(parent)
        meta["chunk_index"] = chunk_index
        return meta

    @staticmethod
    def break_sentences_at_breakpoints(
        sentences: List[str], breakpoint_indices: List[int]
    ) -> List[str]:
        """
        Create chunk strings by splitting sentences at given breakpoint indices.
        Breakpoints refer to the boundary AFTER sentence i, i.e., between i and i+1.
        """
        if not sentences:
            return []
        chunks: List[str] = []
        start = 0
        for bp in breakpoint_indices:
            end = bp + 1  # include sentence at index bp
            chunks.append(" ".join(sentences[start:end]))
            start = end
        # last chunk
        if start < len(sentences):
            chunks.append(" ".join(sentences[start:]))
        return [c.strip() for c in chunks if c and c.strip()]

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Splits a given text into sentences based on punctuation and removes extra spaces.

        The text is split at punctuation marks (e.g., '.', '!', '?') followed by whitespace.
        Empty or whitespace-only sentences are excluded from the result.

        Args:
            text (str): The input text to split into sentences.

        Returns:
            List[str]: A list of cleaned sentences.
        """
        if not text:
            return []
        parts = SemanticTextSplitter._SPLIT_SENTENCES_RE.split(text)
        return [p.strip() for p in parts if p and p.strip()]

    @staticmethod
    def windowed_concat(sentences: List[str], bufsz: int) -> List[str]:
        """
        Concatenates a window of surrounding sentences for each sentence in the input list.

        For each sentence, this function includes up to `bufsz` sentences to the left and right,
        concatenating them into a single string. At the edges of the list, the window is adjusted
        to include only available sentences.

        Args:
            sentences (List[str]): A list of sentences to process.
            bufsz (int): The number of surrounding sentences to include on each side.

        Returns:
            List[str]: A list of concatenated strings, one for each sentence in the input list.

        Raises:
            ValueError: If `bufsz` is negative.
        """
        if bufsz < 0:
            err.log_and_raise(logger, ValueError("`bufsz` must be a positive integer."))
        n = len(sentences)
        windowed = []
        for i in range(n):
            start = max(0, i - bufsz)
            end = min(n, i + bufsz + 1)
            windowed.append(" ".join(sentences[start:end]))
        return windowed

    @staticmethod
    def calc_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
