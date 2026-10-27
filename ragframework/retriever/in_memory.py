"""In-memory retriever using cosine similarity."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError
from ragframework.utils.vectors import validate_vector

logger = logging.getLogger(__name__)


class InMemoryRetriever(Retriever):
    """A simple retriever that stores chunks in RAM and ranks by cosine similarity.

    Suitable for small corpora and quick experiments. For large-scale use,
    swap this out with a FAISS or ChromaDB retriever (see
    ``.github/GOOD_FIRST_ISSUES.md``).
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray[Any, Any] | None = None  # shape (N, dim)
        self._matrix_blocks: list[np.ndarray[Any, Any]] = []
        self._dimension: int | None = None

    def add(self, chunks: list[Chunk]) -> None:
        """Validate and normalize a batch before changing the stored chunks.

        An empty batch is a no-op. Invalid vectors or inconsistent dimensions
        raise ``RetrieverError`` without adding any part of the batch.
        """
        if not chunks:
            return

        expected_dimension = self._dimension
        vectors: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise RetrieverError(
                    f"Chunk '{chunk.id}' has no embedding. "
                    "Embed chunks before adding them to the retriever."
                )
            vector = validate_vector(chunk.embedding, f"Chunk '{chunk.id}' embedding")
            if expected_dimension is None:
                expected_dimension = int(vector.shape[0])
            elif vector.shape[0] != expected_dimension:
                raise RetrieverError(
                    f"Chunk '{chunk.id}' embedding has dimension {vector.shape[0]}; "
                    f"expected {expected_dimension}."
                )
            vectors.append(vector)

        matrix = np.stack(vectors)
        self._matrix_blocks.append(matrix)

        self._matrix = None
        self._dimension = expected_dimension
        self._chunks.extend(chunks)
        logger.debug(
            "Added chunks count=%d dimension=%d",
            len(chunks),
            matrix.shape[1],
        )

    @staticmethod
    def _validate_filter(filter: Mapping[str, Any] | None) -> None:
        """Validate that filter values are supported scalar types."""
        if filter is None:
            return

        if not isinstance(filter, Mapping):
            raise RetrieverError("filter must be a mapping or None.")

        for key, value in filter.items():
            if not isinstance(key, str):
                raise RetrieverError("Filter keys must be strings.")

            if not isinstance(value, (str, int, float, bool)):
                raise RetrieverError(
                    f"Filter value for '{key}' must be a scalar " "(str, int, float, or bool)."
                )

            if isinstance(value, float) and not np.isfinite(value):
                raise RetrieverError(f"Filter value for '{key}' must be finite.")

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        *,
        filter: Mapping[str, Any] | None = None,
    ) -> list[Chunk]:
        """Rank chunks by cosine similarity, optionally filtering metadata."""
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise RetrieverError("top_k must be an integer.")

        self._validate_filter(filter)

        if top_k <= 0 or not self._chunks:
            return []

        q = validate_vector(query_embedding, "Query embedding")
        if q.shape[0] != self._dimension:
            raise RetrieverError(
                f"Query embedding has dimension {q.shape[0]}; " f"expected {self._dimension}."
            )

        if self._matrix is None:
            if len(self._matrix_blocks) == 1:
                self._matrix = self._matrix_blocks[0]
            else:
                self._matrix = np.concatenate(self._matrix_blocks)
                self._matrix_blocks = [self._matrix]

        scores: np.ndarray[Any, Any] = self._matrix @ q

        # Sort all candidates by similarity before applying metadata filters.
        ordered_indices = np.argsort(scores)[::-1]

        results: list[Chunk] = []
        for index in ordered_indices:
            chunk = self._chunks[int(index)]

            if filter is not None and not all(
                chunk.metadata.get(key) == expected for key, expected in filter.items()
            ):
                continue

            results.append(chunk)
            if len(results) == top_k:
                break

        return results

    def clear(self) -> None:
        """Remove all chunks and reset the accepted embedding dimension."""
        self._chunks = []
        self._matrix = None
        self._matrix_blocks = []
        self._dimension = None

    def __len__(self) -> int:
        return len(self._chunks)
