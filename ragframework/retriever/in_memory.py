"""In-memory retriever using cosine similarity."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError
from ragframework.utils.vectors import validate_vector

logger = logging.getLogger(__name__)


class InMemoryRetriever(Retriever):
    """Store chunks in RAM and rank them by cosine similarity."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray[Any, Any] | None = None
        self._matrix_blocks: list[np.ndarray[Any, Any]] = []
        self._block_ids: list[list[str]] = []
        self._id_to_index: dict[str, int] = {}
        self._id_to_location: dict[str, tuple[int, int]] = {}
        self._dimension: int | None = None

    def add(self, chunks: list[Chunk]) -> None:
        """Validate and upsert chunks; the last duplicate ID wins."""
        if not chunks:
            return

        expected_dimension = self._dimension
        validated: dict[str, tuple[Chunk, np.ndarray[Any, np.dtype[np.float32]]]] = {}

        # Validate the entire batch before changing stored state.
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
                    f"Chunk '{chunk.id}' embedding has dimension "
                    f"{vector.shape[0]}; expected {expected_dimension}."
                )

            # Dict assignment makes the last occurrence win.
            validated[chunk.id] = (chunk, vector)

        assert expected_dimension is not None

        # Prepare the new state separately so failed preparation cannot
        # partially mutate the existing retriever.
        final_chunks = list(self._chunks)
        final_positions = dict(self._id_to_index)
        final_locations = dict(self._id_to_location)
        final_blocks = list(self._matrix_blocks)
        final_block_ids = [list(ids) for ids in self._block_ids]

        new_chunks = list(validated.values())
        new_ids = list(validated.keys())
        new_matrix = np.ascontiguousarray(
            np.stack([vector for _, vector in new_chunks]),
            dtype=np.float32,
        )

        block_index = len(final_blocks)
        final_blocks.append(new_matrix)
        final_block_ids.append(new_ids)

        for row_index, (chunk, _) in enumerate(new_chunks):
            position = final_positions.get(chunk.id)

            if position is None:
                position = len(final_chunks)
                final_positions[chunk.id] = position
                final_chunks.append(chunk)
            else:
                final_chunks[position] = chunk

            # The latest location supersedes any older physical row.
            final_locations[chunk.id] = (block_index, row_index)

        self._chunks = final_chunks
        self._id_to_index = final_positions
        self._id_to_location = final_locations
        self._matrix_blocks = final_blocks
        self._block_ids = final_block_ids
        self._matrix = None
        self._dimension = expected_dimension

        logger.debug(
            "Added chunks count=%d dimension=%d",
            len(validated),
            expected_dimension,
        )

    def _consolidate(self) -> None:
        """Build one matrix from the latest vector for each logical chunk."""
        if self._matrix is not None:
            return

        if not self._chunks:
            return

        vectors: list[np.ndarray[Any, Any]] = []

        for chunk in self._chunks:
            block_index, row_index = self._id_to_location[chunk.id]
            vectors.append(self._matrix_blocks[block_index][row_index])

        matrix = np.ascontiguousarray(np.stack(vectors), dtype=np.float32)

        self._matrix = matrix
        self._matrix_blocks = [matrix]
        self._block_ids = [[chunk.id for chunk in self._chunks]]
        self._id_to_location = {chunk.id: (0, index) for index, chunk in enumerate(self._chunks)}

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Return the most similar chunks first."""
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise RetrieverError("top_k must be an integer.")

        if top_k <= 0 or not self._chunks:
            return []

        q = validate_vector(query_embedding, "Query embedding")

        if q.shape[0] != self._dimension:
            raise RetrieverError(
                f"Query embedding has dimension {q.shape[0]}; " f"expected {self._dimension}."
            )

        self._consolidate()
        assert self._matrix is not None

        scores: np.ndarray[Any, Any] = self._matrix @ q
        k = min(top_k, len(self._chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [self._chunks[int(index)] for index in top_indices]

    def __len__(self) -> int:
        return len(self._chunks)
