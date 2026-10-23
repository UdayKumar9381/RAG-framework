"""FAISS-backed approximate nearest-neighbour retrieval."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:
    _FAISS_AVAILABLE = False
else:
    _FAISS_AVAILABLE = True

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError
from ragframework.utils.vectors import validate_vector

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "FAISS support requires 'ragframework[faiss]'. "
    "Install it with: pip install ragframework[faiss]"
)


class FAISSRetriever(Retriever):
    """Retrieve chunks with an HNSW index and cosine similarity.

    Chunk and query vectors are L2-normalized before being passed to an
    ``IndexHNSWFlat`` configured for inner product. Inner product between
    normalized vectors is cosine similarity.

    Args:
        m: Number of HNSW neighbours stored per node.
        ef_construction: HNSW construction-time search depth.
        ef_search: HNSW query-time search depth.

    Raises:
        ImportError: If the optional ``faiss-cpu`` dependency is unavailable.
        RetrieverError: If an HNSW parameter is not positive.
    """

    def __init__(
        self,
        m: int = 32,
        ef_construction: int = 40,
        ef_search: int = 16,
    ) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(_INSTALL_HINT)

        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (m, ef_construction, ef_search)
        ):
            raise RetrieverError("m, ef_construction, and ef_search must be positive integers.")

        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search

        self._chunks: list[Chunk] = []
        self._dimension: int | None = None
        self._index: Any | None = None

        # Maps logical chunk IDs to their current physical FAISS labels.
        # Old labels remain in the FAISS index as tombstones after replacement.
        self._id_to_index: dict[str, int] = {}

        # Logical chunk ID -> position in self._chunks.
        self._id_to_position: dict[str, int] = {}

    def _build_index(
        self,
        vectors: np.ndarray[Any, Any],
        dimension: int,
    ) -> Any:
        """Build an HNSW index from normalized vectors."""
        index = faiss.IndexHNSWFlat(
            dimension,
            self._m,
            faiss.METRIC_INNER_PRODUCT,
        )

        index.hnsw.efConstruction = self._ef_construction
        index.hnsw.efSearch = self._ef_search

        index.add(vectors)

        return index

    def add(self, chunks: list[Chunk]) -> None:
        """Validate and upsert chunks; the last duplicate ID in a batch wins.

        Existing IDs retain their logical position. New IDs are appended.
        Replaced physical FAISS vectors become tombstones.
        """
        if not chunks:
            return

        expected_dimension = self._dimension
        validated: dict[str, tuple[Chunk, np.ndarray[Any, np.dtype[np.float32]]]] = {}

        # Validate the entire batch before mutating retriever state.
        # Dict reassignment keeps the first-seen order but the last value.
        for chunk in chunks:
            if chunk.embedding is None:
                raise RetrieverError(
                    f"Chunk '{chunk.id}' has no embedding. "
                    "Embed chunks before adding them to the retriever."
                )

            vector = validate_vector(
                chunk.embedding,
                f"Chunk '{chunk.id}' embedding",
            )

            if expected_dimension is None:
                expected_dimension = int(vector.shape[0])
            elif vector.shape[0] != expected_dimension:
                raise RetrieverError(
                    f"Chunk '{chunk.id}' embedding has dimension "
                    f"{vector.shape[0]}; expected {expected_dimension}."
                )

            validated[chunk.id] = (chunk, vector)

        assert expected_dimension is not None

        unique_chunks = [item[0] for item in validated.values()]
        vectors = [item[1] for item in validated.values()]
        matrix = np.ascontiguousarray(
            np.stack(vectors),
            dtype=np.float32,
        )

        # Build a fresh index for the first insertion.
        if self._index is None:
            try:
                new_index = self._build_index(matrix, expected_dimension)
            except Exception as exc:
                raise RetrieverError("FAISS failed to add chunk embeddings.") from exc

            self._index = new_index
            self._dimension = expected_dimension

            for physical_label, chunk in enumerate(unique_chunks):
                logical_position = len(self._chunks)
                self._chunks.append(chunk)
                self._id_to_index[chunk.id] = physical_label
                self._id_to_position[chunk.id] = logical_position

        else:
            # Add only the final version of each ID to FAISS.
            start_label = int(self._index.ntotal)
            try:
                self._index.add(matrix)
            except Exception as exc:
                raise RetrieverError("FAISS failed to add chunk embeddings.") from exc

            # Update the logical mapping only after FAISS succeeds.
            for offset, chunk in enumerate(unique_chunks):
                physical_label = start_label + offset
                existing_position = self._id_to_position.get(chunk.id)

                if existing_position is None:
                    logical_position = len(self._chunks)
                    self._chunks.append(chunk)
                    self._id_to_position[chunk.id] = logical_position
                else:
                    logical_position = existing_position
                    self._chunks[logical_position] = chunk

                self._id_to_index[chunk.id] = physical_label

            self._dimension = expected_dimension

        logger.debug(
            "Added chunks count=%d dimension=%d",
            len(unique_chunks),
            matrix.shape[1],
        )

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[Chunk]:
        """Retrieve active chunks using progressively expanded FAISS searches."""
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise RetrieverError("top_k must be an integer.")

        if top_k <= 0 or self._index is None or self._dimension is None or not self._chunks:
            return []

        query = validate_vector(query_embedding, "Query embedding")

        if query.shape[0] != self._dimension:
            raise RetrieverError(
                f"Query embedding has dimension {query.shape[0]}; " f"expected {self._dimension}."
            )

        total = int(self._index.ntotal)

        # Current physical label -> current logical position.
        active_labels = {
            physical_label: self._id_to_position[chunk_id]
            for chunk_id, physical_label in self._id_to_index.items()
            if chunk_id in self._id_to_position
        }

        # Start with a bounded over-fetch, not every physical vector.
        search_count = min(total, max(top_k * 2, top_k + 8))
        candidates: dict[int, tuple[float, int, Chunk]] = {}

        try:
            while True:
                scores, labels = self._index.search(
                    query.reshape(1, -1),
                    search_count,
                )

                for score, label in zip(
                    scores[0],
                    labels[0],
                    strict=True,
                ):
                    physical_label = int(label)
                    if physical_label < 0:
                        continue

                    logical_position = active_labels.get(physical_label)
                    if logical_position is None:
                        continue

                    chunk = self._chunks[logical_position]
                    candidates[physical_label] = (
                        float(score),
                        logical_position,
                        chunk,
                    )

                if len(candidates) >= min(top_k, len(self._chunks)):
                    break

                if search_count >= total:
                    break

                search_count = min(total, search_count * 2)

        except Exception as exc:
            raise RetrieverError("FAISS failed to search the index.") from exc

        ordered = sorted(
            candidates.values(),
            key=lambda item: (-item[0], item[1]),
        )

        return [chunk for _score, _position, chunk in ordered[:top_k]]

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)
