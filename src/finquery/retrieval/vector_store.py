"""Qdrant vector store: incremental indexing and filtered dense search.

- Local mode (a folder, no server) when vector_store.url is empty, server mode otherwise
  (Docker Compose sets QDRANT_URL=http://qdrant:6333). Same code for both.
- Point ids are UUIDs derived from chunk_id: re-indexing a chunk overwrites it.
- Each point stores content_hash: only new or changed chunks are embedded again,
  chunks that disappeared are deleted.
- Payload indexes on the filter fields keep filtered searches fast.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from qdrant_client import QdrantClient, models

from finquery.config import resolve
from finquery.retrieval.filters import check_filters

KEYWORD_FIELDS = ("doc_name", "company", "doc_type")
INTEGER_FIELDS = ("doc_period", "page")


def get_client(settings: dict[str, Any]) -> QdrantClient:
    store = settings["vector_store"]
    if store.get("url"):
        return QdrantClient(url=store["url"])
    return QdrantClient(path=str(resolve(store["local_path"])))


def is_local(client: QdrantClient) -> bool:
    options = client.init_options
    return bool(options.get("path")) or options.get("location") == ":memory:"


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ensure_collection(client: QdrantClient, name: str, dimension: int) -> None:
    if client.collection_exists(name):
        size = client.get_collection(name).config.params.vectors.size
        if size != dimension:
            raise ValueError(f"Collection {name!r} has vectors of size {size}, the model gives "
                             f"{dimension}. Changed the embedding model? Run with --rebuild.")
        return
    client.create_collection(name, vectors_config=models.VectorParams(
        size=dimension, distance=models.Distance.COSINE))
    if is_local(client):  # payload indexes only exist on a Qdrant server
        return
    for field in KEYWORD_FIELDS:
        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
    for field in INTEGER_FIELDS:
        client.create_payload_index(name, field, models.PayloadSchemaType.INTEGER)


def stored_hashes(client: QdrantClient, name: str) -> dict[str, str]:
    """{chunk_id: content_hash} of everything already in the collection."""
    hashes, offset = {}, None
    while True:
        points, offset = client.scroll(name, limit=1000, offset=offset,
                                       with_payload=["chunk_id", "content_hash"], with_vectors=False)
        hashes.update({p.payload["chunk_id"]: p.payload["content_hash"] for p in points})
        if offset is None:
            return hashes


def upsert_chunks(client: QdrantClient, name: str, chunks: list[dict[str, Any]],
                  embed: Callable[[list[str]], Any], text_of: Callable[[dict], str],
                  batch_size: int = 256, progress: Callable[[int, int], None] | None = None) -> None:
    """Embed and write chunks (adds or overwrites, never deletes)."""
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = embed([text_of(c) for c in batch])
        client.upsert(name, points=[
            models.PointStruct(id=point_id(c["chunk_id"]), vector=list(map(float, v)), payload=c)
            for c, v in zip(batch, vectors, strict=True)])
        if progress:
            progress(start + len(batch), len(chunks))


def delete_document(client: QdrantClient, name: str, doc_name: str) -> None:
    selector = models.FilterSelector(filter=to_qdrant_filter({"doc_name": doc_name}))
    client.delete(name, points_selector=selector)


def sync_chunks(client: QdrantClient, name: str, chunks: list[dict[str, Any]],
                embed: Callable[[list[str]], Any], text_of: Callable[[dict], str],
                batch_size: int = 256, progress: Callable[[int, int], None] | None = None) -> dict[str, int]:
    """Make the collection match `chunks`: embed and upsert new/changed chunks,
    delete chunks that are gone. Returns counts per category."""
    existing = stored_hashes(client, name)
    current = {c["chunk_id"] for c in chunks}
    todo = [c for c in chunks if existing.get(c["chunk_id"]) != c["content_hash"]]
    gone = [cid for cid in existing if cid not in current]

    upsert_chunks(client, name, todo, embed, text_of, batch_size, progress)
    if gone:
        client.delete(name, points_selector=models.PointIdsList(points=[point_id(c) for c in gone]))

    new = sum(1 for c in todo if c["chunk_id"] not in existing)
    return {"new": new, "updated": len(todo) - new, "deleted": len(gone),
            "unchanged": len(chunks) - len(todo)}


def to_qdrant_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    conditions = [models.FieldCondition(key=field, match=models.MatchAny(any=values))
                  for field, values in check_filters(filters).items()]
    return models.Filter(must=conditions) if conditions else None


def dense_search(client: QdrantClient, name: str, vector: Any, limit: int = 20,
                 filters: dict[str, Any] | None = None) -> list[tuple[dict[str, Any], float]]:
    response = client.query_points(name, query=list(map(float, vector)), limit=limit,
                                   query_filter=to_qdrant_filter(filters), with_payload=True)
    return [(point.payload, point.score) for point in response.points]
