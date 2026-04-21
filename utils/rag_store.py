"""
PDF → chunk → local embeddings (sentence-transformers) → Qdrant.
Retrieve by semantic similarity, filtered by document_id(s).
"""

from __future__ import annotations

import os
import threading
import uuid
from io import BytesIO
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_client: Any = None
_model: Any = None
_client_lock = threading.Lock()
_model_lock = threading.Lock()


def _qdrant_url() -> str:
    return (os.getenv('QDRANT_URL') or 'http://127.0.0.1:6333').strip()


def _collection_name() -> str:
    return (os.getenv('QDRANT_COLLECTION') or 'pipeline_documents').strip()


def _embedding_model_name() -> str:
    return (os.getenv('RAG_EMBEDDING_MODEL') or 'all-MiniLM-L6-v2').strip()


def _chunk_size() -> int:
    try:
        return max(200, int(os.getenv('RAG_CHUNK_SIZE', '900')))
    except ValueError:
        return 900


def _chunk_overlap() -> int:
    try:
        return max(0, int(os.getenv('RAG_CHUNK_OVERLAP', '120')))
    except ValueError:
        return 120


def _top_k() -> int:
    try:
        return max(1, min(64, int(os.getenv('RAG_TOP_K', '16'))))
    except ValueError:
        return 16


def chunk_string(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping windows (whitespace-stripped segments)."""
    size = chunk_size if chunk_size is not None else _chunk_size()
    ov = overlap if overlap is not None else _chunk_overlap()
    text = (text or '').strip()
    if not text:
        return []
    if size <= 0:
        return [text]
    step = max(1, size - ov)
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        piece = text[i : i + size].strip()
        if piece:
            chunks.append(piece)
        i += step
    return chunks


def extract_pdf_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    """Return (1-based page number, text) for each page that has extractable text."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(file_bytes))
    out: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text()
        t = (raw or '').strip()
        if t:
            out.append((i + 1, t))
    return out


def _get_qdrant_client():
    global _client
    with _client_lock:
        if _client is None:
            from qdrant_client import QdrantClient

            _client = QdrantClient(
                url=_qdrant_url(),
                timeout=60.0,
                check_compatibility=False,
            )
        return _client


def _embedding_dim(model: Any) -> int:
    """sentence-transformers ≥5 prefers get_embedding_dimension()."""
    fn = getattr(model, 'get_embedding_dimension', None)
    if callable(fn):
        return int(fn())
    return int(model.get_sentence_embedding_dimension())


def _get_embed_model():
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(_embedding_model_name())
        return _model


def _ensure_collection(client: Any, vector_size: int) -> None:
    from qdrant_client.models import Distance, VectorParams

    name = _collection_name()
    cols = client.get_collections().collections
    names = {c.name for c in cols}
    if name in names:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _safe_filename(name: str) -> str:
    base = os.path.basename((name or 'document.pdf').strip()) or 'document.pdf'
    if not base.lower().endswith('.pdf'):
        base = f'{base}.pdf'
    return base[:200]


def ingest_pdf_bytes(file_bytes: bytes, filename: str) -> tuple[str, int]:
    """
    Chunk PDF, embed, upsert to Qdrant. Returns (document_id, num_chunks).
    Raises on empty PDF, extraction failure, or Qdrant/model errors.
    """
    if not file_bytes:
        raise ValueError('Empty file')

    pages = extract_pdf_pages(file_bytes)
    if not pages:
        raise ValueError('No extractable text in PDF (empty or image-only)')

    fname = _safe_filename(filename)
    document_id = str(uuid.uuid4())
    chunks_meta: list[tuple[int, str]] = []
    for page_num, page_text in pages:
        for piece in chunk_string(page_text):
            chunks_meta.append((page_num, piece))

    if not chunks_meta:
        raise ValueError('No text chunks produced from PDF')

    model = _get_embed_model()
    dim = _embedding_dim(model)
    client = _get_qdrant_client()
    _ensure_collection(client, dim)

    texts = [c[1] for c in chunks_meta]
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    from qdrant_client.models import PointStruct

    name = _collection_name()
    points: list[PointStruct] = []
    for idx, ((page_num, piece), vec) in enumerate(zip(chunks_meta, vectors)):
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f'{document_id}:{idx}'))
        points.append(
            PointStruct(
                id=pid,
                vector=vec.tolist(),
                payload={
                    'document_id': document_id,
                    'chunk_index': idx,
                    'page': page_num,
                    'filename': fname,
                    'text': piece,
                },
            )
        )

    batch = 64
    for i in range(0, len(points), batch):
        client.upsert(collection_name=name, points=points[i : i + batch])

    return document_id, len(points)


def search_results_from_rag(query: str, document_ids: list[str]) -> list[dict]:
    """
    Semantic search over stored chunks; same shape as web_search rows for the pipeline.
    On failure or missing deps, logs and returns [].
    """
    ids = [d.strip() for d in document_ids if d and str(d).strip()]
    if not ids:
        return []

    try:
        model = _get_embed_model()
        client = _get_qdrant_client()
        dim = _embedding_dim(model)
        _ensure_collection(client, dim)

        qvec = model.encode(query, show_progress_bar=False, convert_to_numpy=True).tolist()

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = Filter(
            should=[FieldCondition(key='document_id', match=MatchValue(value=did)) for did in ids],
        )

        resp = client.query_points(
            collection_name=_collection_name(),
            query=qvec,
            query_filter=flt,
            limit=_top_k(),
            with_payload=True,
        )
        hits = resp.points or []
    except Exception as e:
        print(f'   RAG retrieve failed (continuing with web only): {e}')
        return []

    out: list[dict] = []
    for h in hits:
        raw_pl = h.payload
        pl = raw_pl if isinstance(raw_pl, dict) else (dict(raw_pl) if raw_pl is not None else {})
        doc_id = pl.get('document_id', '')
        page = pl.get('page', 0)
        fname = pl.get('filename', 'document.pdf')
        chunk_i = pl.get('chunk_index', 0)
        text = (pl.get('text') or '').strip()
        if not text:
            continue
        url = f'upload://{doc_id}#page={page}&chunk={chunk_i}'
        title = f'[Uploaded PDF] {fname} (p. {page}, chunk {chunk_i})'
        score = float(h.score) if h.score is not None else 1.0
        out.append({
            'title': title,
            'url': url,
            'content': text,
            'score': score,
            'source': 'pdf',
        })
    return out
