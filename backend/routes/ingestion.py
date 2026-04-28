"""
routes/ingestion.py — /api/v1/ingest endpoints.

POST /api/v1/ingest/text    — ingest raw text into the knowledge base
POST /api/v1/ingest/file    — upload and ingest a .txt file
GET  /api/v1/ingest/status  — get vector store statistics
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, status

from backend.database import log_ingestion
from backend.models import IngestRequest, IngestResponse
from backend.rag_pipeline import RAGPipeline

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_rag(request: Request) -> RAGPipeline:
    return request.app.state.rag_pipeline


@router.post(
    "/text",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest raw text into the knowledge base",
    description=(
        "Send raw text content with a source name. "
        "The pipeline will chunk, embed, and index it into FAISS automatically."
    ),
)
async def ingest_text(body: IngestRequest, request: Request) -> IngestResponse:
    rag = _get_rag(request)
    logger.info("POST /ingest/text  source='%s'  len=%d", body.source_name, len(body.content))

    try:
        chunks_added = rag.ingest(body.content, body.source_name, body.metadata)
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    if chunks_added == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Content produced zero chunks. Ensure it has sufficient text.",
        )

    rag.persist()   # persist after every successful ingest
    log_ingestion(body.source_name, chunks_added)

    return IngestResponse(
        status="success",
        source_name=body.source_name,
        chunks_added=chunks_added,
        message=f"Successfully ingested {chunks_added} chunks from '{body.source_name}'.",
    )


@router.post(
    "/file",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a .txt file",
    description="Upload a plain-text (.txt) file to add to the knowledge base.",
)
async def ingest_file(request: Request, file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .txt files are supported for upload ingestion.",
        )

    rag = _get_rag(request)
    content_bytes = await file.read()
    content = content_bytes.decode("utf-8", errors="replace")
    logger.info("POST /ingest/file  filename='%s'  size=%d bytes", file.filename, len(content_bytes))

    try:
        chunks_added = rag.ingest(content, source_name=file.filename)
    except Exception as exc:
        logger.exception("File ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File ingestion failed: {exc}",
        )

    rag.persist()
    log_ingestion(file.filename, chunks_added)

    return IngestResponse(
        status="success",
        source_name=file.filename,
        chunks_added=chunks_added,
        message=f"File '{file.filename}' ingested with {chunks_added} chunks.",
    )


@router.get(
    "/status",
    summary="Vector store statistics",
)
async def ingest_status(request: Request) -> dict:
    rag: RAGPipeline = _get_rag(request)
    return {
        "is_loaded":   rag.is_loaded,
        "chunk_count": rag.chunk_count,
        "index_path":  str(__import__("backend.config", fromlist=["settings"]).settings.vector_store_path),
    }
