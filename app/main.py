#!/usr/bin/env python

# usa FastAPI per creare un microservice RAG 

from contextlib import asynccontextmanager
from typing import cast

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openrouter import OpenRouter

from app.config import Settings, get_settings
from app.models import (
    ErrorResponse,
    IngestRequest,
    QueryRequest,
)
from app.services.chunker import RecursiveCharacterChunker
from app.services.embeddings import EmbeddingService
from app.services.pdf_parser import PdfParser
from app.services.rag_pipeline import RagPipeline
from app.services.vector_store import VectorStore




@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = OpenRouter(api_key=settings.openrouter_api_key)
    embedding_service = EmbeddingService(
        client=client,
        model=settings.openrouter_embedding_model,
    )
    vector_store = VectorStore(persist_directory=settings.chroma_persist_dir)
    pdf_parser = PdfParser()
    chunker = RecursiveCharacterChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    app.state.rag_pipeline = RagPipeline(
        client=client,
        embedding_service=embedding_service,
        vector_store=vector_store,
        pdf_parser=pdf_parser,
        chunker=chunker,
        chat_model=settings.openrouter_chat_model,
        top_k=settings.top_k,
    )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ingest")
async def ingest(request: IngestRequest) -> JSONResponse:
    pipeline = cast(RagPipeline, app.state.rag_pipeline)
    try:
        result = pipeline.ingest(request)
        return JSONResponse(content=result.model_dump(by_alias=True))
    except Exception as exc:
        error = ErrorResponse(success=False, message=f"Ingest failed: {exc}")
        return JSONResponse(status_code=500, content=error.model_dump(by_alias=True))


@app.post("/api/query")
async def query(request: QueryRequest) -> JSONResponse:
    pipeline = cast(RagPipeline, app.state.rag_pipeline)
    result = pipeline.query(request)
    return JSONResponse(content=result.model_dump(by_alias=True))


@app.delete("/api/documents/{userId}/{documentId}", status_code=204)
async def delete_document(userId: str, documentId: str) -> Response:
    pipeline = cast(RagPipeline, app.state.rag_pipeline)
    pipeline.vector_store.delete_document(user_id=userId, document_id=documentId)
    return Response(status_code=204)


if __name__ == "__main__":
    settings: Settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
