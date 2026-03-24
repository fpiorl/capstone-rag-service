import base64

from openrouter import OpenRouter

from app.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.services.chunker import RecursiveCharacterChunker
from app.services.embeddings import EmbeddingService
from app.services.pdf_parser import PdfParser
from app.services.vector_store import VectorStore


class RagPipeline:
    def __init__(
        self,
        client: OpenRouter,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        pdf_parser: PdfParser,
        chunker: RecursiveCharacterChunker,
        chat_model: str,
        top_k: int,
    ) -> None:
        self.client: OpenRouter = client
        self.embedding_service: EmbeddingService = embedding_service
        self.vector_store: VectorStore = vector_store
        self.pdf_parser: PdfParser = pdf_parser
        self.chunker: RecursiveCharacterChunker = chunker
        self.chat_model: str = chat_model
        self.top_k: int = top_k

    def ingest(self, request: IngestRequest) -> IngestResponse:
        pdf_bytes = base64.b64decode(request.pdf_content, validate=True)
        extracted_text = self.pdf_parser.extract_text(pdf_bytes)
        chunks = self.chunker.split_text(extracted_text)

        if not chunks:
            message = f"Ingested 0 chunks from {request.filename}"
            return IngestResponse(success=True, chunk_count=0, message=message)

        embeddings = self.embedding_service.embed_texts(chunks)
        chunk_count = self.vector_store.add_chunks(
            user_id=request.user_id,
            document_id=request.document_id,
            filename=request.filename,
            chunks=chunks,
            embeddings=embeddings,
        )

        message = f"Ingested {chunk_count} chunks from {request.filename}"
        return IngestResponse(success=True, chunk_count=chunk_count, message=message)

    def query(self, request: QueryRequest) -> QueryResponse:
        query_embedding = self.embedding_service.embed_query(request.query)
        documents, metadatas = self.vector_store.query(
            user_id=request.user_id,
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        if not documents:
            return QueryResponse(
                answer="No documents uploaded yet. Please upload a PDF first, then ask your question.",
                sources=[],
            )

        context_blocks = "\n---\n".join(documents)
        prompt = (
            "You are a helpful assistant. Answer the question based ONLY on the provided context.\n"
            "If the context doesn't contain enough information, say so honestly.\n\n"
            "Context:\n"
            "---\n"
            f"{context_blocks}\n\n"
            f"Question: {request.query}"
        )

        completion = self.client.chat.send(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = (
            completion.choices[0].message.content or "I could not generate an answer."
        )
        unique_sources: list[str] = []
        for metadata in metadatas:
            filename = str(metadata.get("filename", "")).strip()
            if filename and filename not in unique_sources:
                unique_sources.append(filename)

        return QueryResponse(answer=answer, sources=unique_sources)
