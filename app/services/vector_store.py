# pyright: basic
from typing import Protocol, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Metadata, PyEmbedding


class CollectionProtocol(Protocol):
    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[PyEmbedding],
        metadatas: list[Metadata],
    ) -> object: ...

    def query(
        self, query_embeddings: list[list[float]], n_results: int
    ) -> dict[str, object]: ...

    def delete(self, where: dict[str, str]) -> object: ...


class VectorStore:
    def __init__(self, persist_directory: str) -> None:
        self.client: ClientAPI = chromadb.PersistentClient(path=persist_directory)

    def get_or_create_collection(self, user_id: str) -> CollectionProtocol:
        collection = self.client.get_or_create_collection(name=f"user_{user_id}")
        return cast(CollectionProtocol, cast(object, collection))

    def _get_collection(self, user_id: str) -> CollectionProtocol | None:
        try:
            collection = self.client.get_collection(name=f"user_{user_id}")
            return cast(CollectionProtocol, cast(object, collection))
        except Exception:
            return None

    def add_chunks(
        self,
        user_id: str,
        document_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        if not chunks:
            return 0

        collection = self.get_or_create_collection(user_id)
        ids = [f"{document_id}_{index}" for index in range(len(chunks))]
        metadatas = [
            {
                "documentId": document_id,
                "filename": filename,
                "chunkIndex": index,
            }
            for index in range(len(chunks))
        ]
        embedding_payload: list[PyEmbedding] = [
            cast(PyEmbedding, cast(object, item)) for item in embeddings
        ]
        metadata_payload: list[Metadata] = [cast(Metadata, item) for item in metadatas]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embedding_payload,
            metadatas=metadata_payload,
        )

        return len(chunks)

    def query(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> tuple[list[str], list[dict[str, object]]]:
        collection = self._get_collection(user_id)
        if collection is None:
            return [], []

        result = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        documents_result = cast(list[list[str]], result.get("documents") or [])
        metadatas_result = cast(
            list[list[dict[str, object]]], result.get("metadatas") or []
        )
        documents = documents_result[0] if documents_result else []
        raw_metadatas = metadatas_result[0] if metadatas_result else []
        metadatas: list[dict[str, object]] = [
            dict(metadata) for metadata in raw_metadatas if isinstance(metadata, dict)
        ]
        if not documents:
            return [], []

        return list(documents), metadatas

    def delete_document(self, user_id: str, document_id: str) -> None:
        collection = self._get_collection(user_id)
        if collection is None:
            return

        _ = collection.delete(where={"documentId": document_id})
