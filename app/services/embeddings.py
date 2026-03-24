from openrouter import OpenRouter


class EmbeddingService:
    def __init__(self, client: OpenRouter, model: str) -> None:
        self.client: OpenRouter = client
        self.model: str = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.generate(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.generate(model=self.model, input=text)
        return response.data[0].embedding
