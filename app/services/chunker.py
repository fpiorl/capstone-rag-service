class RecursiveCharacterChunker:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.separators: list[str] = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []

        base_chunks = self._split_recursive(cleaned, self.separators)
        merged_chunks = [
            chunk.strip() for chunk in base_chunks if chunk and chunk.strip()
        ]
        return self._apply_overlap(merged_chunks)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._hard_split(text)

        separator = separators[0]
        if separator == "":
            return self._hard_split(text)

        pieces = text.split(separator)
        if len(pieces) == 1:
            return self._split_recursive(text, separators[1:])

        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = piece if not current else f"{current}{separator}{piece}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.extend(self._split_recursive(current, separators[1:]))
            current = piece

            if len(current) > self.chunk_size:
                chunks.extend(self._split_recursive(current, separators[1:]))
                current = ""

        if current:
            chunks.extend(self._split_recursive(current, separators[1:]))

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        text_length = len(text)
        step = max(self.chunk_size - self.chunk_overlap, 1)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            start += step

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if not chunks or self.chunk_overlap <= 0:
            return chunks

        overlapped: list[str] = [chunks[0]]
        for index in range(1, len(chunks)):
            prefix = chunks[index - 1][-self.chunk_overlap :]
            combined = f"{prefix}{chunks[index]}".strip()
            overlapped.append(combined)
        return overlapped
