from typing import ClassVar

from pydantic import BaseModel, ConfigDict


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class IngestRequest(BaseModel):
    user_id: str
    document_id: str
    filename: str
    pdf_content: str

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel, populate_by_name=True
    )


class IngestResponse(BaseModel):
    success: bool
    chunk_count: int
    message: str

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel, populate_by_name=True
    )


class QueryRequest(BaseModel):
    user_id: str
    query: str

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel, populate_by_name=True
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel, populate_by_name=True
    )


class ErrorResponse(BaseModel):
    success: bool
    message: str

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel, populate_by_name=True
    )
