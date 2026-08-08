"""API FastAPI sobre el workflow RAG. Comparte el núcleo con mcp_server.py."""

from fastapi import FastAPI
from pydantic import BaseModel

from rag_workflow import QueryResult, query_documents

app = FastAPI(title="RAG Inmobiliaria", version="0.1.0")


class QueryRequest(BaseModel):
    question: str
    rerank: bool = False


class SourceOut(BaseModel):
    ref: int
    file: str
    page: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


def to_response(result: QueryResult) -> QueryResponse:
    return QueryResponse(answer=result.answer, sources=result.sources)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    result = await query_documents(req.question, rerank=req.rerank)
    return to_response(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
