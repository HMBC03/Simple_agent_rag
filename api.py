"""API FastAPI sobre el workflow RAG. Comparte el núcleo con mcp_server.py."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_workflow import QueryResult, query_documents

BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent

app = FastAPI(title="RAG Inmobiliaria", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="API REST del RAG")
    parser.add_argument("--port", type=int, default=8000, help="Puerto (default 8000)")
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
