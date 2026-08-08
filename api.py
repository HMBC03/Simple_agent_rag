"""API FastAPI sobre el workflow RAG. Comparte el núcleo con mcp_server.py."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_workflow import QueryResult, query_documents

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="RAG Inmobiliaria", version="0.2.0")

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
    paragraph: int
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


@app.get("/documents")
async def documents() -> dict:
    return {"documents": _list_documents()}


@app.get("/pdf/{filename}")
async def pdf(filename: str) -> FileResponse:
    name = Path(filename).name
    target = (DATA_DIR / name).resolve()
    if not target.is_file() or target.parent != DATA_DIR.resolve():
        raise HTTPException(status_code=404, detail="documento no encontrado")
    return FileResponse(
        str(target),
        media_type="application/pdf",
        filename=name,
        content_disposition_type="inline",
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    result = await query_documents(req.question, rerank=req.rerank)
    return to_response(result)


_doc_cache: list | None = None


def _list_documents() -> list:
    global _doc_cache
    if _doc_cache is None:
        out = []
        for p in sorted(DATA_DIR.glob("*.pdf")):
            try:
                from pypdf import PdfReader

                pages = len(PdfReader(str(p)).pages)
            except Exception:
                pages = 0
            out.append(
                {
                    "file": p.name,
                    "pages": pages,
                    "size_kb": round(p.stat().st_size / 1024, 1),
                }
            )
        _doc_cache = out
    return _doc_cache


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
