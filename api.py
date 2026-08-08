"""API FastAPI sobre el workflow RAG. Comparte el núcleo con mcp_server.py."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_workflow import QueryResult, query_documents

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="RAG Inmobiliaria",
    version="0.3.0",
    description=(
        "API REST del RAG agéntico sobre manuales inmobiliarios (CRM y leads). "
        "`POST /query` responde con citas verificables (archivo, página, párrafo, score); "
        "`GET /documents` cataloga los PDFs indexados; "
        "`GET /pdf/{archivo}` sirve el PDF en línea para el visor del navegador (usa `#page=N` para saltar); "
        "`GET /health` para monitoreo. El frontend web se sirve en la raíz."
    ),
    docs_url=None,
    redoc_url=None,
)

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


class DocumentOut(BaseModel):
    file: str
    pages: int
    size_kb: float


class DocumentsResponse(BaseModel):
    documents: list[DocumentOut]


def to_response(result: QueryResult) -> QueryResponse:
    return QueryResponse(answer=result.answer, sources=result.sources)


@app.get("/health", summary="Estado de la API", tags=["core"])
async def health() -> dict:
    return {"status": "ok"}


@app.get(
    "/documents",
    response_model=DocumentsResponse,
    summary="Catálogo de PDFs indexados",
    tags=["documentos"],
)
async def documents() -> DocumentsResponse:
    return DocumentsResponse(documents=_list_documents())


@app.get(
    "/pdf/{filename}",
    summary="Servir un PDF en línea para el visor del navegador",
    tags=["documentos"],
)
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


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Consultar los manuales con citas verificables",
    tags=["consulta"],
)
async def query(req: QueryRequest) -> QueryResponse:
    result = await query_documents(req.question, rerank=req.rerank)
    return to_response(result)


@app.get("/docs", include_in_schema=False, tags=["core"])
async def swagger_ui() -> HTMLResponse:
    return HTMLResponse(
        """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>API RAG Inmobiliaria — Documentación</title>
<link rel="stylesheet" href="/vendor/swagger-ui/swagger-ui.css">
</head>
<body style="margin:0">
<div id="swagger-ui"></div>
<script src="/vendor/swagger-ui/swagger-ui-bundle.js"></script>
<script src="/vendor/swagger-ui/swagger-ui-standalone-preset.js"></script>
<script>
window.onload = function () {
  window.ui = SwaggerUIBundle({
    url: "/openapi.json",
    dom_id: "#swagger-ui",
    deepLinking: true,
    presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
    layout: "StandaloneLayout"
  });
};
</script>
</body>
</html>
"""
    )


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
