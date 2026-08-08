"""Servidor MCP: expone el RAG como tool para opencode y otros agentes.

Uso:
    python mcp_server.py            # stdio (opencode)
    python mcp_server.py --http     # HTTP (otras herramientas)
"""

import argparse

from fastmcp import FastMCP
from pydantic import Field

from rag_workflow import query_documents

mcp = FastMCP("rag-inmobiliaria")


@mcp.tool()
async def query_documents_tool(
    question: str = Field(description="Pregunta en lenguaje natural sobre los manuales"),
    rerank: bool = Field(default=False, description="Usar reranker (más lento, mejor precisión)"),
) -> str:
    """Consulta los documentos indexados (manual de procesos CRM y manejo de leads)
    y responde citando el archivo y la página de cada fuente."""
    result = await query_documents(question, rerank=rerank)
    lines = [result.answer, "", "FUENTES:"]
    for src in result.sources:
        lines.append(f"[{src['ref']}] {src['file']} p.{src['page']} (score {src['score']})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Servidor MCP del RAG")
    parser.add_argument("--http", action="store_true", help="Servir por HTTP (por defecto: stdio)")
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
