"""Núcleo RAG agéntico: Workflow retrieve -> (rerank) -> generate con citas verificables.

El workflow usa LlamaIndex Workflows (eventos tipados). Cada respuesta viene
acompañada de source_nodes reales; citations.py los convierte en referencias
"[n] archivo.pdf p.X" que el LLM no puede inventar porque el número lo marca
el nodo recuperado, no el modelo.
"""

from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "gpt-oss:20b"
TOP_K = 6
CONTEXT_BUDGET = 4000

SYSTEM_PROMPT = """Eres un asistente que responde consultas SOLO con la información de los documentos proporcionados.

Reglas estrictas:
1. Responde en el mismo idioma de la pregunta (español por defecto).
2. Usa citas [n] al final de cada frase que se base en un fragmento, donde n corresponde al número del fragmento.
3. Si la información no está en los fragmentos, di "No encontré esa información en los documentos" y NO inventes.
4. No uses conocimiento general: todo lo que afirmes debe estar sustentado por un fragmento citado.
5. Sé conciso y directo.

Los fragmentos disponibles son:
{context}

Responde ahora:
{query}"""


@dataclass
class QueryResult:
    answer: str
    sources: list[dict] = field(default_factory=list)


class QueryEvent(Event):
    query: str


class RetrieveEvent(Event):
    query: str
    nodes: list[NodeWithScore] = field(default_factory=list)


class RerankEvent(Event):
    query: str
    nodes: list[NodeWithScore] = field(default_factory=list)


class GenerateEvent(Event):
    query: str
    nodes: list[NodeWithScore] = field(default_factory=list)


class RagWorkflow(Workflow):
    def __init__(
        self,
        top_k: int = TOP_K,
        context_budget: int = CONTEXT_BUDGET,
        rerank: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(timeout=600.0, **kwargs)
        self.top_k = top_k
        self.context_budget = context_budget
        self.use_rerank = rerank
        self._index: VectorStoreIndex | None = None
        self._retriever = None
        self._reranker = None
        self._sentence_window_postprocessor = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )

    # ------------------------------------------------------------------ carga

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        Settings.embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL, base_url=OLLAMA_URL
        )
        Settings.llm = Ollama(
            model=LLM_MODEL, base_url=OLLAMA_URL, request_timeout=300.0
        )
        storage = StorageContext.from_defaults(
            persist_dir=str(STORAGE_DIR)
        )
        from llama_index.core.indices.loading import load_index_from_storage

        self._index = load_index_from_storage(storage)
        self._retriever = self._index.as_retriever(similarity_top_k=self.top_k)

    def _load_reranker(self) -> None:
        if self._reranker is None:
            from llama_index.postprocessor import SentenceTransformerRerank

            self._reranker = SentenceTransformerRerank(
                top_n=min(4, self.top_k),
                model="BAAI/bge-reranker-v2-m3",
                device="cuda",
            )
        return self._reranker

    # ----------------------------------------------------------------- pasos

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        return QueryEvent(query=ev.query)

    @step
    async def retrieve(self, ctx: Context, ev: QueryEvent) -> RetrieveEvent:
        self._ensure_loaded()
        nodes = self._retriever.retrieve(ev.query)
        return RetrieveEvent(query=ev.query, nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieveEvent) -> RerankEvent:
        nodes = ev.nodes
        if self.use_rerank:
            reranker = self._load_reranker()
            nodes = reranker.postprocess_nodes(nodes, query_str=ev.query)
        return RerankEvent(query=ev.query, nodes=nodes)

    @step
    async def generate(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        nodes = self._sentence_window_postprocessor.postprocess_nodes(
            ev.nodes, query_str=ev.query
        )
        context = self._budget_context(ev.query, nodes)

        llm = Settings.llm
        prompt = SYSTEM_PROMPT.format(context=context, query=ev.query)
        response: RESPONSE_TYPE = await llm.acomplete(prompt)

        result = QueryResult(answer=str(response))
        for i, node in enumerate(nodes, start=1):
            meta = node.node.metadata
            result.sources.append(
                {
                    "ref": i,
                    "file": meta.get("file_name", "desconocido"),
                    "page": meta.get("page_label", "?"),
                    "paragraph": int(meta.get("paragraph", 1)),
                    "score": round(float(node.score or 0.0), 4),
                    "text": node.node.get_content()[:400],
                }
            )
        return StopEvent(result=result)

    # ---------------------------------------------------------------- helpers

    def _budget_context(self, query: str, nodes: list[NodeWithScore]) -> str:
        budget = max(self.context_budget - len(query), 500)
        parts = []
        total = 0
        for i, node in enumerate(nodes, start=1):
            text = node.node.get_content()
            if total + len(text) > budget:
                break
            parts.append(f"[{i}] {text}")
            total += len(text)
        return "\n\n".join(parts)

    async def run_query(self, query: str) -> QueryResult:
        handler = self.run(query=query)
        result = await handler
        return result


async def query_documents(query: str, rerank: bool = False) -> QueryResult:
    """Entry point compartido por API y MCP."""
    wf = RagWorkflow(rerank=rerank)
    return await wf.run_query(query)


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Consulta el RAG con citas")
    parser.add_argument("query", nargs="?", default="¿Qué procesos del CRM están documentados?")
    parser.add_argument("--rerank", action="store_true", help="Activa el reranker")
    args = parser.parse_args()

    async def main() -> None:
        import sys

        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
        result = await query_documents(args.query, rerank=args.rerank)
        print("\n--- RESPUESTA ---\n")
        print(result.answer)
        print("\n--- FUENTES ---")
        for src in result.sources:
            print(f"[{src['ref']}] {src['file']} p.{src['page']} párr.{src['paragraph']} (score {src['score']})")

    asyncio.run(main())
