"""Ingesta: PDFs -> chunks -> embeddings bge-m3 -> índice persistido en storage/."""

import argparse
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "gpt-oss:20b"


def get_embedding_model() -> OllamaEmbedding:
    return OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_URL)


def parse_documents() -> list:
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No hay PDFs en {DATA_DIR}")
    return SimpleDirectoryReader(
        input_files=pdfs,
        required_exts=[".pdf"],
        filename_as_id=True,
    ).load_data()


def build_nodes(docs: list) -> list:
    parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    return parser.get_nodes_from_documents(docs)


def main(force: bool = False) -> None:
    if STORAGE_DIR.exists() and not force:
        print(f"Índice ya existe en {STORAGE_DIR}. Usa --force para reindexar.")
        return

    embed_model = get_embedding_model()
    Settings.embed_model = embed_model
    Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL, request_timeout=300.0)

    print("Cargando documentos...")
    docs = parse_documents()
    print(f"  {len(docs)} documentos: {[d.metadata.get('file_name') for d in docs]}")

    print("Chunkizando...")
    nodes = build_nodes(docs)
    print(f"  {len(nodes)} nodos generados")

    print("Generando embeddings e indexando...")
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    print(f"Índice persistido en {STORAGE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa los PDFs de data/")
    parser.add_argument("--force", action="store_true", help="Reindexar aunque exista storage/")
    args = parser.parse_args()
    main(force=args.force)
