"""Evaluación del RAG con Ragas.

Uso:
    python eval.py                         # eval completo sobre el test set
    python eval.py --build-test-set        # genera test_set.json desde los PDFs (con LLM)
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEST_SET_PATH = BASE_DIR / "test_set.json"

sys.path.insert(0, str(BASE_DIR))


def build_test_set() -> None:
    from llama_index.core import Settings, SimpleDirectoryReader
    from llama_index.llms.ollama import Ollama

    Settings.llm = Ollama(model="gpt-oss:20b", base_url="http://127.0.0.1:11434", request_timeout=300.0)

    docs = SimpleDirectoryReader(
        input_files=sorted((BASE_DIR / "data").glob("*.pdf")),
        required_exts=[".pdf"],
        filename_as_id=True,
    ).load_data()

    text = "\n\n".join(d.get_content() for d in docs[:40])[:30000]

    llm = Settings.llm
    prompt = f"""Genera exactamente 12 preguntas de opción múltiple (4 opciones A-D, 1 correcta) sobre el texto siguiente.
    Formato JSON estricto, una lista de objetos:
    [{{"question": "...", "correct_answer": "...", "reference_contexts": ["fragmento del texto que sustenta la respuesta"]}}]
    Las preguntas deben cubrir: procesos CRM, comandos del sistema, manejo de leads, SLAs y buenas prácticas.
    reference_contexts debe ser una lista con 1-2 fragmentos literales del texto donde está la respuesta.

    TEXTO:
    {text}"""

    response = await_llm(llm, prompt)
    try:
        start = response.index("[")
        end = response.rindex("]") + 1
        test_set = json.loads(response[start:end])
    except (ValueError, json.JSONDecodeError) as e:
        print(f"No se pudo parsear JSON del LLM: {e}")
        print(response[:500])
        return

    TEST_SET_PATH.write_text(
        json.dumps(test_set, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Test set guardado en {TEST_SET_PATH} ({len(test_set)} preguntas)")


def await_llm(llm, prompt: str) -> str:
    import asyncio

    return asyncio.run(llm.acomplete(prompt)).text


def run_eval() -> None:
    if not TEST_SET_PATH.exists():
        print(f"No existe {TEST_SET_PATH}. Ejecuta primero: python eval.py --build-test-set")
        return

    test_set = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    questions = [item["question"] for item in test_set]
    ground_truths = [[item["correct_answer"]] for item in test_set]
    reference_contexts = [item.get("reference_contexts", []) for item in test_set]

    from datasets import Dataset
    from llama_index.llms.ollama import Ollama
    from ragas import EvaluationDataset, evaluate
    from ragas.llms.base import LlamaIndexLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        context_precision,
        context_recall,
    )

    from rag_workflow import query_documents

    eval_llm = LlamaIndexLLMWrapper(
        Ollama(model="gpt-oss:20b", base_url="http://127.0.0.1:11434", request_timeout=300.0)
    )

    print(f"Evaluando {len(questions)} preguntas (puede tardar varios minutos)...")
    answers = []
    contexts = []
    for q in questions:
        result = __import__("asyncio").run(query_documents(q, rerank=False))
        answers.append(result.answer)
        contexts.append([src["text"] for src in result.sources])

    ds = Dataset.from_dict(
        {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": ground_truths,
            "reference_contexts": reference_contexts,
        }
    )
    eval_dataset = EvaluationDataset.from_hf_dataset(ds)

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        context_precision,
        context_recall,
    ]
    result = evaluate(dataset=eval_dataset, metrics=metrics, llm=eval_llm)
    df = result.to_pandas()
    print("\n=== RESULTADOS POR PREGUNTA ===")
    print(df.to_string())
    print("\n=== PROMEDIOS ===")
    print(df[["faithfulness", "response_relevancy", "context_precision", "context_recall"]].mean().round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval del RAG con Ragas")
    parser.add_argument("--build-test-set", action="store_true", help="Genera test_set.json con el LLM")
    args = parser.parse_args()
    if args.build_test_set:
        build_test_set()
    else:
        run_eval()
