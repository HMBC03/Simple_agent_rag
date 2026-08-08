# RAG Agéntico Inmobiliaria

Sistema de consulta documental con RAG agéntico: indexa manuales de procesos (CRM y manejo de leads inmobiliarios) y responde preguntas **citando el archivo, la página y el párrafo exactos** de donde salió cada afirmación, en un chat web que te lleva directo a la página del PDF. Todo corre local en una RTX 5060 Ti (16 GB VRAM) — sin nube, sin API keys.

Construido sobre **LlamaIndex Workflows** (orquestación agéntica event-driven) + **Ollama** (inferencia local). Expuesto como API REST (FastAPI, que además sirve el frontend web), como servidor **MCP** (para opencode) y como CLI.

---

## 1. Arquitectura

El sistema tiene dos fases bien separadas: la **ingesta** (una sola vez, fuera de línea) y el **query time** (cada consulta, en línea).

**Fase de ingesta.** Los PDFs de `data/` se leen con `SimpleDirectoryReader`, se dividen en fragmentos con `SentenceWindowNodeParser` (ventana de 3 oraciones alrededor de cada chunk, guardada como metadato), se transforman en vectores con el modelo de embeddings `bge-m3` (servido por Ollama) y se guardan como un `VectorStoreIndex` persistido en `storage/`. Cada nodo queda etiquetado con su **número de párrafo dentro de la página** (los párrafos se detectan como bloques separados por línea en blanco). Esta fase termina cuando el índice queda en disco; no participa ningún LLM.

**Fase de consulta (query time).** La consulta entra como un evento `QueryEvent` y recorre el `RagWorkflow`, una máquina de estados definida con LlamaIndex Workflows (eventos tipados, cada paso es una función async decorada con `@step`). El flujo es: el paso `retrieve` consulta el índice persistido con un retriever top-k (6 fragmentos por defecto); el paso `rerank` reordena esos fragmentos con el reranker `bge-reranker-v2-m3`, opcional y desactivado por defecto porque ocupa VRAM adicional; el paso `generate` expande cada fragmento con su ventana de oraciones, arma el prompt del sistema con los fragmentos numerados `[1..n]` y reglas estrictas de citado, y pide al LLM (`gpt-oss:20b` vía Ollama) que responda. El resultado es un `QueryResult` con la respuesta y la lista de `sources`, donde cada fuente es un fragmento real con su archivo, página, párrafo y score.

**Salidas.** El mismo `QueryResult` se sirve por varias fachadas: la API REST FastAPI (que además sirve el **frontend web** y los PDFs), el servidor MCP (tool `query_documents_tool` que los agentes — incluyendo opencode — pueden invocar) y la evaluación con Ragas (`eval.py`). Todas llaman a la misma función `query_documents()`, así que no hay lógica duplicada entre canales. El script `rag.bat` es un panel de control que ejecuta cualquiera de estos canales sin teclear comandos.

**Frontend web** (`frontend/index.html`, estilo Qwen/Claude): las respuestas citan con chips `[n]` que abren el **PDF en la página exacta** dentro del visor del navegador (`#page=N`); cada fuente muestra `p.X · párr.Y`, barra de relevancia y un botón "Abrir PDF en la página X". Un panel lateral derecho **indexa los documentos** incluidos en el RAG (con páginas y tamaño, vía `GET /documents`) y cada uno abre su PDF. La conversación se guarda en `localStorage` y se restaura al recargar; tema claro/oscuro persistente; indicador de estado de la API.

### 1.1 Decisiones de diseño

- **Retrieval-first**: la calidad de una respuesta citada la pone el *retrieval*, no el LLM. Por eso la arquitectura gira alrededor del índice, el chunking con ventanas de oraciones y (opcionalmente) el reranker.
- **Citas verificables**: el prompt le dice al LLM que cite con `[n]`, pero los números los asigna *el nodo realmente recuperado*, no el modelo. El post-proceso reemplaza cada `[n]` con los metadatos reales (`file_name`, `page_label`, `paragraph`) y su score de similitud. El LLM **no puede inventar** una fuente porque las fuentes se adjuntan desde los `source_nodes` reales.
- **Un solo núcleo, tres fachadas**: API, MCP y CLI comparten `query_documents()`. No hay lógica duplicada; agregar un canal nuevo (p. ej. Streamlit) solo envuelve esa función.
- **Preparado para tools**: el mismo runtime de LlamaIndex soporta `AgentWorkflow` + `FunctionTool` + MCP. Cuando necesites herramientas (web, SQL), el núcleo de retrieval no se toca — solo se envuelve `query_documents` como una tool del agente.

### 1.2 Componentes

| Componente | Rol |
|---|---|
| `ingest.py` | Lee los PDFs de `data/`, los chunkiza con ventana de 3 oraciones, etiqueta cada nodo con su párrafo dentro de la página, genera embeddings bge-m3 y persiste el índice en `storage/`. Idempotente: si el índice existe, no reindexa (salvo `--force`). |
| `rag_workflow.py` | Núcleo. Define los eventos tipados y los 3 pasos del workflow (retrieve → rerank → generate). Carga el índice persistido en memoria una sola vez. Contiene el prompt del sistema con las reglas de citado. |
| `api.py` | FastAPI: `GET /health`, `GET /documents` (catálogo de PDFs indexados), `GET /pdf/{archivo}` (sirve el PDF con path sanitizado para que el navegador salte a `#page=N`) y `POST /query` (json con `question` y `rerank`). Devuelve `answer` + `sources[]` (ref, file, page, paragraph, score, text). Sirve el frontend desde la raíz. |
| `frontend/index.html` | Chat web estilo Qwen/Claude sin frameworks: citas `[n]` que abren el PDF en la página exacta, tarjetas de fuente con párrafo y relevancia, panel derecho con los documentos indexados, caché de conversaciones en `localStorage`, tema claro/oscuro. |
| `rag.bat` | Panel de control (menú numérico) para ingesta, CLI, API + frontend, MCP, evaluación y limpieza de puertos. |
| `mcp_server.py` | FastMCP: expone la tool `query_documents_tool(question, rerank)` por stdio (para opencode) u HTTP (`--http`). |
| `eval.py` | Genera un test set de 12 preguntas con el propio LLM (`--build-test-set`) y mide el sistema con Ragas (faithfulness, response relevancy, context precision/recall). |
| `storage/` | Índice persistido (vector store + docstore, incluye el metadato de párrafo). Se regenera con `ingest.py --force`. |
| `data/` | PDFs fuente. Cualquier PDF que agregues aquí requiere reindexar. |

---

## 2. Dependencias

- **Hardware**: GPU NVIDIA con 16 GB VRAM (RTX 5060 Ti validada). Funciona con menos si usas modelos más pequeños.
- **Software**:
  - Windows 10/11 (probado en PowerShell 5.1) — el proceso es el mismo en Linux/macOS.
  - Python 3.11+.
  - [Ollama](https://ollama.com/) corriendo en `http://127.0.0.1:11434`.
- **Modelos Ollama** (se descargan con `ollama pull`):

| Modelo | Tamaño | Uso | VRAM |
|---|---|---|---|
| `gpt-oss:20b` | ~13 GB | LLM de generación (respuestas y citas) | ~12 GB |
| `bge-m3` | ~1.2 GB | Embeddings (multilingüe, ideal español) | ~1.2 GB |
| `BAAI/bge-reranker-v2-m3` (HuggingFace, opcional) | ~2.2 GB | Reranker de precisión | ~2 GB |

> El reranker **no está en el registry de Ollama** (fue retirado). Se carga desde HuggingFace la primera vez que usas `--rerank`; requiere `pip install sentence-transformers` y descargará el modelo automáticamente. El nombre en el registry de Ollama (`bge-reranker-v2-m3` o `linux6200/bge-reranker-v2-m3`) devuelve "file does not exist".

- **Paquetes Python** (ver `requirements.txt`): `llama-index-core`, `llama-index-llms-ollama`, `llama-index-embeddings-ollama`, `llama-index-readers-file`, `fastapi`, `uvicorn`, `mcp[cli]`, `fastmcp`, `pydantic`, `pypdf`. Para la evaluación: `ragas==0.4.3`, `datasets`, `langchain`, `langchain-core>=1.4.7`, `langchain-community==0.4.2`, `langchain-google-vertexai` + un shim.

---

## 3. Cómo se construyó (proceso)

1. **Benchmark de frameworks** (investigación previa): se compararon LangGraph, LlamaIndex Workflows, Haystack, DSPy, OpenAI Agents SDK, Mastra, Microsoft Agent Framework y Google ADK. Criterios: calidad de retrieval, overhead por query (~6 ms LlamaIndex vs ~14 ms LangGraph), tokens por query (~1.6k vs ~2.0k), madurez de citas y facilidad para modelos locales. **LlamaIndex Workflows ganó** por ser retrieval-first: su capa de ingesta/chunking/recuperación es la que determina si encuentras el dato exacto, y su orquestación event-driven es suficiente para el flujo retrieve→grade→answer sin el peso de un state-machine.
2. **Configuración**: modelo `gpt-oss:20b` registrado en `~/.config/opencode/opencode.json` (proveedor `ollama` con `@ai-sdk/openai-compatible`), para poder seleccionarlo en opencode junto a `qwen3.5:27b`, `gemma4`, etc.
3. **Ingesta**: los 2 PDFs (~24 KB c/u) → `SimpleDirectoryReader` → 292 nodos con `SentenceWindowNodeParser` (ventana de 3 oraciones) → embeddings `bge-m3` → `VectorStoreIndex` persistido.
4. **Workflow**: primero se escribió como `retrieve → generate` y se validó consulta a consulta. Bug notable: un atributo de instancia `self.rerank` (bool) **enmascaraba el método** `rerank` del workflow y el paso no se registraba — el validador falló con "RerankEvent consumed but never produced". Se renombró a `self.use_rerank`. Luego se añadió el paso `rerank` opcional.
5. **API + MCP**: FastAPI para consumo humano/curl; FastMCP (paquete `fastmcp`, separado del `mcp` core en la v2) para que opencode consulte el RAG como tool.
6. **Evaluación**: test set de 12 preguntas generado por el propio LLM local, métricas Ragas con el evaluador apuntando a Ollama (sin usar OpenAI).
7. **Frontend web**: chat estilo Qwen/Claude (solo HTML+CSS+JS, sin frameworks). Se añadió el metadato de **párrafo** en la ingesta, endpoints `GET /documents` y `GET /pdf/{archivo}` en la API, salto directo al PDF (`#page=N` en el visor del navegador), panel derecho con el catálogo de documentos y caché de conversaciones en `localStorage`. Bug detectado en el camino: abrir `index.html` con **Live Server** (puerto 5500) hacía que el fetch relativo `/query` pegara en 5500 y devolviera HTTP 405 — el frontend detecta `location.port` y apunta a `http://127.0.0.1:8000` (CORS `*`).

---

## 4. Cómo correrlo — comando por comando

### 4.1 Prerequisitos (una sola vez)

```powershell
# 1. Instalar Ollama y descargar modelos
ollama pull gpt-oss:20b
ollama pull bge-m3

# 2. Crear entorno virtual e instalar dependencias (desde la raíz del proyecto)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. (Opcional) Para el reranker — fuera del registry de Ollama
.\.venv\Scripts\python.exe -m pip install sentence-transformers
```

### 4.2 Ingesta

```powershell
# Indexa data/*.pdf → storage/. Si ya existe, no hace nada.
.\.venv\Scripts\python.exe ingest.py

# Forzar reindexación (tras agregar/modificar PDFs)
.\.venv\Scripts\python.exe ingest.py --force
```

La primera ejecución tarda (genera embeddings), las siguientes son instantáneas.

### 4.3 Consulta directa (CLI)

```powershell
# Consulta simple
.\.venv\Scripts\python.exe rag_workflow.py "¿Cuáles son las etapas del proceso de manejo de leads?"

# Con reranker (más preciso, ocupa ~2 GB extra de VRAM)
.\.venv\Scripts\python.exe rag_workflow.py "¿Cuál es el SLA de primer contacto?" --rerank
```

Cada fuente se imprime como `[1] archivo.pdf p.3 párr.2 (score 0.60)`.

### 4.4 API REST (FastAPI)

```powershell
# Terminal 1 — levantar el servidor (sirve API + frontend + PDFs)
.\.venv\Scripts\python.exe api.py        # escucha en http://127.0.0.1:8000
# (o con otro puerto: api.py --port 8080)

# Terminal 2 — probar
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/documents
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" `
  -d '{"question": "¿Qué comando del CRM registra una oportunidad?", "rerank": false}'
```

Respuesta: `{"answer": "... [1][2]", "sources": [{"ref": 1, "file": "...pdf", "page": "7", "paragraph": 3, "score": 0.63, "text": "..."}]}`. El frontend se sirve en la raíz (`http://127.0.0.1:8000/`) y los PDFs en `http://127.0.0.1:8000/pdf/{archivo}#page=N` (el visor del navegador salta a la página). Documentación interactiva en `http://127.0.0.1:8000/docs` (Swagger UI servido localmente desde `frontend/vendor/swagger-ui/`, sin CDN — funciona sin internet).

### 4.5 Frontend web

Abre `http://127.0.0.1:8000/` con la API corriendo (opción 5 de `rag.bat`). Funciones:

- **Citas `[n]`**: cada chip abre el PDF en la página exacta de la fuente.
- **Fuentes**: cada tarjeta muestra `p.X · párr.Y`, relevancia y un botón "Abrir PDF en la página X".
- **Panel derecho "Documentos"**: catálogo de los PDFs indexados (`GET /documents`); clic abre el PDF.
- **Caché**: la conversación se guarda en `localStorage` y se restaura al recargar; el botón "Nueva" la borra.
- **Tema** claro/oscuro persistente e indicador de estado de la API.

> Si abres `frontend/index.html` con Live Server de VS Code (u otro servidor estático), el frontend detecta que no está en el puerto 8000 y llama a la API en `http://127.0.0.1:8000` directamente — funciona igual, siempre que la API esté arriba.

### 4.6 MCP (para opencode y otros agentes)

```powershell
# stdio (por defecto — así lo consume opencode)
.\.venv\Scripts\python.exe mcp_server.py

# HTTP (para otras herramientas)
.\.venv\Scripts\python.exe mcp_server.py --http
```

Ya está registrado en `~/.config/opencode/opencode.json`:

```json
"mcp": {
  "rag-inmobiliaria": {
    "type": "local",
    "command": ["C:\\...\\Agent\\.venv\\Scripts\\python.exe", "C:\\...\\Agent\\mcp_server.py"],
    "enabled": true
  }
}
```

Tras reiniciar opencode, cualquier agente puede invocar la tool `query_documents_tool(question, rerank)` — es decir, **opencode consulta tus documentos citando página, párrafo y archivo**.

### 4.7 Evaluación

```powershell
# 1. Generar test set (12 preguntas, usa el LLM local ~2-4 min)
.\.venv\Scripts\python.exe eval.py --build-test-set

# 2. Correr la evaluación (12 consultas + juicio LLM, ~10-20 min)
.\.venv\Scripts\python.exe eval.py
```

Métricas: `faithfulness` (¿la respuesta se basa solo en el contexto?), `response_relevancy`, `context_precision` (¿los fragmentos recuperados son útiles?), `context_recall` (¿se recuperó todo lo necesario?).

---

## 5. Cómo modificarlo

### 5.1 Cambiar el LLM (generación)

En `rag_workflow.py`, `LLM_MODEL = "gpt-oss:20b"`:

- **Otro local en Ollama**: `LLM_MODEL = "qwen3.5:27b"` o `"gemma4:latest"` (primero `ollama pull`). Los 3 pasos del workflow no cambian.
- **API de un proveedor** (OpenAI/Anthropic/Gemini): instala el paquete (ej. `pip install llama-index-llms-openai`), configura la API key y reemplaza el `Settings.llm` en `_ensure_loaded()`:

```python
from llama_index.llms.openai import OpenAI
Settings.llm = OpenAI(model="gpt-4o-mini", api_key="sk-...", request_timeout=300.0)
```

El resto del flujo (retrieval, citas, post-proceso) es agnóstico al LLM.

### 5.2 Cambiar los embeddings

En `ingest.py` y `rag_workflow.py`, `EMBED_MODEL = "bge-m3"`. Importante: **si cambias el modelo de embeddings, reindexa con `--force`** — los vectores guardados no son compatibles entre modelos. Ejemplo alternativo local: `nomic-embed-text`; de API: `OpenAIEmbedding`.

### 5.3 Ajustar el retrieval

En `rag_workflow.py` (constantes de clase):

| Constante | Efecto |
|---|---|
| `TOP_K` (6) | Fragmentos recuperados por consulta. Subir mejora recall, baja precisión y aumenta tokens/VRAM. |
| `CONTEXT_BUDGET` (4000) | Tope de caracteres de contexto por consulta. Bájalo si el modelo se desvía o para acelerar. |
| `window_size` en `ingest.py` (3) | Oraciones alrededor de cada chunk que se dan al LLM (con `MetadataReplacementPostProcessor`). Subirlo da más contexto por cita. |

### 5.4 Reranker

- Se activa por consulta: `--rerank` (CLI), `"rerank": true` (API/MCP).
- `_load_reranker()` en `rag_workflow.py`: modelo `BAAI/bge-reranker-v2-m3`, `top_n=4`, `device="cuda"`. Cambia el modelo o `top_n` ahí.
- Si la VRAM no alcanza (LLM + reranker simultáneos), usa `device="cpu"` (más lento, sin VRAM extra).

### 5.5 Cambiar el prompt / estilo de citas

`SYSTEM_PROMPT` en `rag_workflow.py`. Ahí viven las reglas: idioma, obligatoriedad de citas `[n]`, qué hacer cuando no hay información (debe responder "No encontré esa información..." en lugar de inventar) y concisión. El formato de las fuentes emitidas se ajusta en `generate()` (el dict de `sources`).

### 5.6 Agregar documentos

```powershell
Copy-Item "nuevo.pdf" data\
.\.venv\Scripts\python.exe ingest.py --force
```

### 5.7 Agregar herramientas (futuro)

El núcleo está listo para volverse agente sin reescribirlo. Patrón: envuelve el query engine como una tool y deja que un `AgentWorkflow` decida cuándo consultarlo:

```python
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import AgentWorkflow

tool = FunctionTool.from_defaults(
    fn=query_documents,  # la misma función del núcleo
    name="consultar_manuales",
    description="Consulta los manuales y responde citando archivo y página",
)
agent = AgentWorkflow.from_tools([tool], llm=Settings.llm)
```

Agregar una tool de búsqueda web o SQL es añadir otro `FunctionTool` a esa lista — el índice y las citas no cambian.

---

## 6. Estado actual

-  Ingesta de 2 PDFs (292 nodos, con metadato de párrafo) persistida en `storage/`
-  Workflow agéntico con citas verificables (archivo + página + párrafo + score)
-  API FastAPI (`/query`, `/health`, `/documents`, `/pdf/{archivo}`)
-  Frontend web estilo Qwen/Claude: citas → PDF en la página exacta, panel de documentos, caché de conversaciones, tema claro/oscuro
-  Servidor MCP registrado en opencode
-  Evaluación Ragas con LLM local (12 preguntas)
-  Reranker disponible pero desactivado por defecto (ocupa VRAM adicional con el LLM cargado)
- `AgentWorkflow` con tools adicionales (diseñado, no implementado)
