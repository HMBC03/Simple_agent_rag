@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

set "PY=.\.venv\Scripts\python.exe"

echo ============================================
echo   RAG INMOBILIARIA - Panel de control
echo ============================================
echo.

if not exist "%PY%" (
    echo [1/3] Creando entorno virtual...
    python -m venv .venv
    if errorlevel 1 ( echo ERROR creando el venv & pause & exit /b 1 )
    echo [2/3] Instalando dependencias...
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 ( echo ERROR instalando dependencias & pause & exit /b 1 )
    echo [3/3] Listo.
    echo.
)

:menu
echo Elige una opcion:
echo   1) Instalar/actualizar dependencias
echo   2) Verificar modelos en Ollama
echo   3) Ingesta (indexar PDFs de data/)
echo   4) Consulta rapida (CLI con citas)
echo   5) Levantar API REST + frontend (http://127.0.0.1:8000)
echo   6) Levantar servidor MCP (para opencode)
echo   7) Generar test set de evaluacion
echo   8) Correr evaluacion Ragas
echo   9) Configurar MCP en opencode.json
echo   0) Salir
echo.
set /p op=Opcion: 

if "%op%"=="1" goto install
if "%op%"=="2" goto models
if "%op%"=="3" goto ingest
if "%op%"=="4" goto query
if "%op%"=="5" goto api
if "%op%"=="6" goto mcp
if "%op%"=="7" goto gen_test
if "%op%"=="8" goto eval
if "%op%"=="9" goto configure_mcp
if "%op%"=="0" exit /b 0
echo Opcion invalida. & echo. & goto menu

:install
echo Instalando dependencias...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 ( echo ERROR & pause & exit /b 1 )
echo Listo. & pause & goto menu

:models
echo Modelos locales en Ollama:
ollama list
echo.
echo Verificando modelo principal (se descarga si falta)...
ollama show gpt-oss:20b >nul 2>&1
if errorlevel 1 (
    echo Descargando gpt-oss:20b (~13 GB)...
    ollama pull gpt-oss:20b
)
ollama show bge-m3 >nul 2>&1
if errorlevel 1 (
    echo Descargando bge-m3 (~1.2 GB)...
    ollama pull bge-m3
)
echo Listo. & pause & goto menu

:ingest
set /p force=Reindexar aunque exista el indice? [s/N]: 
if /i "%force%"=="s" (
    "%PY%" ingest.py --force
) else (
    "%PY%" ingest.py
)
if errorlevel 1 ( echo ERROR & pause & exit /b 1 )
echo Listo. & pause & goto menu

:query
set /p q=Pregunta: 
if "%q%"=="" set "q=Como se asigna un lead a un asesor?"
set /p rk=Usar reranker? [s/N]: 
if /i "%rk%"=="s" (
    "%PY%" rag_workflow.py "%q%" --rerank
) else (
    "%PY%" rag_workflow.py "%q%"
)
echo.
pause & goto menu

:api
echo Levantando API + frontend en http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
"%PY%" api.py
pause & goto menu

:mcp
echo Levantando servidor MCP por stdio (para opencode)...
"%PY%" mcp_server.py
pause & goto menu

:gen_test
echo Generando test set (usa el LLM local, ~2-4 min)...
"%PY%" eval.py --build-test-set
echo. & pause & goto menu

:eval
echo Corriendo evaluacion Ragas (~10-20 min)...
"%PY%" eval.py
echo. & pause & goto menu

:configure_mcp
set "CFG=%USERPROFILE%\.config\opencode\opencode.json"
if not exist "%CFG%" ( echo No existe %CFG% & pause & goto menu )
set "ABS=%~dp0"
set "ABS=%ABS:\=\\%"
"%PY%" -c "import json,pathlib; p=pathlib.Path(r'%CFG%'); c=json.loads(p.read_text(encoding='utf-8')); c.setdefault('mcp',{})['rag-inmobiliaria']={'type':'local','command':[r'%ABS%.venv\Scripts\python.exe', r'%ABS%mcp_server.py'],'enabled':True}; p.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8'); print('MCP rag-inmobiliaria configurado en', p)"
echo Reinicia opencode para aplicar el cambio. & pause & goto menu
