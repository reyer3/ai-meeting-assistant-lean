@echo off
REM AI Collections Assistant - Script de Inicio para Windows

echo Iniciando AI Collections Assistant...

REM Verificar que estamos en el directorio correcto
if not exist "src\main_collections.py" (
    echo Error: No se encontro main_collections.py
    echo Ejecuta este script desde el directorio raiz del proyecto
    pause
    exit /b 1
)

REM Activar entorno virtual
if not exist "venv\Scripts\activate.bat" (
    echo Error: Entorno virtual no encontrado
    echo Ejecuta primero: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

REM Verificar Ollama
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo Iniciando Ollama...
    start "" "ollama" serve
    timeout /t 3 /nobreak >nul
)

REM Ejecutar aplicacion
python src\main_collections.py

if %ERRORLEVEL% neq 0 (
    echo Error ejecutando la aplicacion
    pause
)

deactivate
