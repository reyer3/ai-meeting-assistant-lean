#!/bin/bash

# AI Collections Assistant - Script de Instalación
# Instala y configura el sistema completo para call centers de cobranza

set -e  # Salir en caso de error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
PYTHON_VERSION="3.11"
VENV_NAME="venv"
LOG_FILE="install.log"

echo -e "${BLUE}🚀 AI Collections Assistant - Instalación${NC}"
echo "==============================================="

# Función para logging
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
    echo -e "$1"
}

# Verificar Python
check_python() {
    log "${BLUE}Verificando Python ${PYTHON_VERSION}...${NC}"
    
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        log "Python encontrado: v${PYTHON_VER}"
        
        if [[ "$(printf '%s\n' "$PYTHON_VERSION" "$PYTHON_VER" | sort -V | head -n1)" == "$PYTHON_VERSION" ]]; then
            log "${GREEN}✅ Python ${PYTHON_VER} es compatible${NC}"
        else
            log "${RED}❌ Python ${PYTHON_VER} es muy antiguo. Se requiere ${PYTHON_VERSION}+${NC}"
            exit 1
        fi
    else
        log "${RED}❌ Python no encontrado. Instala Python ${PYTHON_VERSION}+ primero.${NC}"
        exit 1
    fi
}

# Verificar dependencias del sistema
check_system_deps() {
    log "${BLUE}Verificando dependencias del sistema...${NC}"
    
    # Verificar git
    if ! command -v git &> /dev/null; then
        log "${RED}❌ Git no encontrado. Por favor instala git.${NC}"
        exit 1
    fi
    
    # Verificar herramientas de audio (Linux)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if ! command -v pulseaudio &> /dev/null && ! command -v pipewire &> /dev/null; then
            log "${YELLOW}⚠️ PulseAudio o PipeWire no encontrado. Puede ser necesario para captura de audio.${NC}"
        fi
    fi
    
    log "${GREEN}✅ Dependencias del sistema verificadas${NC}"
}

# Crear entorno virtual
setup_venv() {
    log "${BLUE}Configurando entorno virtual...${NC}"
    
    if [ -d "$VENV_NAME" ]; then
        log "${YELLOW}Entorno virtual existente encontrado. Eliminando...${NC}"
        rm -rf "$VENV_NAME"
    fi
    
    python3 -m venv "$VENV_NAME"
    source "$VENV_NAME/bin/activate"
    
    # Actualizar pip
    pip install --upgrade pip setuptools wheel
    
    log "${GREEN}✅ Entorno virtual creado${NC}"
}

# Instalar dependencias Python
install_python_deps() {
    log "${BLUE}Instalando dependencias Python...${NC}"
    
    if [ ! -f "requirements.txt" ]; then
        log "${RED}❌ requirements.txt no encontrado${NC}"
        exit 1
    fi
    
    pip install -r requirements.txt
    
    log "${GREEN}✅ Dependencias Python instaladas${NC}"
}

# Verificar e instalar Ollama
setup_ollama() {
    log "${BLUE}Configurando Ollama...${NC}"
    
    if ! command -v ollama &> /dev/null; then
        log "${YELLOW}Ollama no encontrado. Instalando...${NC}"
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew install ollama
            else
                log "${RED}❌ Homebrew no encontrado. Instala Ollama manualmente desde https://ollama.ai${NC}"
                return 1
            fi
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            curl -fsSL https://ollama.ai/install.sh | sh
        else
            log "${RED}❌ OS no soportado para instalación automática de Ollama${NC}"
            log "${YELLOW}Instala Ollama manualmente desde https://ollama.ai${NC}"
            return 1
        fi
    fi
    
    # Verificar que Ollama esté ejecutándose
    if ! pgrep -x "ollama" > /dev/null; then
        log "${YELLOW}Iniciando Ollama...${NC}"
        ollama serve &
        sleep 5
    fi
    
    # Descargar modelo Qwen
    log "${BLUE}Descargando modelo Qwen 2.5:0.5b...${NC}"
    ollama pull qwen2.5:0.5b
    
    log "${GREEN}✅ Ollama configurado${NC}"
}

# Crear estructura de directorios
create_directories() {
    log "${BLUE}Creando estructura de directorios...${NC}"
    
    directories=(
        "data/chroma"
        "data/voice_profiles"
        "data/dashboard"
        "data/session_reports"
        "logs"
        "models/whisper"
        "models/embeddings"
        "knowledge_base/aeiou_examples"
        "knowledge_base/conflict_patterns"
        "exports"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log "Created: $dir"
    done
    
    log "${GREEN}✅ Estructura de directorios creada${NC}"
}

# Configurar archivos de configuración
setup_config() {
    log "${BLUE}Configurando archivos de configuración...${NC}"
    
    # Copiar configuración de collections si no existe
    if [ ! -f "data/config.json" ]; then
        cp "collections_config.json" "data/config.json"
        log "Configuración de collections copiada"
    fi
    
    # Configurar CRM (disabled por defecto)
    if [ ! -f "data/crm_config.json" ]; then
        cp "crm_config.json" "data/crm_config.json"
        log "Configuración de CRM copiada"
    fi
    
    log "${GREEN}✅ Configuración completada${NC}"
}

# Poblar knowledge base
setup_knowledge_base() {
    log "${BLUE}Poblando knowledge base...${NC}"
    
    python src/rag/collections_knowledge_base.py
    
    log "${GREEN}✅ Knowledge base poblada${NC}"
}

# Ejecutar tests
run_tests() {
    log "${BLUE}Ejecutando tests...${NC}"
    
    if python -m pytest tests/ -v; then
        log "${GREEN}✅ Todos los tests pasaron${NC}"
    else
        log "${YELLOW}⚠️ Algunos tests fallaron. El sistema puede funcionar pero revisa los logs.${NC}"
    fi
}

# Crear script de inicio
create_start_script() {
    log "${BLUE}Creando script de inicio...${NC}"
    
    cat > start_collections.sh << 'EOF'
#!/bin/bash

# AI Collections Assistant - Script de Inicio

echo "Iniciando AI Collections Assistant..."

# Activar entorno virtual
source venv/bin/activate

# Verificar que Ollama esté ejecutándose
if ! pgrep -x "ollama" > /dev/null; then
    echo "Iniciando Ollama..."
    ollama serve &
    sleep 3
fi

# Ejecutar aplicación
python src/main_collections.py
EOF
    
    chmod +x start_collections.sh
    
    log "${GREEN}✅ Script de inicio creado: ./start_collections.sh${NC}"
}

# Mostrar resumen final
show_summary() {
    echo
    echo "==============================================="
    echo -e "${GREEN}🎉 Instalación completada exitosamente!${NC}"
    echo "==============================================="
    echo
    echo -e "${BLUE}Para empezar a usar el sistema:${NC}"
    echo "1. Configurar perfil de voz:"
    echo -e "   ${YELLOW}source venv/bin/activate && python src/voice_profile_setup.py${NC}"
    echo
    echo "2. Iniciar el asistente:"
    echo -e "   ${YELLOW}./start_collections.sh${NC}"
    echo
    echo -e "${BLUE}Archivos importantes:${NC}"
    echo "- Configuración: data/config.json"
    echo "- CRM: data/crm_config.json"
    echo "- Logs: logs/"
    echo "- Knowledge Base: data/chroma/"
    echo
    echo -e "${BLUE}Documentación:${NC}"
    echo "- Setup Guide: docs/collections-setup-guide.md"
    echo "- README Collections: README-Collections.md"
    echo
    echo -e "${YELLOW}Log de instalación guardado en: $LOG_FILE${NC}"
}

# Manejar interrupciones
trap 'echo -e "\n${RED}Instalación interrumpida${NC}"; exit 1' INT TERM

# Ejecutar instalación
main() {
    log "${BLUE}Iniciando instalación en $(date)${NC}"
    
    check_python
    check_system_deps
    setup_venv
    install_python_deps
    setup_ollama
    create_directories
    setup_config
    setup_knowledge_base
    run_tests
    create_start_script
    
    show_summary
}

# Verificar que estamos en el directorio correcto
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo -e "${RED}❌ Error: Ejecuta este script desde el directorio raíz del proyecto${NC}"
    exit 1
fi

# Ejecutar instalación
main "$@"
