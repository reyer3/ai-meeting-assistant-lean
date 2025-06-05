#!/usr/bin/env python3
"""
Setup optimizado para asistente ultra-lean
Configura componentes necesarios para máximo rendimiento
"""

import os
import sys
import subprocess
import asyncio
import time
from pathlib import Path
from loguru import logger

def print_banner():
    """Banner de setup"""
    print("\n" + "🚀" * 60)
    print("AI MEETING ASSISTANT - SETUP ULTRA-OPTIMIZADO")
    print("🚀" * 60)
    print("Configurando asistente para máximo rendimiento...")
    print("Enfoque: Sugerencias ANTES de que hables\n")

def check_system_requirements():
    """Verifica requerimientos del sistema"""
    logger.info("🔍 Verificando requerimientos del sistema...")
    
    issues = []
    
    # Python version
    if sys.version_info < (3, 8):
        issues.append("❌ Python 3.8+ requerido")
    else:
        logger.success(f"✅ Python {sys.version.split()[0]}")
    
    # RAM available
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < 6:
            issues.append(f"⚠️ RAM: {ram_gb:.1f}GB (recomendado: 6GB+)")
        else:
            logger.success(f"✅ RAM: {ram_gb:.1f}GB")
    except ImportError:
        logger.warning("⚠️ No se pudo verificar RAM")
    
    # Audio system
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            issues.append("❌ No se detectaron dispositivos de audio de entrada")
        else:
            logger.success(f"✅ {len(input_devices)} dispositivos de audio detectados")
    except Exception as e:
        logger.warning(f"⚠️ Error verificando audio: {e}")
    
    return issues

def install_optimized_dependencies():
    """Instala dependencias optimizadas"""
    logger.info("📦 Instalando dependencias optimizadas...")
    
    optimized_requirements = [
        "whisper-cpp-python==0.1.2",  # Whisper optimizado
        "numpy==1.24.3",              # Versión estable
        "httpx[http2]==0.25.2",       # Cliente HTTP optimizado
        "psutil==5.9.6",              # Monitoreo sistema
        "loguru==0.7.2",              # Logging eficiente
    ]
    
    for req in optimized_requirements:
        try:
            logger.info(f"Instalando {req}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                req, "--upgrade", "--quiet"
            ], check=True)
            logger.success(f"✅ {req}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error instalando {req}: {e}")
            return False
    
    return True

def setup_ollama_optimized():
    """Setup de Ollama con configuración optimizada"""
    logger.info("🧠 Configurando Ollama optimizado...")
    
    # Verificar si Ollama está instalado
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.success("✅ Ollama ya instalado")
        else:
            logger.error("❌ Ollama no encontrado")
            return False
    except FileNotFoundError:
        logger.error("❌ Ollama no instalado")
        logger.info("💡 Instala Ollama desde: https://ollama.ai/download")
        return False
    
    # Verificar si el modelo está disponible
    try:
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True)
        
        if "qwen2.5:0.5b" in result.stdout:
            logger.success("✅ Modelo qwen2.5:0.5b ya disponible")
        else:
            logger.info("📥 Descargando modelo qwen2.5:0.5b (316MB)...")
            subprocess.run(["ollama", "pull", "qwen2.5:0.5b"], check=True)
            logger.success("✅ Modelo descargado")
    
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error con Ollama: {e}")
        return False
    
    return True

def configure_audio_optimizations():
    """Configura optimizaciones de audio específicas del sistema"""
    logger.info("🎤 Configurando optimizaciones de audio...")
    
    import platform
    system = platform.system()
    
    if system == "Windows":
        logger.info("🪟 Configuración para Windows:")
        logger.info("  1. Habilita 'Mezcla Estéreo' en Configuración de Audio")
        logger.info("  2. Ajusta calidad a 16-bit, 16000 Hz")
        logger.info("  3. Desactiva 'Mejoras de Audio'")
        
    elif system == "Darwin":  # macOS
        logger.info("🍎 Configuración para macOS:")
        logger.info("  1. Instala BlackHole para captura de audio del sistema")
        logger.info("  2. Configura Audio MIDI Setup")
        
    elif system == "Linux":
        logger.info("🐧 Configuración para Linux:")
        logger.info("  1. Configura PulseAudio loopback")
        logger.info("  2. Verifica permisos de audio")
    
    logger.success("✅ Guías de configuración mostradas")
    return True

def setup_voice_profile_optimized():
    """Setup optimizado del perfil de voz"""
    logger.info("👤 Configurando perfil de voz optimizado...")
    
    profile_path = Path("data/user_voice_profile.pkl")
    
    if profile_path.exists():
        logger.success("✅ Perfil de voz ya existe")
        return True
    
    # Crear directorio data si no existe
    profile_path.parent.mkdir(exist_ok=True)
    
    logger.info("🎙️ Se necesita crear perfil de voz personal")
    logger.info("💡 Ejecuta: python scripts/setup_voice_profile.py")
    
    return False

def setup_knowledge_base_optimized():
    """Setup optimizado de la base de conocimiento"""
    logger.info("📚 Configurando base de conocimiento...")
    
    kb_path = Path("data/chroma_db")
    
    if kb_path.exists() and any(kb_path.iterdir()):
        logger.success("✅ Base de conocimiento ya existe")
        return True
    
    logger.info("📊 Creando base de conocimiento AEIOU...")
    
    try:
        # Ejecutar setup de knowledge base
        result = subprocess.run([
            sys.executable, "scripts/setup_knowledge_base.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.success("✅ Base de conocimiento creada")
            return True
        else:
            logger.error(f"❌ Error creando base de conocimiento: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout creando base de conocimiento")
        return False
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        return False

async def test_optimized_pipeline():
    """Test del pipeline optimizado"""
    logger.info("🧪 Probando pipeline optimizado...")
    
    try:
        # Import test
        from ultimate_assistant import create_ultimate_assistant
        
        # Create assistant
        assistant = create_ultimate_assistant({
            "enable_ui_overlay": False  # Sin UI para test
        })
        
        # Test initialization
        logger.info("⚡ Probando inicialización...")
        init_start = time.time()
        
        if await assistant.initialize():
            init_time = time.time() - init_start
            logger.success(f"✅ Inicialización OK en {init_time:.2f}s")
            
            # Test LLM speed
            if assistant.llm:
                benchmark = await assistant.llm.benchmark_speed(2)
                if "avg_time" in benchmark:
                    logger.success(f"✅ LLM speed: {benchmark['avg_time']:.2f}s promedio")
            
            # Cleanup
            await assistant._graceful_shutdown()
            return True
        else:
            logger.error("❌ Fallo en inicialización")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Error de import: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error en test: {e}")
        return False

def create_optimized_startup_script():
    """Crea script de inicio optimizado"""
    logger.info("📝 Creando script de inicio optimizado...")
    
    startup_script = """#!/usr/bin/env python3
\"\"\"
Script de inicio optimizado para AI Meeting Assistant
Uso: python start_optimized.py
\"\"\"

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ultimate_assistant import create_ultimate_assistant

async def main():
    print("🚀 Iniciando AI Meeting Assistant Optimizado...")
    
    # Configuración optimizada (modifica según tu hardware)
    config = {
        "voice_confidence_threshold": 0.75,  # Ajusta si hay false positives
        "suggestion_delay": 0.6,             # 600ms después que otros terminan
        "max_generation_time": 2.5,          # 2.5s máximo para sugerencias
        "enable_ui_overlay": True,           # Cambiar a False si hay problemas
    }
    
    assistant = create_ultimate_assistant(config)
    await assistant.start_ultimate_listening()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n👋 ¡Hasta luego!")
"""
    
    script_path = Path("start_optimized.py")
    script_path.write_text(startup_script)
    logger.success(f"✅ Script creado: {script_path}")
    
    return True

def main():
    """Setup principal"""
    print_banner()
    
    success = True
    
    # 1. Check system requirements
    issues = check_system_requirements()
    if issues:
        logger.warning("⚠️ Problemas detectados:")
        for issue in issues:
            print(f"  {issue}")
        
        if not input("\\nContinuar setup? (y/n): ").lower().startswith('y'):
            return False
    
    # 2. Install dependencies
    if not install_optimized_dependencies():
        logger.error("❌ Error instalando dependencias")
        success = False
    
    # 3. Setup Ollama
    if not setup_ollama_optimized():
        logger.error("❌ Error configurando Ollama")
        success = False
    
    # 4. Audio configuration
    configure_audio_optimizations()
    
    # 5. Voice profile
    if not setup_voice_profile_optimized():
        logger.warning("⚠️ Perfil de voz pendiente")
    
    # 6. Knowledge base
    if not setup_knowledge_base_optimized():
        logger.warning("⚠️ Base de conocimiento pendiente")
    
    # 7. Create startup script
    create_optimized_startup_script()
    
    # 8. Test pipeline
    logger.info("🧪 Ejecutando test final...")
    try:
        test_result = asyncio.run(test_optimized_pipeline())
        if test_result:
            logger.success("✅ Test del pipeline exitoso")
        else:
            logger.error("❌ Test del pipeline falló")
            success = False
    except Exception as e:
        logger.error(f"❌ Error en test: {e}")
        success = False
    
    # Final report
    print("\\n" + "🎯" * 60)
    if success:
        print("✅ SETUP OPTIMIZADO COMPLETADO")
        print("🎯" * 60)
        print("🚀 Para iniciar el asistente:")
        print("   python start_optimized.py")
        print("")
        print("🎯 Para máximo rendimiento:")
        print("   - Usa audifonos para evitar feedback")
        print("   - Configura audio del sistema según tu OS")
        print("   - Crea perfil de voz si no existe")
        print("")
        print("💡 El asistente generará sugerencias AEIOU cuando")
        print("   otros terminen de hablar en reuniones")
    else:
        print("⚠️ SETUP INCOMPLETO")
        print("🎯" * 60)
        print("❌ Revisa los errores arriba y vuelve a ejecutar")
    
    print("🎯" * 60 + "\\n")
    return success

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n👋 Setup cancelado")
    except Exception as e:
        logger.error(f"❌ Error fatal en setup: {e}")
        sys.exit(1)
