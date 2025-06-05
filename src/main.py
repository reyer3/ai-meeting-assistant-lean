#!/usr/bin/env python3
"""
AI Meeting Assistant Lean - Punto de entrada principal
Enfoque: App de escritorio simple que captura audio sin unirse a reuniones
"""

import asyncio
import sys
import signal
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Agregar src al path
sys.path.append(str(Path(__file__).parent))

from lean_pipeline import LeanMeetingAssistant, create_lean_assistant

console = Console()

class LeanApp:
    """Aplicación principal lean simplificada"""
    
    def __init__(self):
        self.assistant: Optional[LeanMeetingAssistant] = None
        self.is_running = False
        
    def setup_logging(self):
        """Configuración simple de logging"""
        logger.remove()
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
        )
        
        # Crear directorio logs si no existe
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        logger.add(
            "logs/lean_assistant.log",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days"
        )
    
    def show_banner(self):
        """Muestra banner inicial"""
        console.print(Panel.fit(
            "[bold blue]🎯 AI Meeting Assistant Lean[/bold blue]\n"
            "[dim]Asistente de reuniones con IA local y framework AEIOU[/dim]\n\n"
            "✅ 100% Local y Privado\n"
            "🎤 Reconoce tu voz automáticamente\n" 
            "💬 Sugerencias AEIOU contextuales\n"
            "⚡ Sin GPU requerida\n"
            "🗄️ Base de conocimiento RAG local",
            border_style="blue"
        ))
    
    def check_prerequisites(self) -> bool:
        """Verifica prerrequisitos del sistema"""
        console.print("\n[cyan]🔍 Verificando prerrequisitos...[/cyan]")
        
        all_good = True
        
        # 1. Verificar perfil de voz
        voice_profile_path = Path("data/user_voice_profile.pkl")
        if not voice_profile_path.exists():
            console.print("[red]❌ Perfil de voz no encontrado[/red]")
            console.print("[cyan]💡 Ejecuta: python scripts/setup_voice_profile.py[/cyan]")
            all_good = False
        else:
            console.print("[green]✅ Perfil de voz disponible[/green]")
        
        # 2. Verificar Ollama y modelo
        import subprocess
        try:
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=5)
            if "qwen2.5:0.5b" not in result.stdout:
                console.print("[red]❌ Modelo qwen2.5:0.5b no encontrado[/red]")
                console.print("[cyan]💡 Ejecuta: ollama pull qwen2.5:0.5b[/cyan]")
                all_good = False
            else:
                console.print("[green]✅ Modelo qwen2.5:0.5b disponible[/green]")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            console.print("[red]❌ Ollama no está disponible[/red]")
            console.print("[cyan]💡 Instala Ollama desde: https://ollama.ai[/cyan]")
            all_good = False
        
        # 3. Verificar ChromaDB (knowledge base)
        chroma_path = Path("data/chroma_db")
        if not chroma_path.exists():
            console.print("[red]❌ Base de conocimiento no inicializada[/red]")
            console.print("[cyan]💡 Ejecuta: python scripts/setup_knowledge_base.py[/cyan]")
            all_good = False
        else:
            console.print("[green]✅ Base de conocimiento disponible[/green]")
        
        # 4. Verificar dependencias críticas
        try:
            import whisper
            console.print("[green]✅ Whisper disponible[/green]")
        except ImportError:
            console.print("[red]❌ Whisper no instalado[/red]")
            console.print("[cyan]💡 Ejecuta: pip install openai-whisper[/cyan]")
            all_good = False
        
        try:
            import chromadb
            console.print("[green]✅ ChromaDB disponible[/green]")
        except ImportError:
            console.print("[red]❌ ChromaDB no instalado[/red]")
            console.print("[cyan]💡 Ejecuta: pip install chromadb[/cyan]")
            all_good = False
        
        # 5. Verificar audio
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if input_devices:
                console.print(f"[green]✅ {len(input_devices)} dispositivos de audio[/green]")
            else:
                console.print("[yellow]⚠️ No se encontraron dispositivos de entrada[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Error verificando audio: {e}[/yellow]")
        
        if all_good:
            console.print("[green]✅ Todos los prerrequisitos están listos[/green]")
        else:
            console.print("\n[yellow]💡 Instalación automática disponible:[/yellow]")
            console.print("[bold]python scripts/install_lean.py[/bold]")
        
        return all_good
    
    def setup_signal_handlers(self):
        """Configura manejadores de señales para shutdown graceful"""
        def signal_handler(signum, frame):
            logger.info("🛑 Señal de interrupción recibida")
            if self.assistant:
                asyncio.create_task(self.assistant.stop())
            self.is_running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Ejecuta la aplicación principal"""
        self.setup_logging()
        self.show_banner()
        
        # Verificar prerrequisitos
        if not self.check_prerequisites():
            console.print("\n[yellow]⚠️ Algunos prerrequisitos faltan[/yellow]")
            
            if not Prompt.ask("¿Continuar de todas formas?", choices=["y", "n"], default="n") == "y":
                console.print("\n[cyan]Para instalación automática, ejecuta:[/cyan]")
                console.print("[bold]python scripts/install_lean.py[/bold]")
                return 1
        
        # Configurar señales
        self.setup_signal_handlers()
        
        try:
            # Crear asistente lean
            logger.info("🚀 Inicializando asistente lean...")
            self.assistant = create_lean_assistant()
            
            # Mostrar configuración
            self.show_configuration()
            
            # Confirmar inicio
            start_now = Prompt.ask(
                "\n¿Iniciar escucha de audio?",
                choices=["y", "n"],
                default="y"
            )
            
            if start_now != "y":
                console.print("[yellow]👋 Hasta luego![/yellow]")
                return 0
            
            # Iniciar asistente
            self.is_running = True
            console.print("\n[bold green]🎧 Iniciando escucha de audio del sistema...[/bold green]")
            console.print("[dim]Habla normalmente en tus reuniones. El asistente detectará tu voz y generará sugerencias AEIOU cuando detecte tensión o conflictos.[/dim]")
            console.print("\n[yellow]Controles:[/yellow]")
            console.print("  [cyan]Ctrl+C[/cyan] - Detener asistente")
            console.print("\n" + "="*80 + "\n")
            
            await self.assistant.start_listening()
            
        except KeyboardInterrupt:
            logger.info("⏹️ Detenido por usuario")
        except Exception as e:
            logger.error(f"❌ Error fatal: {e}")
            console.print(f"[bold red]❌ Error fatal:[/bold red] {e}")
            console.print("\n[cyan]Para obtener ayuda:[/cyan]")
            console.print("  1. Verifica los logs en: logs/lean_assistant.log")
            console.print("  2. Ejecuta: python scripts/install_lean.py")
            console.print("  3. Abre un issue en GitHub con los logs")
            return 1
        finally:
            if self.assistant:
                await self.assistant.stop()
            logger.info("👋 Aplicación cerrada")
        
        return 0
    
    def show_configuration(self):
        """Muestra la configuración actual"""
        console.print("\n[bold cyan]📋 Configuración Lean:[/bold cyan]")
        console.print("  🎤 Audio: Captura de audio del sistema (16kHz)")
        console.print("  👤 Identificación: Diferenciación automática de tu voz") 
        console.print("  🗣️ STT: Whisper base local (sin internet)")
        console.print("  🧠 IA: qwen2.5:0.5b (316MB, sin GPU)")
        console.print("  📚 RAG: ChromaDB local con ejemplos AEIOU")
        console.print("  ⚡ Latencia: <6s por sugerencia")
        console.print("  💬 Framework: AEIOU para comunicación no-violenta")
        console.print("  🔒 Privacidad: 100% local, sin datos en la nube")


def main():
    """Punto de entrada principal"""
    app = LeanApp()
    
    try:
        exit_code = asyncio.run(app.run())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Hasta luego![/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
