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
            "⚡ Sin GPU requerida",
            border_style="blue"
        ))
    
    def check_prerequisites(self) -> bool:
        """Verifica prerrequisitos del sistema"""
        
        # 1. Verificar perfil de voz
        voice_profile_path = Path("data/user_voice_profile.pkl")
        if not voice_profile_path.exists():
            console.print("[yellow]⚠️ Perfil de voz no encontrado[/yellow]")
            
            create_profile = Prompt.ask(
                "¿Quieres crear tu perfil de voz ahora?",
                choices=["y", "n"],
                default="y"
            )
            
            if create_profile == "y":
                console.print("[cyan]📝 Ejecuta: python scripts/setup_voice_profile.py[/cyan]")
                return False
            else:
                console.print("[red]❌ Perfil de voz requerido para funcionar[/red]")
                return False
        
        # 2. Verificar Ollama
        import subprocess
        try:
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=5)
            if "qwen2.5:0.5b" not in result.stdout:
                console.print("[yellow]⚠️ Modelo qwen2.5:0.5b no encontrado[/yellow]")
                console.print("[cyan]📝 Ejecuta: ollama pull qwen2.5:0.5b[/cyan]")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            console.print("[yellow]⚠️ Ollama no está disponible[/yellow]")
            console.print("[cyan]📝 Instala Ollama desde: https://ollama.ai[/cyan]")
            return False
        
        # 3. Verificar ChromaDB (knowledge base)
        chroma_path = Path("data/chroma_db")
        if not chroma_path.exists():
            console.print("[yellow]⚠️ Base de conocimiento no inicializada[/yellow]")
            console.print("[cyan]📝 Ejecuta: python scripts/setup_knowledge_base.py[/cyan]")
            return False
        
        console.print("[green]✅ Todos los prerrequisitos están listos[/green]")
        return True
    
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
            console.print("\n[red]❌ No se puede continuar sin los prerrequisitos[/red]")
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
            console.print("[dim]Presiona Ctrl+C para detener[/dim]\n")
            
            await self.assistant.start_listening()
            
        except KeyboardInterrupt:
            logger.info("⏹️ Detenido por usuario")
        except Exception as e:
            logger.error(f"❌ Error fatal: {e}")
            console.print(f"[bold red]❌ Error fatal:[/bold red] {e}")
            return 1
        finally:
            if self.assistant:
                await self.assistant.stop()
            logger.info("👋 Aplicación cerrada")
        
        return 0
    
    def show_configuration(self):
        """Muestra la configuración actual"""
        console.print("\n[bold cyan]📋 Configuración Lean:[/bold cyan]")
        console.print(f"  🎤 Audio: 16kHz, buffers de 2s")
        console.print(f"  👤 Identificación de voz: umbral 0.65")
        console.print(f"  🧠 Modelo IA: qwen2.5:0.5b (sin GPU)")
        console.print(f"  📚 RAG: ChromaDB local")
        console.print(f"  ⚡ Latencia objetivo: <6s por sugerencia")
        console.print(f"  💬 Framework: AEIOU para comunicación no-violenta")


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
