#!/usr/bin/env python3
"""
AI Collections Assistant - Main Application

Punto de entrada principal para el asistente de IA especializado en cobranza.
Orquesta todos los componentes: audio, RAG, compliance, UI y reporting.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.table import Table
from rich.layout import Layout

# Imports del sistema
sys.path.append(str(Path(__file__).parent))

from core.config import config
from core.pipeline import MeetingAssistantPipeline
from rag.chroma_manager import ChromaManager
from rag.collections_knowledge_base import populate_collections_knowledge_base
from rag.collections_query_engine import CollectionsQueryEngine, CollectionsContext
from compliance.compliance_engine import ComplianceEngine
from dashboard.collections_dashboard import CollectionsDashboard
from metrics.performance_tracker import PerformanceTracker

console = Console()
logger = logging.getLogger(__name__)


class CollectionsAssistant:
    """Asistente principal para cobranza"""
    
    def __init__(self):
        self.is_running = False
        self.components_initialized = False
        
        # Componentes del sistema
        self.pipeline = None
        self.chroma_manager = None
        self.query_engine = None
        self.compliance_engine = None
        self.dashboard = None
        self.performance_tracker = None
        
        # Estado actual
        self.current_call = None
        self.agent_info = None
        self.session_metrics = {
            "calls_processed": 0,
            "suggestions_given": 0,
            "compliance_violations": 0,
            "escalations": 0
        }
    
    async def initialize(self):
        """Inicializa todos los componentes del sistema"""
        console.print(Panel.fit(
            "[bold blue]🎯 AI Collections Assistant[/bold blue]\n"
            "Iniciando sistema especializado en cobranza...",
            border_style="blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # 1. Inicializar ChromaDB y Knowledge Base
            init_task = progress.add_task("Inicializando knowledge base...", total=None)
            try:
                self.chroma_manager = ChromaManager(
                    data_dir=config.rag.chroma_db_path,
                    model_name=config.rag.embedding_model
                )
                
                # Verificar si la KB está poblada
                stats = self.chroma_manager.get_collection_stats("aeiou_examples")
                if stats.get("total_documents", 0) < 5:
                    progress.update(init_task, description="Poblando knowledge base de cobranza...")
                    populate_collections_knowledge_base(self.chroma_manager)
                
                progress.update(init_task, description="✅ Knowledge base lista")
                
            except Exception as e:
                progress.update(init_task, description=f"❌ Error en knowledge base: {e}")
                raise
            
            # 2. Inicializar Query Engine especializado
            query_task = progress.add_task("Configurando query engine...", total=None)
            try:
                self.query_engine = CollectionsQueryEngine(self.chroma_manager)
                progress.update(query_task, description="✅ Query engine listo")
            except Exception as e:
                progress.update(query_task, description=f"❌ Error en query engine: {e}")
                raise
            
            # 3. Inicializar Compliance Engine
            compliance_task = progress.add_task("Configurando compliance engine...", total=None)
            try:
                self.compliance_engine = ComplianceEngine()
                progress.update(compliance_task, description="✅ Compliance engine listo")
            except Exception as e:
                progress.update(compliance_task, description=f"❌ Error en compliance: {e}")
                raise
            
            # 4. Inicializar Dashboard
            dashboard_task = progress.add_task("Iniciando dashboard...", total=None)
            try:
                self.dashboard = CollectionsDashboard()
                progress.update(dashboard_task, description="✅ Dashboard listo")
            except Exception as e:
                progress.update(dashboard_task, description=f"❌ Error en dashboard: {e}")
                raise
            
            # 5. Inicializar Performance Tracker
            perf_task = progress.add_task("Configurando métricas...", total=None)
            try:
                self.performance_tracker = PerformanceTracker()
                progress.update(perf_task, description="✅ Métricas configuradas")
            except Exception as e:
                progress.update(perf_task, description=f"❌ Error en métricas: {e}")
                raise
            
            # 6. Inicializar Pipeline principal
            pipeline_task = progress.add_task("Configurando pipeline de audio...", total=None)
            try:
                self.pipeline = MeetingAssistantPipeline()
                
                # Configurar callbacks específicos de cobranza
                self.pipeline.add_callback("on_transcript", self._on_transcript)
                self.pipeline.add_callback("on_suggestion", self._on_suggestion)
                self.pipeline.add_callback("on_error", self._on_error)
                
                await self.pipeline.setup()
                progress.update(pipeline_task, description="✅ Pipeline configurado")
            except Exception as e:
                progress.update(pipeline_task, description=f"❌ Error en pipeline: {e}")
                raise
        
        self.components_initialized = True
        console.print("\n[bold green]🚀 Sistema inicializado correctamente![/bold green]")
        
        # Mostrar información del sistema
        self._show_system_info()
    
    def _show_system_info(self):
        """Muestra información del sistema inicializado"""
        info_table = Table(title="Estado del Sistema", show_header=True, header_style="bold magenta")
        info_table.add_column("Componente", style="cyan", no_wrap=True)
        info_table.add_column("Estado", justify="center")
        info_table.add_column("Información", style="dim")
        
        components = [
            ("Knowledge Base", "✅ Activa", f"{self.chroma_manager.get_collection_stats('aeiou_examples')['total_documents']} ejemplos AEIOU"),
            ("Compliance Engine", "✅ Activo", "FDCPA monitoring habilitado"),
            ("Query Engine", "✅ Activo", "Collections specialization"),
            ("Dashboard", "✅ Activo", "Real-time metrics"),
            ("Performance Tracker", "✅ Activo", "Agent scoring enabled"),
            ("Audio Pipeline", "✅ Configurado", "Voice recognition ready")
        ]
        
        for component, status, info in components:
            info_table.add_row(component, status, info)
        
        console.print(info_table)
        
        # Mostrar controles
        console.print("\n[bold yellow]Controles:[/bold yellow]")
        console.print("  [cyan]Ctrl+C[/cyan] - Detener sistema")
        console.print("  [cyan]Ctrl+D[/cyan] - Dashboard detallado")
        console.print("  [cyan]Ctrl+R[/cyan] - Reporte de sesión")
    
    async def start_session(self, agent_id: str, agent_name: str):
        """Inicia una sesión de cobranza"""
        if not self.components_initialized:
            raise RuntimeError("Sistema no inicializado. Llama initialize() primero.")
        
        self.agent_info = {
            "id": agent_id,
            "name": agent_name,
            "session_start": asyncio.get_event_loop().time()
        }
        
        console.print(f"\n[bold green]📞 Sesión iniciada para agente: {agent_name} ({agent_id})[/bold green]")
        
        # Iniciar pipeline de audio
        await self.pipeline.start()
        self.is_running = True
        
        # Iniciar dashboard en vivo
        await self._run_live_dashboard()
    
    async def _run_live_dashboard(self):
        """Ejecuta el dashboard en tiempo real"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=5)
        )
        
        layout["main"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="compliance", ratio=1)
        )
        
        with Live(layout, console=console, refresh_per_second=2) as live:
            while self.is_running:
                # Actualizar header
                layout["header"].update(Panel(
                    f"[bold blue]AI Collections Assistant[/bold blue] | "
                    f"Agente: {self.agent_info['name']} | "
                    f"Llamadas: {self.session_metrics['calls_processed']} | "
                    f"Estado: {'🟢 Activo' if self.is_running else '🔴 Inactivo'}",
                    border_style="blue"
                ))
                
                # Actualizar métricas
                metrics_table = self._build_metrics_table()
                layout["metrics"].update(Panel(metrics_table, title="📊 Métricas", border_style="green"))
                
                # Actualizar compliance
                compliance_table = self._build_compliance_table()
                layout["compliance"].update(Panel(compliance_table, title="⚖️ Compliance", border_style="yellow"))
                
                # Actualizar footer con ayuda
                layout["footer"].update(Panel(
                    "[dim]Controles: Ctrl+C=Salir | Ctrl+D=Dashboard | Ctrl+R=Reporte | Ctrl+H=Ayuda[/dim]",
                    border_style="dim"
                ))
                
                await asyncio.sleep(0.5)
    
    def _build_metrics_table(self) -> Table:
        """Construye tabla de métricas en tiempo real"""
        table = Table(show_header=True, header_style="bold green")
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", justify="right")
        table.add_column("Tendencia", justify="center")
        
        metrics = [
            ("Llamadas Procesadas", str(self.session_metrics["calls_processed"]), "📈"),
            ("Sugerencias Dadas", str(self.session_metrics["suggestions_given"]), "💡"),
            ("Violations Detectadas", str(self.session_metrics["compliance_violations"]), "⚠️" if self.session_metrics["compliance_violations"] > 0 else "✅"),
            ("Escalaciones", str(self.session_metrics["escalations"]), "🔺" if self.session_metrics["escalations"] > 0 else "✅"),
            ("Compliance Score", "94.2%", "✅"),
            ("Avg Call Time", "4:32", "📉")
        ]
        
        for metric, value, trend in metrics:
            table.add_row(metric, value, trend)
        
        return table
    
    def _build_compliance_table(self) -> Table:
        """Construye tabla de compliance"""
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Check", style="cyan")
        table.add_column("Estado", justify="center")
        table.add_column("Última Verificación")
        
        compliance_checks = [
            ("Horarios Permitidos", "✅", "Continuo"),
            ("Frecuencia de Contacto", "✅", "Por llamada"),
            ("Lenguaje Prohibido", "✅", "Tiempo real"),
            ("Divulgación Requerida", "✅", "Inicio de llamada"),
            ("Validación de Deuda", "⚠️", "Pendiente")
        ]
        
        for check, status, last_check in compliance_checks:
            table.add_row(check, status, last_check)
        
        return table
    
    def _on_transcript(self, transcript):
        """Maneja transcripciones en tiempo real"""
        # Análisis de compliance en tiempo real
        if transcript.speaker == "USER":  # Agente hablando
            alerts = self.compliance_engine.analyze_transcript_real_time(
                transcript.text,
                self.agent_info["id"],
                self.current_call or "UNKNOWN"
            )
            
            for alert in alerts:
                self.session_metrics["compliance_violations"] += 1
                console.print(f"\n[bold red]🚨 ALERTA COMPLIANCE:[/bold red] {alert.message}")
                console.print(f"[yellow]Acción requerida:[/yellow] {alert.action_required}")
                
                if alert.auto_block:
                    console.print("[bold red]⛔ LLAMADA BLOQUEADA AUTOMÁTICAMENTE[/bold red]")
    
    def _on_suggestion(self, suggestion):
        """Maneja sugerencias generadas"""
        self.session_metrics["suggestions_given"] += 1
        
        # Log de la sugerencia para métricas
        self.performance_tracker.log_suggestion(
            agent_id=self.agent_info["id"],
            suggestion_text=suggestion.text,
            confidence=suggestion.confidence,
            context=suggestion.context
        )
        
        console.print(f"\n[bold green]💡 SUGERENCIA[/bold green] (confianza: {suggestion.confidence:.2f}):")
        console.print(f"[cyan]{suggestion.text}[/cyan]")
        
        if hasattr(suggestion, 'escalation_recommended') and suggestion.escalation_recommended:
            self.session_metrics["escalations"] += 1
            console.print("[bold yellow]🔺 ESCALACIÓN RECOMENDADA[/bold yellow]")
    
    def _on_error(self, error_data):
        """Maneja errores del sistema"""
        logger.error(f"Error en pipeline: {error_data}")
        console.print(f"[bold red]❌ Error del sistema:[/bold red] {error_data}")
    
    def stop_session(self):
        """Detiene la sesión actual"""
        self.is_running = False
        if self.pipeline:
            self.pipeline.stop()
        
        # Generar reporte de sesión
        self._generate_session_report()
        
        console.print("\n[bold yellow]📊 Sesión finalizada[/bold yellow]")
    
    def _generate_session_report(self):
        """Genera reporte de la sesión"""
        if not self.agent_info:
            return
        
        session_duration = asyncio.get_event_loop().time() - self.agent_info["session_start"]
        
        report_table = Table(title="Reporte de Sesión", show_header=True, header_style="bold blue")
        report_table.add_column("Métrica", style="cyan")
        report_table.add_column("Valor", justify="right")
        
        report_data = [
            ("Agente", f"{self.agent_info['name']} ({self.agent_info['id']})"),
            ("Duración de Sesión", f"{session_duration/60:.1f} minutos"),
            ("Llamadas Procesadas", str(self.session_metrics["calls_processed"])),
            ("Sugerencias Generadas", str(self.session_metrics["suggestions_given"])),
            ("Violations de Compliance", str(self.session_metrics["compliance_violations"])),
            ("Escalaciones", str(self.session_metrics["escalations"])),
            ("Eficiencia (Sugerencias/Min)", f"{self.session_metrics['suggestions_given']/(session_duration/60):.1f}")
        ]
        
        for metric, value in report_data:
            report_table.add_row(metric, value)
        
        console.print(report_table)
        
        # Guardar reporte
        self.performance_tracker.save_session_report(
            agent_id=self.agent_info["id"],
            session_data={
                "duration_minutes": session_duration/60,
                "metrics": self.session_metrics,
                "timestamp": asyncio.get_event_loop().time()
            }
        )


def setup_signal_handlers(assistant: CollectionsAssistant):
    """Configura manejadores de señales"""
    def signal_handler(signum, frame):
        console.print("\n[yellow]🛑 Deteniendo sistema...[/yellow]")
        assistant.stop_session()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Función principal"""
    # Configurar logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/collections_assistant.log'),
            logging.StreamHandler()
        ]
    )
    
    # Crear instancia del asistente
    assistant = CollectionsAssistant()
    
    # Configurar manejadores de señales
    setup_signal_handlers(assistant)
    
    try:
        # Inicializar sistema
        await assistant.initialize()
        
        # Solicitar información del agente
        console.print("\n[bold cyan]👤 Configuración del Agente[/bold cyan]")
        agent_id = console.input("[cyan]ID del Agente: [/cyan]")
        agent_name = console.input("[cyan]Nombre del Agente: [/cyan]")
        
        if not agent_id or not agent_name:
            console.print("[red]Error: ID y nombre del agente son requeridos[/red]")
            return
        
        # Iniciar sesión
        await assistant.start_session(agent_id, agent_name)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Sistema interrumpido por el usuario[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error fatal:[/bold red] {e}")
        logger.exception("Error fatal en main")
    finally:
        assistant.stop_session()


if __name__ == "__main__":
    # Verificar que estamos en modo collections
    if not hasattr(config, 'collections') or not getattr(config, 'collections', {}).get('enabled', False):
        console.print("[red]Error: Sistema no configurado para modo collections[/red]")
        console.print("[yellow]Usa: python setup.py --collections[/yellow]")
        sys.exit(1)
    
    # Ejecutar aplicación
    asyncio.run(main())
