"""
Dashboard especializado para call centers de cobranza

Proporciona métricas en tiempo real, análisis de performance
y reportes específicos para operaciones de cobranza.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.layout import Layout
from rich.live import Live

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class AgentMetrics:
    """Métricas de performance de un agente"""
    agent_id: str
    agent_name: str
    calls_today: int = 0
    promises_obtained: int = 0
    payments_scheduled: int = 0
    total_amount_promised: float = 0.0
    average_call_duration: float = 0.0
    compliance_score: float = 1.0
    escalations: int = 0
    right_party_contacts: int = 0
    suggestions_used: int = 0
    suggestions_ignored: int = 0
    recovery_rate: float = 0.0
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()


@dataclass
class TeamMetrics:
    """Métricas del equipo completo"""
    team_id: str
    team_name: str
    total_agents: int = 0
    active_agents: int = 0
    total_calls: int = 0
    total_promises: int = 0
    total_recovery: float = 0.0
    average_compliance: float = 1.0
    escalation_rate: float = 0.0
    target_recovery: float = 0.0
    recovery_percentage: float = 0.0
    top_performers: List[str] = None
    improvement_needed: List[str] = None
    
    def __post_init__(self):
        if self.top_performers is None:
            self.top_performers = []
        if self.improvement_needed is None:
            self.improvement_needed = []


class CollectionsDashboard:
    """Dashboard principal para call centers de cobranza"""
    
    def __init__(self, data_dir: str = "./data/dashboard"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Almacenamiento en memoria de métricas
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.team_metrics: Dict[str, TeamMetrics] = {}
        
        # Configuración de metas
        self.daily_targets = {
            "calls_per_agent": 50,
            "promises_per_agent": 12,
            "compliance_minimum": 0.90,
            "recovery_rate_target": 0.25
        }
        
        # Cargar datos existentes
        self._load_historical_data()
    
    def update_agent_metrics(self, agent_id: str, **kwargs):
        """Actualiza métricas de un agente específico"""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                agent_name=kwargs.get('agent_name', f'Agent_{agent_id}')
            )
        
        # Actualizar campos proporcionados
        for field, value in kwargs.items():
            if hasattr(self.agent_metrics[agent_id], field):
                setattr(self.agent_metrics[agent_id], field, value)
        
        self.agent_metrics[agent_id].last_update = datetime.now()
        
        # Calcular métricas derivadas
        self._calculate_derived_metrics(agent_id)
        
        # Persistir cambios
        self._save_agent_data(agent_id)
    
    def _calculate_derived_metrics(self, agent_id: str):
        """Calcula métricas derivadas para un agente"""
        metrics = self.agent_metrics[agent_id]
        
        # Recovery rate
        if metrics.calls_today > 0:
            metrics.recovery_rate = metrics.promises_obtained / metrics.calls_today
        
        # Suggestion usage rate
        total_suggestions = metrics.suggestions_used + metrics.suggestions_ignored
        if total_suggestions > 0:
            metrics.suggestion_usage_rate = metrics.suggestions_used / total_suggestions
    
    def get_agent_performance_summary(self, agent_id: str) -> Dict[str, Any]:
        """Obtiene resumen de performance de un agente"""
        if agent_id not in self.agent_metrics:
            return {"error": "Agent not found"}
        
        metrics = self.agent_metrics[agent_id]
        targets = self.daily_targets
        
        return {
            "agent_info": {
                "id": metrics.agent_id,
                "name": metrics.agent_name,
                "last_update": metrics.last_update.isoformat()
            },
            "performance": {
                "calls_today": {
                    "value": metrics.calls_today,
                    "target": targets["calls_per_agent"],
                    "percentage": (metrics.calls_today / targets["calls_per_agent"]) * 100
                },
                "promises_obtained": {
                    "value": metrics.promises_obtained,
                    "target": targets["promises_per_agent"],
                    "percentage": (metrics.promises_obtained / targets["promises_per_agent"]) * 100
                },
                "recovery_rate": {
                    "value": metrics.recovery_rate,
                    "target": targets["recovery_rate_target"],
                    "percentage": (metrics.recovery_rate / targets["recovery_rate_target"]) * 100
                },
                "compliance_score": {
                    "value": metrics.compliance_score,
                    "target": targets["compliance_minimum"],
                    "status": "good" if metrics.compliance_score >= targets["compliance_minimum"] else "needs_improvement"
                }
            },
            "financial": {
                "total_amount_promised": metrics.total_amount_promised,
                "average_promise_amount": metrics.total_amount_promised / max(metrics.promises_obtained, 1)
            },
            "efficiency": {
                "average_call_duration": metrics.average_call_duration,
                "suggestions_used": metrics.suggestions_used,
                "suggestions_ignored": metrics.suggestions_ignored,
                "right_party_contact_rate": metrics.right_party_contacts / max(metrics.calls_today, 1)
            }
        }
    
    def generate_real_time_dashboard(self) -> Layout:
        """Genera dashboard en tiempo real"""
        layout = Layout()
        
        # Dividir en secciones
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=2)
        )
        
        layout["main"].split_row(
            Layout(name="team_summary", ratio=1),
            Layout(name="top_performers", ratio=1),
            Layout(name="alerts", ratio=1)
        )
        
        # Header con información general
        header_info = self._build_header_panel()
        layout["header"].update(header_info)
        
        # Resumen del equipo
        team_summary = self._build_team_summary_table()
        layout["team_summary"].update(Panel(team_summary, title="📊 Resumen del Equipo", border_style="blue"))
        
        # Top performers
        top_performers = self._build_top_performers_table()
        layout["top_performers"].update(Panel(top_performers, title="🏆 Top Performers", border_style="green"))
        
        # Alertas y métricas críticas
        alerts = self._build_alerts_panel()
        layout["alerts"].update(Panel(alerts, title="🚨 Alertas", border_style="red"))
        
        # Footer con controles
        footer_info = Panel(
            "[dim]Actualización automática cada 30s | Ctrl+R para reporte completo | Ctrl+E para exportar[/dim]",
            border_style="dim"
        )
        layout["footer"].update(footer_info)
        
        return layout
    
    def _build_header_panel(self) -> Panel:
        """Construye panel de header con información general"""
        total_agents = len(self.agent_metrics)
        active_agents = sum(1 for m in self.agent_metrics.values() 
                          if (datetime.now() - m.last_update).seconds < 300)  # Activos en últimos 5 min
        
        total_calls = sum(m.calls_today for m in self.agent_metrics.values())
        total_promises = sum(m.promises_obtained for m in self.agent_metrics.values())
        total_recovery = sum(m.total_amount_promised for m in self.agent_metrics.values())
        
        header_text = (
            f"[bold blue]🎯 AI Collections Dashboard[/bold blue] | "
            f"Agentes: {active_agents}/{total_agents} | "
            f"Llamadas: {total_calls} | "
            f"Promesas: {total_promises} | "
            f"Recuperación: ${total_recovery:,.2f}"
        )
        
        return Panel(header_text, border_style="blue")
    
    def _build_team_summary_table(self) -> Table:
        """Construye tabla de resumen del equipo"""
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Métrica", style="cyan")
        table.add_column("Actual", justify="right")
        table.add_column("Meta", justify="right")
        table.add_column("%", justify="right")
        table.add_column("Estado", justify="center")
        
        # Calcular métricas del equipo
        total_calls = sum(m.calls_today for m in self.agent_metrics.values())
        total_promises = sum(m.promises_obtained for m in self.agent_metrics.values())
        avg_compliance = sum(m.compliance_score for m in self.agent_metrics.values()) / max(len(self.agent_metrics), 1)
        
        # Metas del equipo (asumiendo 10 agentes)
        team_size = max(len(self.agent_metrics), 1)
        
        metrics = [
            ("Llamadas Totales", total_calls, self.daily_targets["calls_per_agent"] * team_size),
            ("Promesas Obtenidas", total_promises, self.daily_targets["promises_per_agent"] * team_size),
            ("Compliance Promedio", f"{avg_compliance:.1%}", f"{self.daily_targets['compliance_minimum']:.1%}")
        ]
        
        for metric_name, actual, target in metrics:
            if isinstance(actual, str):  # Para porcentajes
                percentage = float(actual.strip('%')) / float(target.strip('%')) * 100
                percentage_str = f"{percentage:.1f}%"
            else:
                percentage = (actual / target) * 100 if target > 0 else 0
                percentage_str = f"{percentage:.1f}%"
            
            status = "✅" if percentage >= 100 else "🟡" if percentage >= 80 else "🔴"
            
            table.add_row(
                metric_name,
                str(actual),
                str(target),
                percentage_str,
                status
            )
        
        return table
    
    def _build_top_performers_table(self) -> Table:
        """Construye tabla de mejores performers"""
        table = Table(show_header=True, header_style="bold green")
        table.add_column("Agente", style="cyan")
        table.add_column("Promesas", justify="right")
        table.add_column("Recovery $", justify="right")
        table.add_column("Compliance", justify="right")
        
        # Ordenar agentes por recovery amount
        sorted_agents = sorted(
            self.agent_metrics.values(),
            key=lambda x: x.total_amount_promised,
            reverse=True
        )[:5]  # Top 5
        
        for agent in sorted_agents:
            table.add_row(
                agent.agent_name,
                str(agent.promises_obtained),
                f"${agent.total_amount_promised:,.0f}",
                f"{agent.compliance_score:.1%}"
            )
        
        return table
    
    def _build_alerts_panel(self) -> str:
        """Construye panel de alertas"""
        alerts = []
        
        # Verificar compliance de agentes
        for agent in self.agent_metrics.values():
            if agent.compliance_score < self.daily_targets["compliance_minimum"]:
                alerts.append(f"🚨 {agent.agent_name}: Compliance bajo ({agent.compliance_score:.1%})")
            
            if agent.escalations > 3:
                alerts.append(f"⚠️ {agent.agent_name}: Muchas escalaciones ({agent.escalations})")
        
        # Verificar métricas del equipo
        total_calls = sum(m.calls_today for m in self.agent_metrics.values())
        if total_calls < len(self.agent_metrics) * self.daily_targets["calls_per_agent"] * 0.5:
            alerts.append("📉 Equipo por debajo del 50% de meta de llamadas")
        
        if not alerts:
            alerts.append("✅ Sin alertas críticas")
        
        return "\n".join(alerts[:10])  # Máximo 10 alertas
    
    def generate_daily_report(self, date: datetime = None) -> Dict[str, Any]:
        """Genera reporte diario completo"""
        if date is None:
            date = datetime.now()
        
        # Filtrar métricas del día
        daily_metrics = {
            agent_id: metrics for agent_id, metrics in self.agent_metrics.items()
            if metrics.last_update.date() == date.date()
        }
        
        # Calcular totales
        total_calls = sum(m.calls_today for m in daily_metrics.values())
        total_promises = sum(m.promises_obtained for m in daily_metrics.values())
        total_recovery = sum(m.total_amount_promised for m in daily_metrics.values())
        avg_compliance = sum(m.compliance_score for m in daily_metrics.values()) / max(len(daily_metrics), 1)
        
        # Identificar top performers
        top_performers = sorted(
            daily_metrics.values(),
            key=lambda x: x.total_amount_promised,
            reverse=True
        )[:3]
        
        # Identificar agentes que necesitan mejora
        improvement_needed = [
            agent for agent in daily_metrics.values()
            if agent.compliance_score < self.daily_targets["compliance_minimum"] or
               agent.recovery_rate < self.daily_targets["recovery_rate_target"] * 0.7
        ]
        
        return {
            "date": date.isoformat(),
            "summary": {
                "total_agents": len(daily_metrics),
                "total_calls": total_calls,
                "total_promises": total_promises,
                "total_recovery": total_recovery,
                "average_compliance": avg_compliance,
                "team_recovery_rate": total_promises / max(total_calls, 1)
            },
            "performance": {
                "calls_target_achievement": total_calls / (len(daily_metrics) * self.daily_targets["calls_per_agent"]),
                "promises_target_achievement": total_promises / (len(daily_metrics) * self.daily_targets["promises_per_agent"]),
                "compliance_target_achievement": avg_compliance / self.daily_targets["compliance_minimum"]
            },
            "top_performers": [
                {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.agent_name,
                    "recovery_amount": agent.total_amount_promised,
                    "promises": agent.promises_obtained,
                    "compliance_score": agent.compliance_score
                }
                for agent in top_performers
            ],
            "improvement_needed": [
                {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.agent_name,
                    "issues": [
                        "Low compliance" if agent.compliance_score < self.daily_targets["compliance_minimum"] else None,
                        "Low recovery rate" if agent.recovery_rate < self.daily_targets["recovery_rate_target"] * 0.7 else None
                    ]
                }
                for agent in improvement_needed
            ]
        }
    
    def export_metrics(self, format_type: str = "json", output_path: str = None) -> str:
        """Exporta métricas en formato especificado"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.data_dir / f"metrics_export_{timestamp}.{format_type}"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "agent_metrics": {aid: asdict(metrics) for aid, metrics in self.agent_metrics.items()},
            "team_metrics": {tid: asdict(metrics) for tid, metrics in self.team_metrics.items()},
            "daily_targets": self.daily_targets
        }
        
        if format_type.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        elif format_type.lower() == "csv":
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    "Agent ID", "Agent Name", "Calls Today", "Promises Obtained",
                    "Total Recovery", "Compliance Score", "Recovery Rate", "Escalations"
                ])
                
                # Data
                for metrics in self.agent_metrics.values():
                    writer.writerow([
                        metrics.agent_id, metrics.agent_name, metrics.calls_today,
                        metrics.promises_obtained, metrics.total_amount_promised,
                        metrics.compliance_score, metrics.recovery_rate, metrics.escalations
                    ])
        
        return str(output_path)
    
    def _load_historical_data(self):
        """Carga datos históricos si existen"""
        try:
            history_file = self.data_dir / "metrics_history.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Cargar métricas de agentes
                for agent_id, agent_data in data.get("agent_metrics", {}).items():
                    # Convertir timestamp strings de vuelta a datetime
                    if "last_update" in agent_data:
                        agent_data["last_update"] = datetime.fromisoformat(agent_data["last_update"])
                    
                    self.agent_metrics[agent_id] = AgentMetrics(**agent_data)
                
                logger.info(f"Cargados datos históricos de {len(self.agent_metrics)} agentes")
        
        except Exception as e:
            logger.warning(f"No se pudieron cargar datos históricos: {e}")
    
    def _save_agent_data(self, agent_id: str):
        """Guarda datos de un agente específico"""
        try:
            history_file = self.data_dir / "metrics_history.json"
            
            # Cargar datos existentes
            data = {"agent_metrics": {}, "team_metrics": {}}
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Actualizar datos del agente
            data["agent_metrics"][agent_id] = asdict(self.agent_metrics[agent_id])
            
            # Guardar
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        except Exception as e:
            logger.error(f"Error guardando datos del agente {agent_id}: {e}")


if __name__ == "__main__":
    # Test del dashboard
    dashboard = CollectionsDashboard()
    
    # Simular algunos agentes
    test_agents = [
        {"agent_id": "AGT_001", "agent_name": "María González", "calls_today": 45, "promises_obtained": 12, "total_amount_promised": 3500.0, "compliance_score": 0.95},
        {"agent_id": "AGT_002", "agent_name": "Carlos Ruiz", "calls_today": 38, "promises_obtained": 8, "total_amount_promised": 2100.0, "compliance_score": 0.89},
        {"agent_id": "AGT_003", "agent_name": "Ana López", "calls_today": 52, "promises_obtained": 15, "total_amount_promised": 4200.0, "compliance_score": 0.97}
    ]
    
    # Actualizar métricas
    for agent_data in test_agents:
        dashboard.update_agent_metrics(**agent_data)
    
    # Mostrar dashboard
    layout = dashboard.generate_real_time_dashboard()
    console.print(layout)
    
    # Generar reporte
    report = dashboard.generate_daily_report()
    console.print("\n📊 Reporte Diario:")
    console.print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
