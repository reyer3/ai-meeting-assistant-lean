"""
Performance Tracker para Cobranza

Rastrear y analizar métricas de performance de agentes,
efectividad de sugerencias y tendencias del sistema.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SuggestionLog:
    """Log de una sugerencia generada"""
    suggestion_id: str
    agent_id: str
    timestamp: datetime
    suggestion_text: str
    confidence_score: float
    objection_type: str
    debtor_profile: str
    was_used: Optional[bool] = None
    effectiveness_rating: Optional[int] = None  # 1-5 scale
    call_outcome: Optional[str] = None
    context_data: Optional[Dict] = None


@dataclass
class CallMetrics:
    """Métricas de una llamada específica"""
    call_id: str
    agent_id: str
    debtor_id: str
    timestamp: datetime
    duration_seconds: int
    outcome: str  # "promise", "payment", "no_contact", "refusal", "escalation"
    amount_promised: float = 0.0
    payment_date: Optional[datetime] = None
    suggestions_shown: int = 0
    suggestions_used: int = 0
    compliance_violations: int = 0
    escalation_reason: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class AgentPerformanceReport:
    """Reporte de performance de un agente"""
    agent_id: str
    period_start: datetime
    period_end: datetime
    total_calls: int
    total_contact_time: float
    promises_obtained: int
    total_amount_promised: float
    average_call_duration: float
    right_party_contact_rate: float
    promise_to_pay_rate: float
    compliance_score: float
    suggestion_usage_rate: float
    suggestion_effectiveness: float
    improvement_trends: Dict[str, float]
    recommendations: List[str]


class PerformanceTracker:
    """Rastreador de performance para call centers de cobranza"""
    
    def __init__(self, db_path: str = "./data/performance.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Inicializar base de datos
        self._init_database()
        
        # Cache en memoria para métricas recientes
        self.recent_suggestions: List[SuggestionLog] = []
        self.recent_calls: List[CallMetrics] = []
        
        # Configuración de análisis
        self.analysis_config = {
            "min_calls_for_analysis": 10,
            "effectiveness_threshold": 3.5,  # Rating mínimo para considerar efectiva
            "compliance_threshold": 0.90,
            "trend_analysis_days": 30
        }
    
    def _init_database(self):
        """Inicializa las tablas de la base de datos"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla de sugerencias
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS suggestions (
                    suggestion_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    suggestion_text TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    objection_type TEXT NOT NULL,
                    debtor_profile TEXT NOT NULL,
                    was_used INTEGER,
                    effectiveness_rating INTEGER,
                    call_outcome TEXT,
                    context_data TEXT
                )
            """)
            
            # Tabla de llamadas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    debtor_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    duration_seconds INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    amount_promised REAL DEFAULT 0.0,
                    payment_date TEXT,
                    suggestions_shown INTEGER DEFAULT 0,
                    suggestions_used INTEGER DEFAULT 0,
                    compliance_violations INTEGER DEFAULT 0,
                    escalation_reason TEXT,
                    notes TEXT
                )
            """)
            
            # Tabla de agentes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    hire_date TEXT,
                    team_id TEXT,
                    specialization TEXT,
                    target_calls_per_day INTEGER DEFAULT 50,
                    target_recovery_rate REAL DEFAULT 0.25
                )
            """)
            
            # Índices para optimización
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_agent_timestamp ON suggestions(agent_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_calls_agent_timestamp ON calls(agent_id, timestamp)")
            
            conn.commit()
    
    def log_suggestion(self, agent_id: str, suggestion_text: str, confidence: float, context: Dict[str, Any]) -> str:
        """Registra una sugerencia generada"""
        suggestion_id = f"SUG_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        suggestion_log = SuggestionLog(
            suggestion_id=suggestion_id,
            agent_id=agent_id,
            timestamp=datetime.now(),
            suggestion_text=suggestion_text,
            confidence_score=confidence,
            objection_type=context.get('objection_type', 'unknown'),
            debtor_profile=context.get('debtor_profile', 'unknown'),
            context_data=context
        )
        
        # Guardar en base de datos
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO suggestions 
                (suggestion_id, agent_id, timestamp, suggestion_text, confidence_score, 
                 objection_type, debtor_profile, context_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                suggestion_log.suggestion_id, suggestion_log.agent_id,
                suggestion_log.timestamp.isoformat(), suggestion_log.suggestion_text,
                suggestion_log.confidence_score, suggestion_log.objection_type,
                suggestion_log.debtor_profile, json.dumps(suggestion_log.context_data)
            ))
            conn.commit()
        
        # Agregar al cache
        self.recent_suggestions.append(suggestion_log)
        
        # Mantener cache limitado
        if len(self.recent_suggestions) > 1000:
            self.recent_suggestions = self.recent_suggestions[-500:]
        
        logger.info(f"Suggestion logged: {suggestion_id} for agent {agent_id}")
        return suggestion_id
    
    def log_suggestion_feedback(self, suggestion_id: str, was_used: bool, effectiveness_rating: int = None, call_outcome: str = None):
        """Registra feedback sobre una sugerencia"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE suggestions 
                SET was_used = ?, effectiveness_rating = ?, call_outcome = ?
                WHERE suggestion_id = ?
            """, (1 if was_used else 0, effectiveness_rating, call_outcome, suggestion_id))
            conn.commit()
        
        # Actualizar cache si existe
        for suggestion in self.recent_suggestions:
            if suggestion.suggestion_id == suggestion_id:
                suggestion.was_used = was_used
                suggestion.effectiveness_rating = effectiveness_rating
                suggestion.call_outcome = call_outcome
                break
        
        logger.info(f"Suggestion feedback updated: {suggestion_id}")
    
    def log_call_metrics(self, call_data: Dict[str, Any]) -> str:
        """Registra métricas de una llamada"""
        call_id = call_data.get('call_id') or f"CALL_{call_data['agent_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        call_metrics = CallMetrics(
            call_id=call_id,
            agent_id=call_data['agent_id'],
            debtor_id=call_data['debtor_id'],
            timestamp=datetime.now(),
            duration_seconds=call_data['duration_seconds'],
            outcome=call_data['outcome'],
            amount_promised=call_data.get('amount_promised', 0.0),
            payment_date=call_data.get('payment_date'),
            suggestions_shown=call_data.get('suggestions_shown', 0),
            suggestions_used=call_data.get('suggestions_used', 0),
            compliance_violations=call_data.get('compliance_violations', 0),
            escalation_reason=call_data.get('escalation_reason'),
            notes=call_data.get('notes')
        )
        
        # Guardar en base de datos
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO calls 
                (call_id, agent_id, debtor_id, timestamp, duration_seconds, outcome,
                 amount_promised, payment_date, suggestions_shown, suggestions_used,
                 compliance_violations, escalation_reason, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                call_metrics.call_id, call_metrics.agent_id, call_metrics.debtor_id,
                call_metrics.timestamp.isoformat(), call_metrics.duration_seconds,
                call_metrics.outcome, call_metrics.amount_promised,
                call_metrics.payment_date.isoformat() if call_metrics.payment_date else None,
                call_metrics.suggestions_shown, call_metrics.suggestions_used,
                call_metrics.compliance_violations, call_metrics.escalation_reason,
                call_metrics.notes
            ))
            conn.commit()
        
        # Agregar al cache
        self.recent_calls.append(call_metrics)
        
        # Mantener cache limitado
        if len(self.recent_calls) > 1000:
            self.recent_calls = self.recent_calls[-500:]
        
        logger.info(f"Call metrics logged: {call_id} for agent {call_metrics.agent_id}")
        return call_id
    
    def analyze_suggestion_effectiveness(self, agent_id: str = None, days: int = 30) -> Dict[str, Any]:
        """Analiza la efectividad de las sugerencias"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Query base
            query = """
                SELECT objection_type, debtor_profile, confidence_score, was_used, 
                       effectiveness_rating, call_outcome
                FROM suggestions 
                WHERE timestamp >= ? AND effectiveness_rating IS NOT NULL
            """
            params = [(datetime.now() - timedelta(days=days)).isoformat()]
            
            # Filtrar por agente si se especifica
            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
        
        if not results:
            return {"error": "No hay suficientes datos para análisis"}
        
        # Convertir a estructura analizable
        data = []
        for row in results:
            objection_type, debtor_profile, confidence, was_used, rating, outcome = row
            data.append({
                'objection_type': objection_type,
                'debtor_profile': debtor_profile,
                'confidence_score': confidence,
                'was_used': bool(was_used) if was_used is not None else False,
                'effectiveness_rating': rating,
                'call_outcome': outcome
            })
        
        # Análisis estadístico
        usage_rate = sum(1 for d in data if d['was_used']) / len(data)
        avg_effectiveness = np.mean([d['effectiveness_rating'] for d in data])
        
        # Análisis por tipo de objeción
        objection_analysis = {}
        for objection_type in set(d['objection_type'] for d in data):
            objection_data = [d for d in data if d['objection_type'] == objection_type]
            objection_analysis[objection_type] = {
                'count': len(objection_data),
                'usage_rate': sum(1 for d in objection_data if d['was_used']) / len(objection_data),
                'avg_effectiveness': np.mean([d['effectiveness_rating'] for d in objection_data]),
                'success_rate': sum(1 for d in objection_data if d['call_outcome'] in ['promise', 'payment']) / len(objection_data)
            }
        
        # Análisis por perfil de deudor
        profile_analysis = {}
        for profile in set(d['debtor_profile'] for d in data):
            profile_data = [d for d in data if d['debtor_profile'] == profile]
            profile_analysis[profile] = {
                'count': len(profile_data),
                'usage_rate': sum(1 for d in profile_data if d['was_used']) / len(profile_data),
                'avg_effectiveness': np.mean([d['effectiveness_rating'] for d in profile_data])
            }
        
        return {
            'period_days': days,
            'total_suggestions': len(data),
            'overall_usage_rate': usage_rate,
            'overall_effectiveness': avg_effectiveness,
            'by_objection_type': objection_analysis,
            'by_debtor_profile': profile_analysis,
            'recommendations': self._generate_effectiveness_recommendations(objection_analysis, profile_analysis)
        }
    
    def generate_agent_performance_report(self, agent_id: str, days: int = 30) -> AgentPerformanceReport:
        """Genera reporte completo de performance de un agente"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Métricas de llamadas
            cursor.execute("""
                SELECT COUNT(*), SUM(duration_seconds), SUM(amount_promised),
                       AVG(duration_seconds), 
                       SUM(CASE WHEN outcome IN ('promise', 'payment') THEN 1 ELSE 0 END),
                       SUM(suggestions_shown), SUM(suggestions_used),
                       SUM(compliance_violations)
                FROM calls 
                WHERE agent_id = ? AND timestamp BETWEEN ? AND ?
            """, (agent_id, start_date.isoformat(), end_date.isoformat()))
            
            call_stats = cursor.fetchone()
            
            if not call_stats or call_stats[0] == 0:
                raise ValueError(f"No hay datos suficientes para el agente {agent_id}")
            
            total_calls, total_time, total_promised, avg_duration, promises, suggestions_shown, suggestions_used, violations = call_stats
            
            # Métricas de sugerencias
            cursor.execute("""
                SELECT COUNT(*), AVG(effectiveness_rating)
                FROM suggestions 
                WHERE agent_id = ? AND timestamp BETWEEN ? AND ? AND effectiveness_rating IS NOT NULL
            """, (agent_id, start_date.isoformat(), end_date.isoformat()))
            
            suggestion_stats = cursor.fetchone()
            suggestion_count, avg_effectiveness = suggestion_stats if suggestion_stats[0] else (0, 0)
        
        # Calcular métricas derivadas
        right_party_rate = 0.8  # Placeholder - necesitaría más datos
        promise_rate = promises / total_calls if total_calls > 0 else 0
        compliance_score = max(0, 1 - (violations / max(total_calls, 1)) * 0.1)
        suggestion_usage_rate = suggestions_used / max(suggestions_shown, 1)
        
        # Análisis de tendencias
        trends = self._analyze_agent_trends(agent_id, days)
        
        # Generar recomendaciones
        recommendations = self._generate_agent_recommendations(
            promise_rate, compliance_score, suggestion_usage_rate, avg_effectiveness or 0
        )
        
        return AgentPerformanceReport(
            agent_id=agent_id,
            period_start=start_date,
            period_end=end_date,
            total_calls=total_calls,
            total_contact_time=total_time / 3600,  # Convertir a horas
            promises_obtained=promises,
            total_amount_promised=total_promised or 0,
            average_call_duration=avg_duration or 0,
            right_party_contact_rate=right_party_rate,
            promise_to_pay_rate=promise_rate,
            compliance_score=compliance_score,
            suggestion_usage_rate=suggestion_usage_rate,
            suggestion_effectiveness=avg_effectiveness or 0,
            improvement_trends=trends,
            recommendations=recommendations
        )
    
    def _analyze_agent_trends(self, agent_id: str, days: int) -> Dict[str, float]:
        """Analiza tendencias de mejora/deterioro del agente"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Obtener datos por semana
            cursor.execute("""
                SELECT DATE(timestamp) as call_date,
                       COUNT(*) as calls,
                       SUM(CASE WHEN outcome IN ('promise', 'payment') THEN 1 ELSE 0 END) as promises,
                       SUM(compliance_violations) as violations
                FROM calls 
                WHERE agent_id = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY call_date
            """, (agent_id, (datetime.now() - timedelta(days=days)).isoformat()))
            
            daily_data = cursor.fetchall()
        
        if len(daily_data) < 7:  # Necesitamos al menos una semana de datos
            return {"insufficient_data": True}
        
        # Calcular tendencias usando regresión lineal
        days_numeric = list(range(len(daily_data)))
        
        calls_per_day = [row[1] for row in daily_data]
        promise_rates = [row[2] / max(row[1], 1) for row in daily_data]
        violation_rates = [row[3] / max(row[1], 1) for row in daily_data]
        
        # Regresión lineal para tendencias
        calls_trend = stats.linregress(days_numeric, calls_per_day).slope
        promise_trend = stats.linregress(days_numeric, promise_rates).slope
        compliance_trend = -stats.linregress(days_numeric, violation_rates).slope  # Negativo porque menos violaciones = mejor
        
        return {
            'calls_trend': calls_trend,
            'promise_rate_trend': promise_trend,
            'compliance_trend': compliance_trend,
            'overall_trend': (calls_trend + promise_trend + compliance_trend) / 3
        }
    
    def _generate_effectiveness_recommendations(self, objection_analysis: Dict, profile_analysis: Dict) -> List[str]:
        """Genera recomendaciones basadas en análisis de efectividad"""
        recommendations = []
        
        # Recomendar enfoques para objeciones con baja efectividad
        for objection_type, stats in objection_analysis.items():
            if stats['avg_effectiveness'] < self.analysis_config['effectiveness_threshold']:
                recommendations.append(
                    f"Mejorar técnicas para objeción '{objection_type}' - efectividad actual: {stats['avg_effectiveness']:.1f}/5"
                )
            
            if stats['usage_rate'] < 0.5:
                recommendations.append(
                    f"Incrementar uso de sugerencias para '{objection_type}' - tasa actual: {stats['usage_rate']:.1%}"
                )
        
        # Recomendar entrenamientos específicos
        if not recommendations:
            recommendations.append("Excelente performance - continuar con estrategias actuales")
        
        return recommendations
    
    def _generate_agent_recommendations(self, promise_rate: float, compliance_score: float, 
                                      suggestion_usage: float, effectiveness: float) -> List[str]:
        """Genera recomendaciones específicas para un agente"""
        recommendations = []
        
        if promise_rate < 0.20:
            recommendations.append("Enfocar en técnicas de cierre y obtención de compromisos")
        
        if compliance_score < self.analysis_config['compliance_threshold']:
            recommendations.append("Revisión urgente de compliance - programar entrenamiento")
        
        if suggestion_usage < 0.6:
            recommendations.append("Incrementar uso de sugerencias del sistema de IA")
        
        if effectiveness < self.analysis_config['effectiveness_threshold']:
            recommendations.append("Mejorar interpretación y aplicación de sugerencias")
        
        if promise_rate > 0.25 and compliance_score > 0.95:
            recommendations.append("Excelente performer - considerar para mentoría de otros agentes")
        
        return recommendations if recommendations else ["Continuar con estrategias actuales"]
    
    def save_session_report(self, agent_id: str, session_data: Dict[str, Any]):
        """Guarda reporte de sesión"""
        report_file = Path(f"./data/session_reports/session_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Session report saved: {report_file}")


if __name__ == "__main__":
    # Test del performance tracker
    tracker = PerformanceTracker()
    
    # Simular algunos datos
    agent_id = "AGT_001"
    
    # Log de sugerencia
    suggestion_id = tracker.log_suggestion(
        agent_id=agent_id,
        suggestion_text="Entiendo tu situación...",
        confidence=0.85,
        context={
            'objection_type': 'cannot_pay',
            'debtor_profile': 'financially_stressed'
        }
    )
    
    # Feedback de sugerencia
    tracker.log_suggestion_feedback(
        suggestion_id=suggestion_id,
        was_used=True,
        effectiveness_rating=4,
        call_outcome='promise'
    )
    
    # Log de llamada
    call_id = tracker.log_call_metrics({
        'agent_id': agent_id,
        'debtor_id': 'DBT_001',
        'duration_seconds': 320,
        'outcome': 'promise',
        'amount_promised': 150.0,
        'suggestions_shown': 2,
        'suggestions_used': 1,
        'compliance_violations': 0
    })
    
    print(f"Suggestion logged: {suggestion_id}")
    print(f"Call logged: {call_id}")
    
    # Análisis de efectividad
    effectiveness = tracker.analyze_suggestion_effectiveness(agent_id=agent_id)
    print(f"\nEfectividad: {json.dumps(effectiveness, indent=2, default=str)}")
