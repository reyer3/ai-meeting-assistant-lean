"""
Compliance Engine para Cobranza

Este módulo monitorea en tiempo real las conversaciones
para prevenir violaciones de FDCPA y otras regulaciones
de cobranza.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, time

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Tipos de violaciones de compliance"""
    HARASSMENT = "harassment"
    FALSE_REPRESENTATION = "false_representation"
    UNFAIR_PRACTICES = "unfair_practices"
    IMPROPER_CONTACT = "improper_contact"
    UNAUTHORIZED_DISCLOSURE = "unauthorized_disclosure"
    TIME_VIOLATION = "time_violation"
    FREQUENCY_VIOLATION = "frequency_violation"


class SeverityLevel(Enum):
    """Niveles de severidad de violaciones"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceViolation:
    """Violación de compliance detectada"""
    violation_type: ViolationType
    severity: SeverityLevel
    description: str
    detected_phrase: str
    suggested_replacement: str
    timestamp: datetime
    agent_id: Optional[str] = None
    call_id: Optional[str] = None
    regulation_reference: Optional[str] = None


@dataclass
class ComplianceAlert:
    """Alerta de compliance en tiempo real"""
    message: str
    severity: SeverityLevel
    action_required: str
    auto_block: bool = False


class ComplianceEngine:
    """Motor de compliance para prevenir violaciones en tiempo real"""
    
    def __init__(self):
        # Frases prohibidas por categoría
        self.prohibited_phrases = {
            ViolationType.HARASSMENT: [
                r"\bva a tener problemas\b",
                r"\bte vamos a demandar\b",
                r"\bte vamos a embargar\b",
                r"\barruinar tu crédito\b",
                r"\bno me hagas perder el tiempo\b",
                r"\bestás mintiendo\b",
                r"\bno me importa\b",
                r"\bsiempre tienes excusas\b"
            ],
            ViolationType.FALSE_REPRESENTATION: [
                r"\bsoy abogado\b",
                r"\bsoy del juzgado\b",
                r"\borden judicial\b",
                r"\bpolicía va a ir\b",
                r"\bte van a arrestar\b",
                r"\btiene 24 horas\b",
                r"\búltima oportunidad\b",
                r"\bgobierno nos autorizó\b"
            ],
            ViolationType.UNFAIR_PRACTICES: [
                r"\bintereses adicionales\b",
                r"\bmultas extras\b",
                r"\bcostos de cobranza falsos\b",
                r"\bva a costar más\b",
                r"\btarifa de procesamiento\b"
            ]
        }
        
        # Frases recomendadas como reemplazo
        self.compliance_alternatives = {
            "va a tener problemas": "esto podría afectar su historial crediticio",
            "te vamos a demandar": "podríamos considerar opciones legales",
            "te vamos a embargar": "podrían aplicarse remedios legales",
            "arruinar tu crédito": "impactar negativamente su historial crediticio",
            "estás mintiendo": "necesito verificar esa información",
            "no me importa": "entiendo su situación",
            "soy abogado": "represento a la compañía",
            "orden judicial": "proceso legal autorizado",
            "última oportunidad": "esta es una oportunidad importante"
        }
        
        # Horarios permitidos para llamadas (8 AM - 9 PM)
        self.allowed_call_hours = {
            "start": time(8, 0),
            "end": time(21, 0)
        }
        
        # Contadores de frecuencia por deudor
        self.contact_frequency = {}
        
        # Máximo 7 contactos en 7 días
        self.max_contacts_per_week = 7
    
    def analyze_transcript_real_time(self, text: str, agent_id: str, call_id: str) -> List[ComplianceAlert]:
        """Analiza transcripción en tiempo real para detectar violaciones"""
        alerts = []
        text_lower = text.lower()
        
        # Detectar frases prohibidas
        for violation_type, patterns in self.prohibited_phrases.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Encontrar la frase específica
                    match = re.search(pattern, text_lower)
                    detected_phrase = match.group() if match else pattern
                    
                    # Buscar alternativa
                    suggested_replacement = self._find_compliance_alternative(detected_phrase)
                    
                    # Determinar severidad
                    severity = self._determine_severity(violation_type)
                    
                    # Crear alerta
                    alert = ComplianceAlert(
                        message=f"ALERTA: Posible violación {violation_type.value} detectada",
                        severity=severity,
                        action_required=f"Reemplazar: '{detected_phrase}' por '{suggested_replacement}'",
                        auto_block=severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
                    )
                    
                    alerts.append(alert)
                    
                    # Log para auditoría
                    self._log_violation(
                        violation_type=violation_type,
                        severity=severity,
                        detected_phrase=detected_phrase,
                        suggested_replacement=suggested_replacement,
                        agent_id=agent_id,
                        call_id=call_id
                    )
        
        return alerts
    
    def check_call_time_compliance(self, debtor_timezone: str = "US/Eastern") -> Optional[ComplianceAlert]:
        """Verifica si la llamada está dentro de horarios permitidos"""
        current_time = datetime.now().time()
        
        if not (self.allowed_call_hours["start"] <= current_time <= self.allowed_call_hours["end"]):
            return ComplianceAlert(
                message="VIOLACIÓN DE HORARIO: Llamada fuera de horario permitido",
                severity=SeverityLevel.HIGH,
                action_required="Terminar llamada inmediatamente y reagendar",
                auto_block=True
            )
        
        return None
    
    def check_contact_frequency(self, debtor_id: str) -> Optional[ComplianceAlert]:
        """Verifica frecuencia de contacto con el deudor"""
        if debtor_id not in self.contact_frequency:
            self.contact_frequency[debtor_id] = []
        
        # Limpiar contactos antiguos (más de 7 días)
        now = datetime.now()
        week_ago = now.replace(day=now.day-7) if now.day > 7 else now.replace(month=now.month-1, day=30)
        
        self.contact_frequency[debtor_id] = [
            contact for contact in self.contact_frequency[debtor_id]
            if contact > week_ago
        ]
        
        # Verificar si excede el límite
        if len(self.contact_frequency[debtor_id]) >= self.max_contacts_per_week:
            return ComplianceAlert(
                message="VIOLACIÓN DE FRECUENCIA: Exceso de contactos en 7 días",
                severity=SeverityLevel.HIGH,
                action_required="No contactar por 7 días adicionales",
                auto_block=True
            )
        
        # Registrar este contacto
        self.contact_frequency[debtor_id].append(now)
        return None
    
    def validate_debt_validation_request(self, text: str) -> Optional[ComplianceAlert]:
        """Detecta solicitudes de validación de deuda"""
        validation_keywords = [
            "validación", "verificación", "prueba de la deuda",
            "documentación", "por escrito", "no reconozco",
            "disputo esta deuda", "no es mía"
        ]
        
        text_lower = text.lower()
        
        for keyword in validation_keywords:
            if keyword in text_lower:
                return ComplianceAlert(
                    message="SOLICITUD DE VALIDACIÓN DETECTADA",
                    severity=SeverityLevel.CRITICAL,
                    action_required="CESAR COBRANZA INMEDIATAMENTE. Transferir a legal para envío de validación",
                    auto_block=True
                )
        
        return None
    
    def generate_compliance_suggestion(self, detected_issue: str) -> str:
        """Genera sugerencia alternativa compliant"""
        suggestions = {
            "amenaza": "En lugar de amenazar, explique las consecuencias naturales del no pago",
            "presión": "Use técnicas de persuasión suave, no presión agresiva",
            "mentira": "Proporcione solo información precisa y verificable",
            "acoso": "Mantenga un tono profesional y respetuoso en todo momento",
            "horario": "Verifique zona horaria del deudor antes de llamar",
            "frecuencia": "Revise historial de contactos antes de nueva llamada"
        }
        
        for issue_type, suggestion in suggestions.items():
            if issue_type in detected_issue.lower():
                return suggestion
        
        return "Consulte manual de compliance para orientación específica"
    
    def _find_compliance_alternative(self, detected_phrase: str) -> str:
        """Encuentra alternativa compliant para frase problemática"""
        for prohibited, alternative in self.compliance_alternatives.items():
            if prohibited.lower() in detected_phrase.lower():
                return alternative
        
        return "[Usar lenguaje profesional y respetuoso]"
    
    def _determine_severity(self, violation_type: ViolationType) -> SeverityLevel:
        """Determina la severidad basada en el tipo de violación"""
        severity_mapping = {
            ViolationType.HARASSMENT: SeverityLevel.HIGH,
            ViolationType.FALSE_REPRESENTATION: SeverityLevel.CRITICAL,
            ViolationType.UNFAIR_PRACTICES: SeverityLevel.HIGH,
            ViolationType.IMPROPER_CONTACT: SeverityLevel.MEDIUM,
            ViolationType.UNAUTHORIZED_DISCLOSURE: SeverityLevel.HIGH,
            ViolationType.TIME_VIOLATION: SeverityLevel.HIGH,
            ViolationType.FREQUENCY_VIOLATION: SeverityLevel.HIGH
        }
        
        return severity_mapping.get(violation_type, SeverityLevel.MEDIUM)
    
    def _log_violation(self, **kwargs):
        """Registra violación para auditoría"""
        violation_log = {
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        # En implementación real, esto iría a base de datos de auditoría
        logger.warning(f"Compliance violation detected: {violation_log}")
    
    def get_compliance_score(self, call_transcript: str, agent_id: str) -> Dict[str, Any]:
        """Calcula score de compliance para la llamada completa"""
        total_violations = 0
        violation_details = []
        
        # Analizar toda la transcripción
        for violation_type, patterns in self.prohibited_phrases.items():
            for pattern in patterns:
                matches = re.findall(pattern, call_transcript.lower())
                if matches:
                    total_violations += len(matches)
                    violation_details.append({
                        "type": violation_type.value,
                        "count": len(matches),
                        "severity": self._determine_severity(violation_type).value
                    })
        
        # Calcular score (100 = perfecto, 0 = muchas violaciones)
        base_score = 100
        penalty_per_violation = 10
        
        compliance_score = max(0, base_score - (total_violations * penalty_per_violation))
        
        return {
            "compliance_score": compliance_score,
            "total_violations": total_violations,
            "violation_details": violation_details,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "pass_threshold": compliance_score >= 80
        }


class ComplianceReporter:
    """Generador de reportes de compliance"""
    
    def __init__(self, compliance_engine: ComplianceEngine):
        self.engine = compliance_engine
    
    def generate_daily_report(self, date: datetime) -> Dict[str, Any]:
        """Genera reporte diario de compliance"""
        # En implementación real, consultaría base de datos
        return {
            "date": date.isoformat(),
            "total_calls": 150,
            "compliant_calls": 142,
            "violation_calls": 8,
            "compliance_rate": 94.7,
            "most_common_violations": [
                {"type": "improper_language", "count": 5},
                {"type": "time_pressure", "count": 3}
            ],
            "agents_needing_training": [
                {"agent_id": "AGT_123", "violation_count": 3},
                {"agent_id": "AGT_456", "violation_count": 2}
            ]
        }
    
    def generate_agent_scorecard(self, agent_id: str, period_days: int = 30) -> Dict[str, Any]:
        """Genera scorecard de compliance para agente específico"""
        return {
            "agent_id": agent_id,
            "period_days": period_days,
            "total_calls": 95,
            "average_compliance_score": 87.3,
            "violation_trend": "improving",
            "training_recommendations": [
                "FDCPA harassment prevention",
                "Professional language techniques"
            ],
            "strengths": [
                "Excellent time management",
                "Good rapport building"
            ]
        }


if __name__ == "__main__":
    # Test del compliance engine
    engine = ComplianceEngine()
    
    # Test de frases problemáticas
    test_phrases = [
        "Te vamos a demandar si no pagas",
        "Soy abogado y tienes 24 horas",
        "Entiendo tu situación, ¿podríamos encontrar una solución?",
        "Estás mintiendo sobre tu situación económica"
    ]
    
    print("=== Test de Compliance Engine ===")
    
    for phrase in test_phrases:
        alerts = engine.analyze_transcript_real_time(phrase, "AGT_001", "CALL_001")
        
        print(f"\nFrase: '{phrase}'")
        if alerts:
            for alert in alerts:
                print(f"  ⚠️  {alert.severity.value.upper()}: {alert.message}")
                print(f"     Acción: {alert.action_required}")
        else:
            print("  ✅ Compliant")
    
    # Test de compliance score
    sample_transcript = "Buenos días, soy María de ABC Collections. Te llamo sobre tu cuenta. Entiendo que puede ser difícil, pero necesitamos encontrar una solución. ¿Podríamos hablar sobre un plan de pagos?"
    
    score = engine.get_compliance_score(sample_transcript, "AGT_001")
    print(f"\n=== Compliance Score ===")
    print(f"Score: {score['compliance_score']}/100")
    print(f"Violaciones: {score['total_violations']}")
    print(f"Pasa umbral: {'Sí' if score['pass_threshold'] else 'No'}")
