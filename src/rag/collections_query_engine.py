"""
Motor de consultas especializado para Cobranza

Extiende el query engine base con lógica específica para
detectar objeciones, evaluar situaciones de cobranza y
generar sugerencias contextuales optimizadas.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from query_engine import QueryEngine, ConversationContext, RAGResponse
from chroma_manager import ChromaManager
from collections_knowledge_base import get_collections_objections_responses

logger = logging.getLogger(__name__)


class ObjectionType(Enum):
    """Tipos de objeciones comunes en cobranza"""
    ALREADY_PAID = "already_paid"
    CANNOT_PAY = "cannot_pay"
    NOT_MY_DEBT = "not_my_debt"
    CALL_LATER = "call_later"
    DISPUTE_AMOUNT = "dispute_amount"
    FINANCIAL_HARDSHIP = "financial_hardship"
    BROKEN_PROMISES = "broken_promises"
    PAYMENT_PLAN_REQUEST = "payment_plan_request"
    HOSTILE_RESPONSE = "hostile_response"
    LEGAL_THREAT = "legal_threat"
    UNKNOWN = "unknown"


class DebtorProfile(Enum):
    """Perfiles de deudor detectables"""
    COOPERATIVE = "cooperative"  # Dispuesto a colaborar
    EVASIVE = "evasive"  # Evita comprometerse
    HOSTILE = "hostile"  # Agresivo o confrontativo
    FINANCIALLY_STRESSED = "financially_stressed"  # Genuina dificultad económica
    REPEAT_BREAKER = "repeat_breaker"  # Historia de promesas rotas
    DISPUTER = "disputer"  # Disputa la validez de la deuda


@dataclass
class CollectionsContext(ConversationContext):
    """Contexto específico para cobranza"""
    debtor_id: Optional[str] = None
    account_balance: Optional[float] = None
    days_past_due: Optional[int] = None
    previous_promises: Optional[int] = None
    payment_history: Optional[List[Dict]] = None
    objection_detected: Optional[ObjectionType] = None
    debtor_profile: Optional[DebtorProfile] = None
    compliance_risk: Optional[str] = None


@dataclass
class CollectionsRAGResponse(RAGResponse):
    """Respuesta específica para cobranza"""
    objection_type: ObjectionType
    debtor_profile: DebtorProfile
    compliance_score: float
    escalation_recommended: bool
    payment_plan_suggestion: Optional[Dict[str, Any]] = None


class CollectionsQueryEngine(QueryEngine):
    """Motor de consultas especializado para cobranza"""
    
    def __init__(self, chroma_manager: ChromaManager):
        super().__init__(chroma_manager)
        
        # Patrones de objeciones específicas
        self.objection_patterns = {
            ObjectionType.ALREADY_PAID: [
                r"ya paqué", r"ya pagué", r"ya está pagado", r"ya lo pagué",
                r"se liquidó", r"está saldado", r"fue cancelado"
            ],
            ObjectionType.CANNOT_PAY: [
                r"no tengo dinero", r"no puedo pagar", r"sin dinero",
                r"no tengo recursos", r"estoy quebrado", r"no hay fondos"
            ],
            ObjectionType.NOT_MY_DEBT: [
                r"no es mía", r"no es mi deuda", r"no debo nada",
                r"no reconozco", r"nunca tuve", r"error"
            ],
            ObjectionType.CALL_LATER: [
                r"llamen después", r"la próxima semana", r"no es buen momento",
                r"estoy ocupado", r"llamen mañana", r"otro día"
            ],
            ObjectionType.FINANCIAL_HARDSHIP: [
                r"desempleado", r"sin trabajo", r"enfermo", r"hospital",
                r"divorciándome", r"problemas familiares", r"crisis"
            ],
            ObjectionType.DISPUTE_AMOUNT: [
                r"no es correcto el monto", r"cantidad incorrecta",
                r"no debe tanto", r"está mal calculado"
            ],
            ObjectionType.HOSTILE_RESPONSE: [
                r"déjenme en paz", r"no me molesten", r"ya basta",
                r"estoy harto", r"qué fastidio"
            ]
        }
        
        # Indicadores de perfil de deudor
        self.profile_indicators = {
            DebtorProfile.COOPERATIVE: [
                r"entiendo", r"quiero resolver", r"ayuden", r"solución",
                r"qué opciones", r"plan de pagos"
            ],
            DebtorProfile.EVASIVE: [
                r"tal vez", r"no sé", r"tengo que pensar",
                r"después veo", r"ya veremos"
            ],
            DebtorProfile.HOSTILE: [
                r"maldito", r"carajo", r"joder", r"fastidio",
                r"déjenme", r"no me jodan"
            ],
            DebtorProfile.FINANCIALLY_STRESSED: [
                r"no tengo", r"sin dinero", r"desempleado",
                r"crisis", r"difícil", r"problemas"
            ]
        }
        
        # Umbrales de escalación
        self.escalation_thresholds = {
            "hostile_language": 2,  # 2+ palabras hostiles
            "legal_threats": 1,     # Cualquier amenaza legal del deudor
            "dispute_debt": 1,      # Disputa formal de la deuda
            "payment_inability": 3  # 3+ indicadores de incapacidad real
        }
    
    def analyze_collections_context(self, context: CollectionsContext) -> Tuple[ObjectionType, DebtorProfile]:
        """
        Analiza el contexto específico de cobranza para detectar objeciones y perfil
        
        Args:
            context: Contexto de la conversación de cobranza
            
        Returns:
            Tupla con tipo de objeción y perfil del deudor
        """
        transcript_lower = context.recent_transcript.lower()
        
        # Detectar tipo de objeción
        objection_scores = {}
        for objection_type, patterns in self.objection_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, transcript_lower))
            if score > 0:
                objection_scores[objection_type] = score
        
        detected_objection = (
            max(objection_scores.items(), key=lambda x: x[1])[0] 
            if objection_scores else ObjectionType.UNKNOWN
        )
        
        # Detectar perfil del deudor
        profile_scores = {}
        for profile_type, indicators in self.profile_indicators.items():
            score = sum(1 for indicator in indicators if re.search(indicator, transcript_lower))
            if score > 0:
                profile_scores[profile_type] = score
        
        detected_profile = (
            max(profile_scores.items(), key=lambda x: x[1])[0]
            if profile_scores else DebtorProfile.COOPERATIVE
        )
        
        logger.info(f"Objeción detectada: {detected_objection.value}, Perfil: {detected_profile.value}")
        
        return detected_objection, detected_profile
    
    def build_collections_query(self, context: CollectionsContext, objection_type: ObjectionType) -> str:
        """
        Construye una consulta optimizada para situaciones de cobranza
        
        Args:
            context: Contexto de cobranza
            objection_type: Tipo de objeción detectada
            
        Returns:
            Query optimizada para RAG
        """
        query_parts = [objection_type.value]
        
        # Agregar contexto de cuenta si está disponible
        if context.days_past_due:
            if context.days_past_due > 90:
                query_parts.append("severely_delinquent")
            elif context.days_past_due > 30:
                query_parts.append("moderately_delinquent")
        
        # Agregar historial de promesas
        if context.previous_promises and context.previous_promises > 2:
            query_parts.append("broken_promises")
        
        # Agregar monto si es relevante
        if context.account_balance:
            if context.account_balance > 5000:
                query_parts.append("large_balance")
            elif context.account_balance < 500:
                query_parts.append("small_balance")
        
        return " ".join(query_parts)
    
    def evaluate_escalation_need(self, context: CollectionsContext, objection_type: ObjectionType) -> bool:
        """
        Evalúa si la situación requiere escalación
        
        Args:
            context: Contexto de cobranza
            objection_type: Tipo de objeción
            
        Returns:
            True si requiere escalación
        """
        escalation_triggers = [
            objection_type == ObjectionType.NOT_MY_DEBT,  # Disputa formal
            objection_type == ObjectionType.HOSTILE_RESPONSE,  # Hostilidad
            "validación" in context.recent_transcript.lower(),  # Solicitud legal
            "abogado" in context.recent_transcript.lower(),  # Mención legal
            context.previous_promises and context.previous_promises > 3  # Demasiadas promesas rotas
        ]
        
        return any(escalation_triggers)
    
    def calculate_compliance_risk(self, context: CollectionsContext, objection_type: ObjectionType) -> float:
        """
        Calcula el riesgo de compliance de la situación
        
        Args:
            context: Contexto de cobranza
            objection_type: Tipo de objeción
            
        Returns:
            Score de riesgo (0.0 = bajo riesgo, 1.0 = alto riesgo)
        """
        risk_factors = {
            ObjectionType.NOT_MY_DEBT: 0.9,  # Alto riesgo legal
            ObjectionType.HOSTILE_RESPONSE: 0.7,  # Riesgo de escalación
            ObjectionType.FINANCIAL_HARDSHIP: 0.3,  # Bajo riesgo si se maneja bien
            ObjectionType.ALREADY_PAID: 0.6,  # Riesgo medio
            ObjectionType.DISPUTE_AMOUNT: 0.5   # Riesgo medio
        }
        
        base_risk = risk_factors.get(objection_type, 0.2)
        
        # Ajustar por factores adicionales
        if "amenaza" in context.recent_transcript.lower():
            base_risk += 0.2
        
        if context.previous_promises and context.previous_promises > 2:
            base_risk += 0.1
        
        return min(1.0, base_risk)
    
    def suggest_payment_plan(self, context: CollectionsContext, debtor_profile: DebtorProfile) -> Optional[Dict[str, Any]]:
        """
        Sugiere un plan de pagos basado en el perfil y contexto
        
        Args:
            context: Contexto de cobranza
            debtor_profile: Perfil del deudor
            
        Returns:
            Diccionario con sugerencia de plan de pagos
        """
        if not context.account_balance:
            return None
        
        balance = context.account_balance
        
        # Estrategias por perfil
        if debtor_profile == DebtorProfile.FINANCIALLY_STRESSED:
            # Plan más conservador
            monthly_payment = min(balance * 0.05, 50)  # 5% o $50 máximo
            plan_months = max(6, int(balance / monthly_payment))
        elif debtor_profile == DebtorProfile.COOPERATIVE:
            # Plan estándar
            monthly_payment = min(balance * 0.15, 200)  # 15% o $200 máximo
            plan_months = max(3, int(balance / monthly_payment))
        elif debtor_profile == DebtorProfile.REPEAT_BREAKER:
            # Plan más agresivo con pagos menores pero más frecuentes
            weekly_payment = min(balance * 0.05, 50)
            plan_weeks = max(8, int(balance / weekly_payment))
            return {
                "type": "weekly",
                "amount": weekly_payment,
                "frequency": "weekly",
                "duration_weeks": plan_weeks,
                "total_payments": plan_weeks,
                "rationale": "Pagos frecuentes para reconstruir confianza"
            }
        else:
            # Plan por defecto
            monthly_payment = min(balance * 0.10, 150)
            plan_months = max(4, int(balance / monthly_payment))
        
        return {
            "type": "monthly",
            "amount": round(monthly_payment, 2),
            "frequency": "monthly",
            "duration_months": plan_months,
            "total_payments": plan_months,
            "rationale": f"Plan adaptado para perfil {debtor_profile.value}"
        }
    
    def get_collections_suggestions(
        self, 
        context: CollectionsContext,
        n_examples: int = 3
    ) -> CollectionsRAGResponse:
        """
        Obtiene sugerencias específicas para cobranza
        
        Args:
            context: Contexto de cobranza
            n_examples: Número de ejemplos a recuperar
            
        Returns:
            Respuesta especializada para cobranza
        """
        # 1. Analizar contexto específico de cobranza
        objection_type, debtor_profile = self.analyze_collections_context(context)
        
        # 2. Construir query especializada
        query = self.build_collections_query(context, objection_type)
        logger.info(f"Query de cobranza: {query}")
        
        # 3. Buscar ejemplos similares
        try:
            # Buscar en ejemplos AEIOU con filtro específico
            filter_criteria = {"objection_type": objection_type.value}
            
            aeiou_results = self.chroma.query_similar(
                "aeiou_examples",
                query,
                n_results=n_examples,
                where_filter=filter_criteria
            )
            
            # Buscar técnicas específicas
            technique_results = self.chroma.query_similar(
                "communication_styles",
                f"{objection_type.value} {debtor_profile.value}",
                n_results=1
            )
            
        except Exception as e:
            logger.error(f"Error en consulta RAG de cobranza: {e}")
            return self._fallback_collections_response(objection_type, debtor_profile)
        
        # 4. Procesar resultados
        if not aeiou_results["documents"]:
            return self._fallback_collections_response(objection_type, debtor_profile)
        
        # 5. Construir respuesta especializada
        source_examples = []
        for i, doc in enumerate(aeiou_results["documents"]):
            source_examples.append({
                "content": doc,
                "metadata": aeiou_results["metadatas"][i],
                "similarity_distance": aeiou_results["distances"][i],
                "id": aeiou_results["ids"][i]
            })
        
        # Calcular métricas específicas
        avg_distance = sum(aeiou_results["distances"]) / len(aeiou_results["distances"])
        confidence_score = max(0.0, 1.0 - avg_distance)
        
        compliance_risk = self.calculate_compliance_risk(context, objection_type)
        compliance_score = 1.0 - compliance_risk
        
        escalation_needed = self.evaluate_escalation_need(context, objection_type)
        
        # Sugerir plan de pagos si es apropiado
        payment_plan = None
        if objection_type in [ObjectionType.CANNOT_PAY, ObjectionType.FINANCIAL_HARDSHIP]:
            payment_plan = self.suggest_payment_plan(context, debtor_profile)
        
        # Seleccionar mejor respuesta
        best_response = source_examples[0]["content"]
        
        # Generar razonamiento específico
        reasoning = self._generate_collections_reasoning(
            objection_type, debtor_profile, source_examples, compliance_score
        )
        
        return CollectionsRAGResponse(
            suggested_response=best_response,
            confidence_score=confidence_score,
            source_examples=source_examples,
            conflict_type=self.conflict_keywords.get(objection_type, "collections_objection"),
            reasoning=reasoning,
            objection_type=objection_type,
            debtor_profile=debtor_profile,
            compliance_score=compliance_score,
            escalation_recommended=escalation_needed,
            payment_plan_suggestion=payment_plan
        )
    
    def _fallback_collections_response(self, objection_type: ObjectionType, debtor_profile: DebtorProfile) -> CollectionsRAGResponse:
        """Respuesta de fallback específica para cobranza"""
        fallback_responses = {
            ObjectionType.ALREADY_PAID: "Entiendo que crees que ya pagaste (A). Quiero verificar la información en nuestros registros (E). ¿Podrías ayudarme con la fecha y método de pago? (I) Mi objetivo es resolver cualquier discrepancia (O). ¿Tienes el número de confirmación? (U)",
            ObjectionType.CANNOT_PAY: "Reconozco que la situación está difícil (A). Queremos encontrar una solución que funcione (E). ¿Qué cantidad pequeña podrías manejar semanalmente? (I) Mi objetivo es evitar que esto afecte más tu crédito (O). ¿Te parece razonable $25 por semana? (U)",
            ObjectionType.NOT_MY_DEBT: "Entiendo que no reconoces esta deuda (A). Es importante verificar toda la información (E). Voy a enviarte la documentación completa por correo (I). Mi objetivo es que tengas todos los detalles (O). ¿Cuál es tu dirección postal actual? (U)",
            ObjectionType.CALL_LATER: "Entiendo que no es buen momento (A). Prefiero llamar cuando sea conveniente (E). ¿Qué día específico sería mejor para 5 minutos? (I) Mi objetivo es resolver esto rápidamente (O). ¿Te parece mañana a las 2 PM? (U)"
        }
        
        response = fallback_responses.get(
            objection_type, 
            "Entiendo tu situación (A). Quiero ayudarte a encontrar una solución (E). ¿Podríamos explorar algunas opciones? (I) Mi objetivo es resolver esto de manera justa (O). ¿Qué sería más útil para ti? (U)"
        )
        
        return CollectionsRAGResponse(
            suggested_response=response,
            confidence_score=0.4,
            source_examples=[],
            conflict_type="collections_objection",
            reasoning="Respuesta genérica de fallback - base de conocimiento requiere más ejemplos",
            objection_type=objection_type,
            debtor_profile=debtor_profile,
            compliance_score=0.8,  # Fallbacks son conservadores
            escalation_recommended=objection_type in [ObjectionType.NOT_MY_DEBT, ObjectionType.HOSTILE_RESPONSE]
        )
    
    def _generate_collections_reasoning(self, objection_type: ObjectionType, debtor_profile: DebtorProfile, examples: List[Dict], compliance_score: float) -> str:
        """Genera razonamiento específico para cobranza"""
        if not examples:
            return "Sin ejemplos similares - usando respuesta estándar"
        
        best_example = examples[0]
        effectiveness = best_example["metadata"].get("effectiveness_score", 0)
        
        reasoning = f"Basado en objeción '{objection_type.value}' con perfil '{debtor_profile.value}'. "
        reasoning += f"Efectividad histórica: {effectiveness:.0%}. "
        reasoning += f"Compliance score: {compliance_score:.0%}. "
        
        if len(examples) > 1:
            reasoning += f"Consultados {len(examples)} casos similares. "
        
        # Agregar nota de compliance si es baja
        if compliance_score < 0.7:
            reasoning += "⚠️ Situación de alto riesgo - usar lenguaje muy cuidadoso."
        
        return reasoning


if __name__ == "__main__":
    # Test del collections query engine
    from chroma_manager import ChromaManager
    
    chroma = ChromaManager()
    engine = CollectionsQueryEngine(chroma)
    
    # Context de prueba para cobranza
    test_context = CollectionsContext(
        current_speaker="OTHER",  # Cliente hablando
        recent_transcript="Ya pagué esa deuda el mes pasado, están equivocados",
        meeting_type="collections_call",
        debtor_id="DBT_12345",
        account_balance=1250.00,
        days_past_due=45,
        previous_promises=1
    )
    
    response = engine.get_collections_suggestions(test_context)
    
    print("=== Test Collections Query Engine ===")
    print(f"Objeción detectada: {response.objection_type.value}")
    print(f"Perfil del deudor: {response.debtor_profile.value}")
    print(f"Confianza: {response.confidence_score:.2f}")
    print(f"Compliance score: {response.compliance_score:.2f}")
    print(f"Escalación recomendada: {response.escalation_recommended}")
    print(f"\nSugerencia: {response.suggested_response}")
    print(f"\nRazonamiento: {response.reasoning}")
    
    if response.payment_plan_suggestion:
        plan = response.payment_plan_suggestion
        print(f"\nPlan de pagos sugerido:")
        print(f"  Tipo: {plan['type']}")
        print(f"  Monto: ${plan['amount']}")
        print(f"  Frecuencia: {plan['frequency']}")
        print(f"  Duración: {plan.get('duration_months', plan.get('duration_weeks', 'N/A'))}")
