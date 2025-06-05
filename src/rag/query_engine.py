"""
Motor de consultas RAG

Este módulo proporciona una interfaz de alto nivel para realizar
consultas inteligentes a la knowledge base y generar contexto
relevante para el LLM.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Tipos de conflicto detectables"""
    DEADLINE_PRESSURE = "deadline_pressure"
    TECHNICAL_DISAGREEMENT = "technical_disagreement"
    COMMUNICATION_BREAKDOWN = "communication_breakdown"
    TEAM_DYNAMICS = "team_dynamics"
    DECISION_PARALYSIS = "decision_paralysis"
    RESOURCE_ALLOCATION = "resource_allocation"
    PERFORMANCE_CONCERNS = "performance_concerns"
    UNKNOWN = "unknown"


@dataclass
class ConversationContext:
    """Contexto de la conversación actual"""
    current_speaker: str  # "USER" o "OTHER"
    recent_transcript: str
    detected_emotion: Optional[str] = None
    conflict_indicators: List[str] = None
    meeting_type: Optional[str] = None
    participant_count: Optional[int] = None
    
    def __post_init__(self):
        if self.conflict_indicators is None:
            self.conflict_indicators = []


@dataclass 
class RAGResponse:
    """Respuesta del sistema RAG"""
    suggested_response: str
    confidence_score: float
    source_examples: List[Dict[str, Any]]
    conflict_type: ConflictType
    reasoning: str


class QueryEngine:
    """Motor de consultas RAG para sugerencias contextuales"""
    
    def __init__(self, chroma_manager: ChromaManager):
        self.chroma = chroma_manager
        
        # Palabras clave para detección de conflictos
        self.conflict_keywords = {
            ConflictType.DEADLINE_PRESSURE: [
                "deadline", "presión", "tiempo", "urgente", "rápido", "ya", "cuando"
            ],
            ConflictType.TECHNICAL_DISAGREEMENT: [
                "implementación", "técnico", "código", "arquitectura", "enfoque", "método"
            ],
            ConflictType.COMMUNICATION_BREAKDOWN: [
                "entender", "claro", "confuso", "explicar", "punto", "significa"
            ],
            ConflictType.TEAM_DYNAMICS: [
                "equipo", "roles", "responsabilidad", "quien", "participación"
            ],
            ConflictType.DECISION_PARALYSIS: [
                "decidir", "opción", "alternativa", "elegir", "qué hacer"
            ]
        }
    
    def analyze_conversation_context(self, context: ConversationContext) -> ConflictType:
        """
        Analiza el contexto y detecta el tipo de conflicto potencial
        
        Args:
            context: Contexto de la conversación
            
        Returns:
            Tipo de conflicto detectado
        """
        transcript_lower = context.recent_transcript.lower()
        
        # Scoring por tipo de conflicto
        conflict_scores = {}
        
        for conflict_type, keywords in self.conflict_keywords.items():
            score = sum(1 for keyword in keywords if keyword in transcript_lower)
            if score > 0:
                conflict_scores[conflict_type] = score
        
        # Detectar indicadores adicionales
        negative_indicators = [
            "no", "pero", "sin embargo", "problema", "error", "mal", "incorrecto"
        ]
        
        negative_score = sum(1 for indicator in negative_indicators if indicator in transcript_lower)
        
        # Ajustar scores basado en negatividad
        for conflict_type in conflict_scores:
            conflict_scores[conflict_type] += negative_score * 0.5
        
        # Devolver el tipo con mayor score
        if conflict_scores:
            return max(conflict_scores.items(), key=lambda x: x[1])[0]
        
        return ConflictType.UNKNOWN
    
    def build_context_query(self, context: ConversationContext, conflict_type: ConflictType) -> str:
        """
        Construye una consulta contextual para el RAG
        
        Args:
            context: Contexto de la conversación
            conflict_type: Tipo de conflicto detectado
            
        Returns:
            Query optimizada para RAG
        """
        # Extraer palabras clave del transcript
        key_phrases = []
        
        # Añadir palabras específicas del tipo de conflicto
        if conflict_type in self.conflict_keywords:
            for keyword in self.conflict_keywords[conflict_type]:
                if keyword in context.recent_transcript.lower():
                    key_phrases.append(keyword)
        
        # Construir query
        base_query = f"{conflict_type.value} "
        
        if context.meeting_type:
            base_query += f"{context.meeting_type} "
            
        base_query += " ".join(key_phrases[:3])  # Limitar a 3 palabras clave
        
        return base_query.strip()
    
    def get_contextual_suggestions(
        self, 
        context: ConversationContext,
        n_examples: int = 3
    ) -> RAGResponse:
        """
        Obtiene sugerencias contextuales basadas en la conversación
        
        Args:
            context: Contexto de la conversación actual
            n_examples: Número de ejemplos a recuperar
            
        Returns:
            Respuesta RAG con sugerencias
        """
        # 1. Detectar tipo de conflicto
        conflict_type = self.analyze_conversation_context(context)
        logger.info(f"Conflicto detectado: {conflict_type.value}")
        
        # 2. Construir query contextual
        query = self.build_context_query(context, conflict_type)
        logger.info(f"Query RAG: {query}")
        
        # 3. Buscar ejemplos similares
        try:
            # Buscar primero en ejemplos AEIOU
            aeiou_results = self.chroma.query_similar(
                "aeiou_examples",
                query,
                n_results=n_examples,
                where_filter={"category": conflict_type.value} if conflict_type != ConflictType.UNKNOWN else None
            )
            
            # Buscar en patrones de conflicto para contexto adicional
            pattern_results = self.chroma.query_similar(
                "conflict_patterns",
                query,
                n_results=1
            )
            
        except Exception as e:
            logger.error(f"Error en consulta RAG: {e}")
            return self._fallback_response(conflict_type)
        
        # 4. Procesar resultados
        if not aeiou_results["documents"]:
            return self._fallback_response(conflict_type)
        
        # 5. Construir respuesta
        source_examples = []
        for i, doc in enumerate(aeiou_results["documents"]):
            source_examples.append({
                "content": doc,
                "metadata": aeiou_results["metadatas"][i],
                "similarity_distance": aeiou_results["distances"][i],
                "id": aeiou_results["ids"][i]
            })
        
        # Calcular confidence score basado en similarity
        avg_distance = sum(aeiou_results["distances"]) / len(aeiou_results["distances"])
        confidence_score = max(0.0, 1.0 - avg_distance)  # Convertir distancia a confianza
        
        # Seleccionar mejor ejemplo como respuesta sugerida
        best_example = source_examples[0]["content"]
        
        # Generar reasoning
        reasoning = self._generate_reasoning(conflict_type, source_examples)
        
        return RAGResponse(
            suggested_response=best_example,
            confidence_score=confidence_score,
            source_examples=source_examples,
            conflict_type=conflict_type,
            reasoning=reasoning
        )
    
    def _fallback_response(self, conflict_type: ConflictType) -> RAGResponse:
        """Respuesta de fallback cuando no se encuentran ejemplos específicos"""
        fallback_responses = {
            ConflictType.DEADLINE_PRESSURE: "Entiendo la presión de tiempo (A). Yo también siento la urgencia (E). ¿Podríamos revisar las prioridades juntos? (I) Mi objetivo es encontrar una solución realista (O). ¿Qué opciones ves para el deadline? (U)",
            ConflictType.TECHNICAL_DISAGREEMENT: "Reconozco que tienes una perspectiva técnica diferente (A). Yo valoro tu experiencia (E). ¿Qué te parece si evaluamos ambos enfoques? (I) Mi objetivo es tomar la mejor decisión técnica (O). ¿Qué criterios deberíamos usar? (U)",
            ConflictType.COMMUNICATION_BREAKDOWN: "Entiendo que puede no estar siendo claro (A). Yo quiero asegurarme de comunicar bien (E). ¿Podrías ayudarme a entender qué necesitas que clarifique? (I) Mi objetivo es que ambos estemos en la misma página (O). ¿Cómo puedo explicarlo mejor? (U)",
            ConflictType.UNKNOWN: "Entiendo tu perspectiva (A). Yo valoro tu punto de vista (E). ¿Podríamos explorar esto juntos? (I) Mi objetivo es que lleguemos a un entendimiento mutuo (O). ¿Qué sería más útil para ti? (U)"
        }
        
        return RAGResponse(
            suggested_response=fallback_responses.get(conflict_type, fallback_responses[ConflictType.UNKNOWN]),
            confidence_score=0.3,  # Baja confianza para fallbacks
            source_examples=[],
            conflict_type=conflict_type,
            reasoning="Respuesta genérica - no se encontraron ejemplos específicos en la knowledge base"
        )
    
    def _generate_reasoning(self, conflict_type: ConflictType, examples: List[Dict]) -> str:
        """Genera explicación del razonamiento detrás de la sugerencia"""
        if not examples:
            return "Sin ejemplos similares disponibles"
        
        best_example = examples[0]
        effectiveness = best_example["metadata"].get("effectiveness_score", 0)
        
        reasoning = f"Basado en situación similar de tipo '{conflict_type.value}' con efectividad de {effectiveness:.0%}. "
        
        if len(examples) > 1:
            reasoning += f"Consultados {len(examples)} ejemplos similares. "
        
        # Añadir contexto específico si está disponible
        context_type = best_example["metadata"].get("context_type")
        if context_type:
            reasoning += f"Contexto: {context_type}."
        
        return reasoning


if __name__ == "__main__":
    # Test del query engine
    chroma = ChromaManager()
    engine = QueryEngine(chroma)
    
    # Context de prueba
    test_context = ConversationContext(
        current_speaker="OTHER",
        recent_transcript="No estás entendiendo el punto principal del deadline del proyecto",
        meeting_type="technical_meeting"
    )
    
    response = engine.get_contextual_suggestions(test_context)
    
    print(f"Conflicto detectado: {response.conflict_type.value}")
    print(f"Confianza: {response.confidence_score:.2f}")
    print(f"Sugerencia: {response.suggested_response}")
    print(f"Razonamiento: {response.reasoning}")
