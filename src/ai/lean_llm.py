"""
LLM local optimizado para desarrollo lean
Enfoque: Generar sugerencias AEIOU para reuniones usando modelos pequeños sin GPU
"""

import asyncio
import time
import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class SuggestionTrigger(Enum):
    """Tipos de situaciones que requieren sugerencias AEIOU"""
    TENSION = "tension"
    DISAGREEMENT = "disagreement"
    DIFFICULT_QUESTION = "difficult_question"
    CONFLICT = "conflict"
    DEADLINE_PRESSURE = "deadline_pressure"
    TECHNICAL_DISPUTE = "technical_dispute"


@dataclass
class AIResponse:
    """Respuesta del LLM lean"""
    text: str
    confidence: float
    processing_time: float
    trigger_type: SuggestionTrigger
    aeiou_components: Dict[str, str]
    
    def __post_init__(self):
        """Valida que la respuesta tenga estructura AEIOU"""
        if not self.aeiou_components:
            self.aeiou_components = self._extract_aeiou_components()
    
    def _extract_aeiou_components(self) -> Dict[str, str]:
        """Extrae componentes AEIOU del texto"""
        # Implementación simple para detectar componentes
        text_lower = self.text.lower()
        
        components = {}
        
        # A - Acknowledge (reconocer)
        if any(word in text_lower for word in ['entiendo', 'comprendo', 'reconozco', 'veo']):
            components['acknowledge'] = "Presente"
        
        # E - Express (expresar)
        if any(word in text_lower for word in ['yo siento', 'yo pienso', 'mi perspectiva', 'considero']):
            components['express'] = "Presente"
        
        # I - Identify (identificar solución)
        if any(word in text_lower for word in ['propongo', 'sugiero', 'podríamos', 'qué tal si']):
            components['identify'] = "Presente"
        
        # O - Outcome (resultado deseado)
        if any(word in text_lower for word in ['objetivo', 'meta', 'resultado', 'queremos lograr']):
            components['outcome'] = "Presente"
        
        # U - Understanding (comprensión mutua)
        if any(word in text_lower for word in ['qué piensas', 'cómo lo ves', 'tu opinión', 'entendiste']):
            components['understanding'] = "Presente"
        
        return components


class LocalLLM:
    """LLM local optimizado para desarrollo lean y AEIOU"""
    
    def __init__(self, 
                 model_name: str = "qwen2.5:0.5b",
                 base_url: str = "http://localhost:11434",
                 max_context_length: int = 1024,
                 temperature: float = 0.3):
        
        self.model_name = model_name
        self.base_url = base_url
        self.max_context_length = max_context_length
        self.temperature = temperature
        
        # Verificar conexión
        self._verify_connection()
        
        # Templates AEIOU optimizados
        self.aeiou_template = self._create_aeiou_template()
        
        logger.info(f"🧠 LocalLLM lean inicializado: {model_name}")
    
    def _verify_connection(self) -> bool:
        """Verifica conexión con Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                if self.model_name in models:
                    logger.success(f"✅ Conectado a {self.model_name}")
                    return True
                else:
                    logger.warning(f"⚠️ Modelo {self.model_name} no encontrado")
                    logger.info(f"💡 Instalar con: ollama pull {self.model_name}")
                    return False
            else:
                raise ConnectionError(f"Ollama status: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Error conectando a Ollama: {e}")
            logger.info("💡 Iniciar Ollama con: ollama serve")
            return False
    
    def _create_aeiou_template(self) -> str:
        """Template optimizado para respuestas AEIOU"""
        return \"\"\"\nEres un asistente experto en comunicación no-violenta que usa el framework AEIOU para generar respuestas diplomáticas en reuniones de trabajo.\n\nFramework AEIOU:\n- A (Acknowledge): Reconoce y valida la perspectiva de la otra persona\n- E (Express): Expresa tu posición usando \"yo\" en lugar de \"tú\"\n- I (Identify): Propone una solución específica y constructiva\n- O (Outcome): Define claramente el resultado que buscas\n- U (Understanding): Busca comprensión mutua haciendo preguntas\n\nPrincipios importantes:\n- Mantén un tono profesional pero empático\n- Evita lenguaje confrontacional o acusatorio\n- Enfócate en soluciones, no en culpas\n- Usa preguntas abiertas para generar diálogo\n- Mantén respuestas concisas (máximo 2-3 oraciones)\n\nSituación actual: {situation}\nContexto adicional: {context}\n\nGenera una respuesta AEIOU apropiada:\"\"\"\n    \n    async def generate_aeiou_suggestion(self,\n                                      user_speech: str,\n                                      conversation_context: List[Dict],\n                                      rag_context: List[Dict] = None) -> AIResponse:\n        \"\"\"Genera sugerencia AEIOU contextual\"\"\"\n        \n        start_time = time.time()\n        \n        try:\n            # Detectar tipo de trigger\n            trigger_type = self._detect_trigger_type(user_speech)\n            \n            # Construir contexto\n            context_summary = self._build_context_summary(conversation_context, rag_context)\n            \n            # Generar prompt\n            prompt = self.aeiou_template.format(\n                situation=user_speech,\n                context=context_summary\n            )\n            \n            # Llamar al LLM\n            response_text = await self._call_ollama(prompt)\n            \n            # Calcular métricas\n            processing_time = time.time() - start_time\n            confidence = self._calculate_confidence(response_text, user_speech)\n            \n            return AIResponse(\n                text=response_text,\n                confidence=confidence,\n                processing_time=processing_time,\n                trigger_type=trigger_type,\n                aeiou_components={}\n            )\n            \n        except Exception as e:\n            logger.error(f\"❌ Error generando sugerencia: {e}\")\n            return self._create_fallback_response(start_time)\n    \n    def _detect_trigger_type(self, text: str) -> SuggestionTrigger:\n        \"\"\"Detecta tipo de situación que requiere AEIOU\"\"\"\n        text_lower = text.lower()\n        \n        # Tensión/conflicto\n        tension_words = ['problema', 'no funciona', 'mal', 'error', 'fallando']\n        if any(word in text_lower for word in tension_words):\n            return SuggestionTrigger.TENSION\n        \n        # Desacuerdo\n        disagreement_words = ['no estoy de acuerdo', 'no creo', 'pienso diferente']\n        if any(phrase in text_lower for phrase in disagreement_words):\n            return SuggestionTrigger.DISAGREEMENT\n        \n        # Presión de deadlines\n        deadline_words = ['deadline', 'fecha límite', 'urgente', 'rápido', 'tiempo']\n        if any(word in text_lower for word in deadline_words):\n            return SuggestionTrigger.DEADLINE_PRESSURE\n        \n        # Disputa técnica\n        tech_words = ['implementación', 'código', 'arquitectura', 'solución técnica']\n        if any(word in text_lower for word in tech_words):\n            return SuggestionTrigger.TECHNICAL_DISPUTE\n        \n        # Pregunta difícil\n        if '?' in text and len(text.split()) > 10:\n            return SuggestionTrigger.DIFFICULT_QUESTION\n        \n        return SuggestionTrigger.CONFLICT  # Default\n    \n    def _build_context_summary(self, \n                              conversation_context: List[Dict],\n                              rag_context: List[Dict] = None) -> str:\n        \"\"\"Construye resumen del contexto\"\"\"\n        summary_parts = []\n        \n        # Contexto de conversación reciente\n        if conversation_context:\n            recent_context = conversation_context[-3:]  # Últimas 3 intervenciones\n            summary_parts.append(f\"Conversación reciente: {len(recent_context)} intervenciones\")\n        \n        # Contexto RAG si está disponible\n        if rag_context:\n            summary_parts.append(f\"Ejemplos similares encontrados: {len(rag_context)}\")\n            if rag_context:\n                summary_parts.append(f\"Efectividad promedio: {rag_context[0].get('effectiveness', 0.8):.0%}\")\n        \n        return \", \".join(summary_parts) if summary_parts else \"Contexto limitado\"\n    \n    async def _call_ollama(self, prompt: str) -> str:\n        \"\"\"Llama al modelo Ollama\"\"\"\n        payload = {\n            \"model\": self.model_name,\n            \"prompt\": prompt,\n            \"stream\": False,\n            \"options\": {\n                \"temperature\": self.temperature,\n                \"num_predict\": 100,  # Respuestas concisas\n                \"top_k\": 10,\n                \"top_p\": 0.9,\n                \"stop\": [\"\\n\\n\"]  # Parar en doble salto de línea\n            }\n        }\n        \n        # Usar asyncio para no bloquear\n        loop = asyncio.get_event_loop()\n        \n        def make_request():\n            response = requests.post(\n                f\"{self.base_url}/api/generate\",\n                json=payload,\n                timeout=10  # Timeout corto para lean\n            )\n            response.raise_for_status()\n            return response.json().get('response', '').strip()\n        \n        # Ejecutar en thread pool para no bloquear async\n        response_text = await loop.run_in_executor(None, make_request)\n        \n        if not response_text:\n            raise ValueError(\"LLM no generó respuesta\")\n        \n        return response_text\n    \n    def _calculate_confidence(self, response: str, original_text: str) -> float:\n        \"\"\"Calcula confianza de la respuesta\"\"\"\n        confidence = 0.5  # Base\n        \n        # Factor 1: Longitud apropiada (50-200 caracteres)\n        length = len(response)\n        if 50 <= length <= 200:\n            confidence += 0.3\n        elif 30 <= length <= 300:\n            confidence += 0.1\n        \n        # Factor 2: Presencia de estructura AEIOU\n        aeiou_indicators = ['entiendo', 'yo', 'propongo', 'objetivo', 'qué']\n        found = sum(1 for word in aeiou_indicators if word.lower() in response.lower())\n        confidence += min(0.4, found * 0.1)\n        \n        # Factor 3: Evita repetir exactamente el problema\n        original_words = set(original_text.lower().split())\n        response_words = set(response.lower().split())\n        overlap = len(original_words.intersection(response_words))\n        if overlap < len(original_words) * 0.5:  # Menos del 50% de overlap\n            confidence += 0.2\n        \n        return min(1.0, confidence)\n    \n    def _create_fallback_response(self, start_time: float) -> AIResponse:\n        \"\"\"Crea respuesta de fallback si el LLM falla\"\"\"\n        fallback_responses = [\n            \"Entiendo tu perspectiva (A). Yo también veo algunos desafíos aquí (E). ¿Podríamos explorar algunas alternativas juntos? (I) Mi objetivo es que encontremos una solución que funcione para todos (O). ¿Qué opinas de esta propuesta? (U)\",\n            \"Reconozco que esto es importante para ti (A). Yo pienso que hay varias maneras de abordar esto (E). ¿Qué tal si revisamos las opciones disponibles? (I) Queremos llegar a un acuerdo que beneficie al proyecto (O). ¿Cómo te suena esto? (U)\",\n            \"Veo que tienes preocupaciones válidas (A). Yo también quiero asegurarme de que tomemos la decisión correcta (E). ¿Podríamos analizar esto desde diferentes ángulos? (I) El resultado que busco es una solución sólida (O). ¿Qué información adicional necesitarías? (U)\"\n        ]\n        \n        import random\n        fallback_text = random.choice(fallback_responses)\n        \n        return AIResponse(\n            text=fallback_text,\n            confidence=0.6,  # Confianza media para fallbacks\n            processing_time=time.time() - start_time,\n            trigger_type=SuggestionTrigger.CONFLICT,\n            aeiou_components={}\n        )\n    \n    async def test_connection(self) -> Dict[str, Any]:\n        \"\"\"Prueba la conexión con una consulta simple\"\"\"\n        try:\n            start_time = time.time()\n            \n            test_prompt = \"Responde brevemente: ¿Estás funcionando correctamente?\"\n            response = await self._call_ollama(test_prompt)\n            \n            return {\n                \"success\": True,\n                \"response_time\": time.time() - start_time,\n                \"model\": self.model_name,\n                \"response\": response[:100]  # Primeros 100 caracteres\n            }\n            \n        except Exception as e:\n            return {\n                \"success\": False,\n                \"error\": str(e),\n                \"model\": self.model_name\n            }\n\n\n# Factory function para configuración lean\ndef create_lean_llm(model_name: str = \"qwen2.5:0.5b\") -> LocalLLM:\n    \"\"\"Factory para crear LLM con configuración lean optimizada\"\"\"\n    return LocalLLM(\n        model_name=model_name,\n        max_context_length=1024,  # Contexto reducido para velocidad\n        temperature=0.3  # Respuestas más determinísticas\n    )\n\n\nif __name__ == \"__main__\":\n    \"\"\"Test del LLM lean\"\"\"\n    \n    async def test_lean_llm():\n        logger.info(\"🧪 Iniciando test de LLM lean...\")\n        \n        llm = create_lean_llm()\n        \n        # Test de conexión\n        connection_test = await llm.test_connection()\n        if connection_test[\"success\"]:\n            logger.success(f\"✅ Conexión exitosa en {connection_test['response_time']:.2f}s\")\n        else:\n            logger.error(f\"❌ Error: {connection_test['error']}\")\n            return\n        \n        # Test de sugerencia AEIOU\n        logger.info(\"💡 Generando sugerencia AEIOU...\")\n        \n        test_speech = \"No estoy de acuerdo con la implementación que propones, creo que va a causar problemas.\"\n        test_context = [\n            {\"text\": \"Estamos discutiendo la arquitectura del nuevo sistema\", \"speaker\": \"other\"},\n            {\"text\": \"Necesitamos decidir esto antes del deadline\", \"speaker\": \"other\"}\n        ]\n        \n        suggestion = await llm.generate_aeiou_suggestion(\n            user_speech=test_speech,\n            conversation_context=test_context\n        )\n        \n        logger.info(f\"📝 Sugerencia generada:\")\n        logger.info(f\"   Texto: {suggestion.text}\")\n        logger.info(f\"   Confianza: {suggestion.confidence:.2f}\")\n        logger.info(f\"   Tiempo: {suggestion.processing_time:.2f}s\")\n        logger.info(f\"   Trigger: {suggestion.trigger_type.value}\")\n        \n        logger.success(\"✅ Test completado\")\n    \n    asyncio.run(test_lean_llm())\n