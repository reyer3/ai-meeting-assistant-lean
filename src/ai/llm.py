"""
Local LLM Integration para Collections

Este módulo integra modelos de lenguaje locales (Ollama) para:
- Generar respuestas AEIOU contextuales
- Analizar sentimientos y objeciones
- Crear sugerencias de cobranza personalizadas
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import requests
from urllib.parse import urljoin

from ..core.config import config
from ..rag.collections_query_engine import CollectionsRAGResponse, ObjectionType, DebtorProfile

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Tipos de respuesta que puede generar el LLM"""
    AEIOU_RESPONSE = "aeiou_response"
    OBJECTION_HANDLING = "objection_handling"
    PAYMENT_PLAN = "payment_plan"
    ESCALATION_SCRIPT = "escalation_script"
    COMPLIANCE_CHECK = "compliance_check"


@dataclass
class LLMPrompt:
    """Prompt estructurado para el LLM"""
    system_prompt: str
    user_prompt: str
    context: Dict[str, Any]
    response_type: ResponseType
    max_tokens: int = 150
    temperature: float = 0.3


@dataclass
class LLMResponse:
    """Respuesta del LLM"""
    text: str
    confidence: float
    response_type: ResponseType
    processing_time: float
    tokens_used: int
    metadata: Dict[str, Any]


class CollectionsLLM:
    """Integrador de LLM local para cobranza"""
    
    def __init__(self, 
                 model_name: str = None,
                 base_url: str = "http://localhost:11434",
                 timeout: int = None):
        
        self.model_name = model_name or config.llm.model_name
        self.base_url = base_url
        self.timeout = timeout or config.llm.timeout
        
        # Verificar conexión con Ollama
        self._verify_ollama_connection()
        
        # Templates de prompts especializados
        self.prompt_templates = self._load_prompt_templates()
        
        # Métricas
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "total_tokens": 0,
            "errors": 0
        }
        
        logger.info(f"CollectionsLLM inicializado con modelo: {self.model_name}")
    
    def _verify_ollama_connection(self):
        """Verifica conexión con Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name in model_names:
                    logger.info(f"✅ Conexión con Ollama exitosa, modelo {self.model_name} disponible")
                else:
                    logger.warning(f"⚠️ Modelo {self.model_name} no encontrado. Modelos disponibles: {model_names}")
            else:
                raise ConnectionError(f"Ollama respondió con status {response.status_code}")
                
        except requests.RequestException as e:
            logger.error(f"❌ Error conectando con Ollama: {e}")
            logger.info("Asegúrate de que Ollama esté ejecutándose: 'ollama serve'")
            raise
    
    def _load_prompt_templates(self) -> Dict[ResponseType, str]:
        """Carga templates de prompts especializados"""
        return {
            ResponseType.AEIOU_RESPONSE: """
Eres un asistente experto en cobranza que usa el framework AEIOU para generar respuestas empáticas pero efectivas.

Framework AEIOU:
- A (Acknowledge): Reconoce la perspectiva del cliente
- E (Express): Expresa tu posición con "yo siento/pienso"
- I (Identify): Propone una solución específica
- O (Outcome): Define el resultado deseado
- U (Understanding): Busca comprensión mutua

SIEMPRE:
- Mantén un tono profesional y empático
- Cumple con regulaciones FDCPA
- Evita lenguaje coercitivo o amenazante
- Enfoca en soluciones, no en problemas

Contexto de la situación:
{context}

Genera una respuesta AEIOU apropiada:""",
            
            ResponseType.OBJECTION_HANDLING: """
Eres un experto en manejo de objeciones de cobranza. Tu trabajo es convertir objeciones en oportunidades de resolución.

Objeción detectada: {objection_type}
Perfil del cliente: {debtor_profile}

Contexto:
{context}

Genera una respuesta que:
1. Valide la preocupación del cliente
2. Redirija hacia una solución
3. Mantenga la conversación productiva
4. Cumpla con regulaciones de cobranza

Respuesta:""",
            
            ResponseType.PAYMENT_PLAN: """
Eres un especialista en estructurar planes de pago realistas y atractivos para el cliente.

Información de la cuenta:
Balance: ${balance}
Días vencidos: {days_past_due}
Perfil del cliente: {debtor_profile}

Contexto adicional:
{context}

Crea una propuesta de plan de pagos que:
1. Sea realista para la situación del cliente
2. Maximice la probabilidad de cumplimiento
3. Include términos claros y específicos
4. Mencione beneficios de cumplir el plan

Propuesta:""",
            
            ResponseType.ESCALATION_SCRIPT: """
Eres un especialista en escalación profesional de casos de cobranza.

Razón de escalación: {escalation_reason}
Contexto: {context}

Genera un script de escalación que:
1. Mantenga la relación profesional
2. Explique claramente los próximos pasos
3. Dé al cliente una última oportunidad de resolver
4. Cumpla con todas las regulaciones legales

Script de escalación:""",
            
            ResponseType.COMPLIANCE_CHECK: """
Eres un auditor de compliance especializado en regulaciones de cobranza (FDCPA, TCPA, etc.).

Analiza el siguiente mensaje para detectar posibles violaciones:

Mensaje: {message}

Evalúa:
1. Cumplimiento con FDCPA
2. Lenguaje apropiado y profesional
3. Ausencia de amenazas o coerción
4. Precisión de la información

Respuesta en formato JSON:
{
  "compliant": true/false,
  "violations": ["lista de violaciones"],
  "suggestions": ["mejoras sugeridas"],
  "risk_level": "low/medium/high"
}"""
        }
    
    def generate_aeiou_response(self, 
                              rag_response: CollectionsRAGResponse,
                              additional_context: Dict[str, Any] = None) -> LLMResponse:
        """Genera respuesta AEIOU basada en contexto RAG"""
        
        context = {
            "objection_type": rag_response.objection_type.value,
            "debtor_profile": rag_response.debtor_profile.value,
            "rag_suggestion": rag_response.suggested_response,
            "confidence": rag_response.confidence_score,
            "similar_examples": len(rag_response.source_examples),
            "compliance_score": rag_response.compliance_score
        }
        
        if additional_context:
            context.update(additional_context)
        
        prompt = LLMPrompt(
            system_prompt=self.prompt_templates[ResponseType.AEIOU_RESPONSE],
            user_prompt=f"Basado en la situación descrita, genera una respuesta AEIOU mejorada que sea más personalizada que: '{rag_response.suggested_response}'",
            context=context,
            response_type=ResponseType.AEIOU_RESPONSE,
            temperature=0.3  # Baja creatividad para consistency
        )
        
        return self._generate_response(prompt)
    
    def handle_objection(self, 
                        objection_type: ObjectionType,
                        debtor_profile: DebtorProfile,
                        context: Dict[str, Any]) -> LLMResponse:
        """Maneja objeción específica"""
        
        prompt_context = {
            "objection_type": objection_type.value,
            "debtor_profile": debtor_profile.value,
            **context
        }
        
        prompt = LLMPrompt(
            system_prompt=self.prompt_templates[ResponseType.OBJECTION_HANDLING].format(**prompt_context),
            user_prompt="Genera la mejor respuesta para esta objeción específica.",
            context=prompt_context,
            response_type=ResponseType.OBJECTION_HANDLING,
            temperature=0.4
        )
        
        return self._generate_response(prompt)
    
    def suggest_payment_plan(self, 
                           balance: float,
                           days_past_due: int,
                           debtor_profile: DebtorProfile,
                           context: Dict[str, Any]) -> LLMResponse:
        """Sugiere plan de pagos personalizado"""
        
        prompt_context = {
            "balance": balance,
            "days_past_due": days_past_due,
            "debtor_profile": debtor_profile.value,
            **context
        }
        
        prompt = LLMPrompt(
            system_prompt=self.prompt_templates[ResponseType.PAYMENT_PLAN].format(**prompt_context),
            user_prompt="Crea un plan de pagos atractivo y realista.",
            context=prompt_context,
            response_type=ResponseType.PAYMENT_PLAN,
            temperature=0.2,  # Muy baja creatividad para planes financieros
            max_tokens=200
        )
        
        return self._generate_response(prompt)
    
    def generate_escalation_script(self, 
                                 escalation_reason: str,
                                 context: Dict[str, Any]) -> LLMResponse:
        """Genera script de escalación"""
        
        prompt_context = {
            "escalation_reason": escalation_reason,
            **context
        }
        
        prompt = LLMPrompt(
            system_prompt=self.prompt_templates[ResponseType.ESCALATION_SCRIPT].format(**prompt_context),
            user_prompt="Genera un script de escalación profesional y efectivo.",
            context=prompt_context,
            response_type=ResponseType.ESCALATION_SCRIPT,
            temperature=0.2
        )
        
        return self._generate_response(prompt)
    
    def check_compliance(self, message: str) -> LLMResponse:
        """Verifica compliance de un mensaje"""
        
        prompt = LLMPrompt(
            system_prompt=self.prompt_templates[ResponseType.COMPLIANCE_CHECK].format(message=message),
            user_prompt="Analiza este mensaje para compliance.",
            context={"message": message},
            response_type=ResponseType.COMPLIANCE_CHECK,
            temperature=0.1,  # Mínima creatividad para compliance
            max_tokens=300
        )
        
        return self._generate_response(prompt)
    
    def _generate_response(self, prompt: LLMPrompt) -> LLMResponse:
        """Genera respuesta usando Ollama"""
        start_time = time.time()
        
        try:
            self.metrics["total_requests"] += 1
            
            # Preparar prompt completo
            full_prompt = f"{prompt.system_prompt}\n\nContexto: {json.dumps(prompt.context, ensure_ascii=False, indent=2)}\n\n{prompt.user_prompt}"
            
            # Preparar request para Ollama
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": prompt.temperature,
                    "num_predict": prompt.max_tokens,
                    "top_k": 10,
                    "top_p": 0.9
                }
            }
            
            # Hacer request
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Procesar respuesta
            generated_text = result.get('response', '').strip()
            
            if not generated_text:
                raise ValueError("LLM no generó respuesta")
            
            processing_time = time.time() - start_time
            
            # Estimar tokens (aproximado)
            tokens_used = len(generated_text.split()) + len(full_prompt.split())
            
            # Calcular confianza basado en coherencia y longitud
            confidence = self._calculate_response_confidence(generated_text, prompt)
            
            # Actualizar métricas
            self._update_metrics(processing_time, tokens_used, True)
            
            llm_response = LLMResponse(
                text=generated_text,
                confidence=confidence,
                response_type=prompt.response_type,
                processing_time=processing_time,
                tokens_used=tokens_used,
                metadata={
                    "model": self.model_name,
                    "temperature": prompt.temperature,
                    "prompt_length": len(full_prompt)
                }
            )
            
            logger.debug(f"LLM Response ({prompt.response_type.value}): {generated_text[:100]}...")
            return llm_response
            
        except Exception as e:
            logger.error(f"Error generando respuesta LLM: {e}")
            self.metrics["errors"] += 1
            
            # Respuesta de fallback
            return self._create_fallback_response(prompt.response_type, time.time() - start_time)
    
    def _calculate_response_confidence(self, text: str, prompt: LLMPrompt) -> float:
        """Calcula confianza de la respuesta generada"""
        confidence = 0.5  # Base
        
        # Factor 1: Longitud apropiada
        if 50 <= len(text) <= 300:
            confidence += 0.2
        elif 20 <= len(text) <= 500:
            confidence += 0.1
        
        # Factor 2: Estructura AEIOU (para respuestas AEIOU)
        if prompt.response_type == ResponseType.AEIOU_RESPONSE:
            aeiou_indicators = ['entiendo', 'reconozco', 'yo', 'propongo', 'objetivo', 'qué']
            found_indicators = sum(1 for indicator in aeiou_indicators if indicator.lower() in text.lower())
            confidence += min(0.3, found_indicators * 0.1)
        
        # Factor 3: Ausencia de texto repetitivo
        words = text.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio > 0.7:
            confidence += 0.1
        
        # Factor 4: Presencia de vocabulario de cobranza apropiado
        collections_terms = ['pago', 'plan', 'solución', 'ayuda', 'resolver', 'opciones']
        found_terms = sum(1 for term in collections_terms if term in text.lower())
        if found_terms > 0:
            confidence += min(0.2, found_terms * 0.05)
        
        return min(1.0, confidence)
    
    def _update_metrics(self, processing_time: float, tokens: int, success: bool):
        """Actualiza métricas del LLM"""
        if success:
            self.metrics["successful_requests"] += 1
        
        # Promedio móvil
        alpha = 0.1
        self.metrics["average_response_time"] = (
            (1 - alpha) * self.metrics["average_response_time"] + 
            alpha * processing_time
        )
        
        self.metrics["total_tokens"] += tokens
    
    def _create_fallback_response(self, response_type: ResponseType, processing_time: float) -> LLMResponse:
        """Crea respuesta de fallback cuando el LLM falla"""
        fallback_texts = {
            ResponseType.AEIOU_RESPONSE: "Entiendo tu situación (A). Yo quiero ayudarte a encontrar una solución (E). ¿Podríamos explorar algunas opciones de pago? (I) Mi objetivo es resolver esto de manera justa (O). ¿Qué te parece más conveniente? (U)",
            ResponseType.OBJECTION_HANDLING: "Comprendo tu preocupación. Vamos a trabajar juntos para encontrar una solución que funcione para ambos. ¿Puedes contarme más sobre tu situación actual?",
            ResponseType.PAYMENT_PLAN: "Podemos estructurar un plan de pagos cómodo que se ajuste a tu presupuesto. ¿Qué cantidad podrías manejar mensualmente?",
            ResponseType.ESCALATION_SCRIPT: "Entiendo que no hemos podido llegar a un acuerdo hoy. Voy a transferirte con mi supervisor quien tiene más opciones disponibles.",
            ResponseType.COMPLIANCE_CHECK: '{"compliant": true, "violations": [], "suggestions": [], "risk_level": "low"}'
        }
        
        return LLMResponse(
            text=fallback_texts.get(response_type, "Lo siento, permíteme reformular eso."),
            confidence=0.3,
            response_type=response_type,
            processing_time=processing_time,
            tokens_used=0,
            metadata={"fallback": True, "reason": "LLM_ERROR"}
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del LLM"""
        success_rate = 0.0
        if self.metrics["total_requests"] > 0:
            success_rate = self.metrics["successful_requests"] / self.metrics["total_requests"]
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "model_name": self.model_name,
            "average_tokens_per_request": (
                self.metrics["total_tokens"] / max(1, self.metrics["total_requests"])
            )
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Prueba la conexión y respuesta del LLM"""
        try:
            test_prompt = LLMPrompt(
                system_prompt="Responde brevemente que estás funcionando correctamente.",
                user_prompt="Test de conexión",
                context={},
                response_type=ResponseType.AEIOU_RESPONSE,
                max_tokens=50
            )
            
            response = self._generate_response(test_prompt)
            
            return {
                "success": True,
                "response_time": response.processing_time,
                "model": self.model_name,
                "response_text": response.text
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.model_name
            }


if __name__ == "__main__":
    # Test del LLM
    print("=== Test Collections LLM ===")
    
    try:
        llm = CollectionsLLM()
        
        # Test de conexión
        connection_test = llm.test_connection()
        if connection_test["success"]:
            print(f"✅ Conexión exitosa con {llm.model_name}")
            print(f"Respuesta de prueba: {connection_test['response_text']}")
        else:
            print(f"❌ Error de conexión: {connection_test['error']}")
            exit(1)
        
        # Test de generación AEIOU
        print("\n💬 Test de respuesta AEIOU...")
        
        from ..rag.collections_query_engine import ObjectionType, DebtorProfile
        
        # Crear RAG response simulado
        class MockRAGResponse:
            def __init__(self):
                self.objection_type = ObjectionType.CANNOT_PAY
                self.debtor_profile = DebtorProfile.FINANCIALLY_STRESSED
                self.suggested_response = "Entiendo que no tienes dinero ahora..."
                self.confidence_score = 0.85
                self.source_examples = [{}]
                self.compliance_score = 0.9
        
        mock_rag = MockRAGResponse()
        
        aeiou_response = llm.generate_aeiou_response(
            mock_rag,
            {"balance": 1250.00, "days_past_due": 45}
        )
        
        print(f"Respuesta AEIOU generada:")
        print(f"  Texto: {aeiou_response.text}")
        print(f"  Confianza: {aeiou_response.confidence:.2f}")
        print(f"  Tiempo: {aeiou_response.processing_time:.2f}s")
        
        # Test de manejo de objeciones
        print("\n🚫 Test de manejo de objeciones...")
        
        objection_response = llm.handle_objection(
            ObjectionType.ALREADY_PAID,
            DebtorProfile.COOPERATIVE,
            {"transcript": "Ya pagué esa deuda el mes pasado"}
        )
        
        print(f"Manejo de objeción:")
        print(f"  Texto: {objection_response.text}")
        print(f"  Confianza: {objection_response.confidence:.2f}")
        
        # Test de compliance
        print("\n⚖️ Test de compliance...")
        
        compliance_response = llm.check_compliance(
            "Te vamos a demandar si no pagas ahora mismo"
        )
        
        print(f"Análisis de compliance:")
        print(f"  Resultado: {compliance_response.text}")
        
        # Métricas finales
        print("\n📈 Métricas del LLM:")
        metrics = llm.get_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        print("Asegúrate de que Ollama esté ejecutándose: 'ollama serve'")
        print("Y que tengas el modelo instalado: 'ollama pull qwen2.5:0.5b'")
    
    print("🏁 Test completado")
