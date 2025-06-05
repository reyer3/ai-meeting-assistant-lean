#!/usr/bin/env python3
"""
Optimizaciones para Qwen2.5:0.5b en CPU para máxima velocidad
Enfoque: Respuestas AEIOU en <3 segundos sin GPU
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import httpx
from loguru import logger

@dataclass
class LLMOptimization:
    """Configuración optimizada para velocidad sin GPU"""
    model_name: str = "qwen2.5:0.5b"
    
    # Parámetros de generación optimizados
    temperature: float = 0.15  # Más determinístico = más rápido
    top_p: float = 0.85  # Reducir espacio de búsqueda
    top_k: int = 20  # Limitar candidatos
    max_tokens: int = 200  # Respuestas concisas
    
    # Optimizaciones de contexto
    max_context_tokens: int = 800  # Contexto reducido
    system_prompt_tokens: int = 150  # Prompt corto y específico
    
    # Performance settings
    num_predict: int = 200  # Máximo de tokens
    num_ctx: int = 1024  # Contexto reducido
    repeat_penalty: float = 1.1  # Evitar repetición
    
    # Timeout settings
    generation_timeout: float = 4.0  # 4s máximo
    connection_timeout: float = 1.0  # 1s timeout conexión


class OptimizedLocalLLM:
    """
    LLM local optimizado para generación rápida de sugerencias AEIOU
    """
    
    def __init__(self, config: LLMOptimization):
        self.config = config
        self.ollama_base_url = "http://localhost:11434"
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(config.connection_timeout))
        
        # Templates optimizados para AEIOU
        self.aeiou_template = self._build_aeiou_template()
        
        logger.info(f"🧠 OptimizedLocalLLM init: {config.model_name}")
    
    def _build_aeiou_template(self) -> str:
        """Template optimizado para respuestas AEIOU rápidas"""
        return """Eres un asistente de comunicación que genera respuestas AEIOU breves y efectivas.

AEIOU significa:
A-Acknowledge: Reconocer la perspectiva del otro
E-Express: Expresar tu perspectiva sin culpar  
I-Identify: Proponer soluciones específicas
O-Outcome: Definir el resultado deseado
U-Understanding: Buscar entendimiento mutuo

INSTRUCCIONES:
- Responde solo con la sugerencia AEIOU 
- Máximo 2-3 oraciones
- Usa lenguaje directo y profesional
- No expliques el framework

Situación: {others_statement}
Contexto: {context}

Sugerencia AEIOU:"""
    
    async def generate_aeiou_suggestion(self,
                                      others_statement: str,
                                      conversation_context: List[Dict] = None,
                                      rag_context: List[Dict] = None) -> Dict[str, Any]:
        """
        Genera sugerencia AEIOU optimizada para velocidad
        """
        start_time = time.time()
        
        try:
            # 1. Preparar contexto mínimo pero efectivo
            context = self._prepare_minimal_context(
                conversation_context or [],
                rag_context or []
            )
            
            # 2. Construir prompt optimizado
            prompt = self.aeiou_template.format(
                others_statement=others_statement,
                context=context
            )
            
            # 3. Llamada optimizada a Ollama
            response = await self._fast_ollama_call(prompt)
            
            # 4. Post-procesar respuesta
            suggestion = self._parse_aeiou_response(response)
            
            generation_time = time.time() - start_time
            
            return {
                "text": suggestion,
                "confidence": 0.85,  # Alta confianza por template estructurado
                "generation_time": generation_time,
                "model": self.config.model_name,
                "success": True
            }
            
        except asyncio.TimeoutError:
            logger.warning(f"⏰ LLM timeout después de {self.config.generation_timeout}s")
            return self._fallback_response(others_statement)
            
        except Exception as e:
            logger.error(f"❌ Error en generación LLM: {e}")
            return self._fallback_response(others_statement)
    
    def _prepare_minimal_context(self, 
                               conversation: List[Dict],
                               rag: List[Dict]) -> str:
        """
        Prepara contexto mínimo para velocidad máxima
        """
        context_parts = []
        
        # Solo últimos 3 turnos de conversación
        recent_turns = conversation[-3:] if conversation else []
        for turn in recent_turns:
            speaker = "Tú" if turn["speaker"] == "user" else "Otros"
            context_parts.append(f"{speaker}: {turn['text'][:100]}...")
        
        # Solo el ejemplo RAG más relevante
        if rag and len(rag) > 0:
            best_example = rag[0]
            context_parts.append(f"Ejemplo: {best_example['text'][:80]}...")
        
        return " | ".join(context_parts) if context_parts else "Sin contexto previo"
    
    async def _fast_ollama_call(self, prompt: str) -> str:
        """
        Llamada optimizada a Ollama para máxima velocidad
        """
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,  # Sin streaming para simplicidad
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "num_predict": self.config.num_predict,
                "num_ctx": self.config.num_ctx,
                "repeat_penalty": self.config.repeat_penalty,
                # Optimizaciones específicas CPU
                "num_thread": 4,  # Usar 4 cores
                "num_gpu": 0,  # Forzar CPU
                "low_vram": True,  # Optimización memoria
            }
        }
        
        try:
            response = await asyncio.wait_for(
                self.client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload
                ),
                timeout=self.config.generation_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                raise Exception(f"Ollama error: {response.status_code}")
                
        except asyncio.TimeoutError:
            logger.warning("⏰ Ollama timeout")
            raise
        except Exception as e:
            logger.error(f"❌ Ollama call failed: {e}")
            raise
    
    def _parse_aeiou_response(self, response: str) -> str:
        """
        Limpia y optimiza la respuesta AEIOU
        """
        if not response:
            return "Entiendo tu punto. ¿Podríamos explorar alternativas que funcionen para todos?"
        
        # Limpiar respuesta
        cleaned = response.strip()
        
        # Remover prefijos comunes
        prefixes_to_remove = [
            "Sugerencia AEIOU:",
            "Respuesta:",
            "Mi sugerencia:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Asegurar que no sea demasiado larga (máximo 300 chars)
        if len(cleaned) > 300:
            sentences = cleaned.split('. ')
            cleaned = '. '.join(sentences[:2])
            if not cleaned.endswith('.'):
                cleaned += '.'
        
        # Fallback si está vacía
        if len(cleaned) < 10:
            return "Entiendo tu perspectiva. ¿Podríamos buscar una solución que funcione para ambos?"
        
        return cleaned
    
    def _fallback_response(self, others_statement: str) -> Dict[str, Any]:
        """
        Respuesta de fallback cuando el LLM falla
        """
        fallback_suggestions = [
            "Entiendo tu punto. ¿Podríamos explorar alternativas que funcionen para todos?",
            "Aprecio tu perspectiva. ¿Qué opinas si revisamos otras opciones juntos?",
            "Veo tu preocupación. ¿Podríamos encontrar un enfoque que aborde tus puntos?"
        ]
        
        # Seleccionar basado en keywords
        statement_lower = others_statement.lower()
        
        if any(word in statement_lower for word in ['problema', 'error', 'mal']):
            suggestion = fallback_suggestions[2]  # Para problemas
        elif '?' in others_statement:
            suggestion = fallback_suggestions[1]  # Para preguntas
        else:
            suggestion = fallback_suggestions[0]  # General
        
        return {
            "text": suggestion,
            "confidence": 0.6,  # Menor confianza para fallback
            "generation_time": 0.1,
            "model": "fallback",
            "success": False,
            "fallback": True
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test rápido de conexión con Ollama
        """
        try:
            test_prompt = "Hola"
            response = await self._fast_ollama_call(test_prompt)
            
            return {
                "success": True,
                "model": self.config.model_name,
                "response_length": len(response),
                "message": "Conexión exitosa"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Error de conexión"
            }
    
    async def benchmark_speed(self, num_tests: int = 3) -> Dict[str, float]:
        """
        Benchmark de velocidad del modelo
        """
        logger.info(f"🏁 Benchmarking {self.config.model_name}...")
        
        test_statements = [
            "No estoy de acuerdo con este enfoque",
            "¿Cómo vamos a resolver este problema?",
            "Necesitamos más tiempo para el proyecto"
        ]
        
        times = []
        successes = 0
        
        for i in range(num_tests):
            statement = test_statements[i % len(test_statements)]
            
            start_time = time.time()
            result = await self.generate_aeiou_suggestion(statement)
            end_time = time.time()
            
            if result["success"]:
                times.append(end_time - start_time)
                successes += 1
                logger.info(f"Test {i+1}: {end_time - start_time:.2f}s - OK")
            else:
                logger.warning(f"Test {i+1}: FAILED")
        
        if times:
            return {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "success_rate": successes / num_tests,
                "total_tests": num_tests
            }
        else:
            return {"error": "Todos los tests fallaron"}
    
    async def cleanup(self):
        """Limpieza de recursos"""
        await self.client.aclose()


# Factory optimizado
def create_optimized_llm(model_name: str = "qwen2.5:0.5b") -> OptimizedLocalLLM:
    """
    Factory para crear LLM optimizado para velocidad
    """
    config = LLMOptimization(
        model_name=model_name,
        temperature=0.15,  # Muy determinístico
        top_p=0.85,
        top_k=20,
        max_tokens=200,  # Respuestas concisas
        max_context_tokens=800,
        generation_timeout=3.0,  # 3s máximo
        connection_timeout=1.0
    )
    
    return OptimizedLocalLLM(config)


if __name__ == "__main__":
    """Test del LLM optimizado"""
    
    async def test_optimized_llm():
        logger.info("🧪 Testing optimized LLM...")
        
        llm = create_optimized_llm()
        
        # Test conexión
        connection_test = await llm.test_connection()
        logger.info(f"🔗 Conexión: {connection_test}")
        
        if connection_test["success"]:
            # Benchmark velocidad
            benchmark = await llm.benchmark_speed(5)
            logger.info(f"🏁 Benchmark: {benchmark}")
            
            # Test ejemplo real
            result = await llm.generate_aeiou_suggestion(
                "No creo que este plan vaya a funcionar en el timeline propuesto"
            )
            
            logger.info(f"💡 Sugerencia ejemplo:")
            logger.info(f"   Texto: {result['text']}")
            logger.info(f"   Tiempo: {result['generation_time']:.2f}s")
            logger.info(f"   Confianza: {result['confidence']:.0%}")
        
        await llm.cleanup()
        logger.success("✅ Test completado")
    
    asyncio.run(test_optimized_llm())
