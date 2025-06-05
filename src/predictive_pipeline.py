#!/usr/bin/env python3
"""
Pipeline optimizado para sugerencias ANTES de que hables
Integra VAD mejorado para timing preciso
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from audio.system_capture import SystemAudioCapture, create_system_capture
from audio.lean_speaker_id import SpeakerIdentifier, create_lean_speaker_id  
from audio.enhanced_vad import EnhancedVAD, create_enhanced_vad
from ai.lean_stt import WhisperSTT, create_lean_stt
from ai.lean_llm import LocalLLM, create_lean_llm
from rag.query_engine import RAGQueryEngine

@dataclass
class OptimizedConfig:
    """Configuración optimizada para timing predictivo"""
    
    # Audio settings - optimizado para respuesta rápida
    sample_rate: int = 16000
    frame_duration: float = 0.03  # 30ms frames para detección precisa
    voice_threshold: float = 0.7  # Más estricto para evitar false positives
    
    # Model settings - balance velocidad/calidad
    model_name: str = "qwen2.5:0.5b"
    stt_model: str = "tiny"  # Más rápido para contexto de otros
    user_stt_model: str = "base"  # Mejor calidad para tu voz
    
    # Timing crítico
    suggestion_trigger_delay: float = 0.5  # Esperar 500ms antes de generar
    max_suggestion_time: float = 3.0  # Máximo 3s para generar sugerencia
    min_silence_for_suggestion: float = 0.8  # 800ms silencio = fin speech
    
    # Performance
    max_context_turns: int = 6  # Solo últimas 6 intervenciones
    rag_results: int = 1  # Un solo resultado para velocidad


class PredictiveMeetingAssistant:
    """
    Asistente con timing optimizado para sugerencias predictivas
    """
    
    def __init__(self, config: OptimizedConfig, user_profile_path: str):
        self.config = config
        self.user_profile_path = user_profile_path
        self.is_running = False
        
        # Core components
        self.audio_capture: Optional[SystemAudioCapture] = None
        self.speaker_id: Optional[SpeakerIdentifier] = None
        self.vad: Optional[EnhancedVAD] = None
        self.stt_fast: Optional[WhisperSTT] = None  # Para otros (rápido)
        self.stt_quality: Optional[WhisperSTT] = None  # Para ti (calidad)
        self.llm: Optional[LocalLLM] = None
        self.rag: Optional[RAGQueryEngine] = None
        
        # Estado de conversación
        self.conversation_turns: List[Dict[str, Any]] = []
        self.pending_suggestion_task: Optional[asyncio.Task] = None
        self.last_suggestion_time = 0
        
        logger.info("🚀 Predictive Meeting Assistant inicializado")
    
    async def initialize(self) -> bool:
        """Inicialización optimizada"""
        try:
            logger.info("⚡ Iniciando componentes optimizados...")
            
            # 1. Audio capture
            self.audio_capture = create_system_capture(
                sample_rate=self.config.sample_rate,
                buffer_duration=self.config.frame_duration
            )
            
            # 2. VAD para timing preciso
            self.vad = create_enhanced_vad(self.config.sample_rate)
            
            # 3. Speaker ID
            self.speaker_id = create_lean_speaker_id(self.user_profile_path)
            if not await self.speaker_id.load_user_profile():
                logger.error("❌ Perfil de voz no disponible")
                return False
            
            # 4. STT doble: rápido para otros, calidad para ti
            logger.info("🗣️ Inicializando STT dual...")
            self.stt_fast = create_lean_stt(self.config.stt_model, "es")
            self.stt_quality = create_lean_stt(self.config.user_stt_model, "es")
            
            if not await self.stt_fast.initialize():
                return False
            if not await self.stt_quality.initialize():
                return False
            
            # 5. LLM local
            self.llm = create_lean_llm(self.config.model_name)
            test_result = await self.llm.test_connection()
            if not test_result["success"]:
                logger.error(f"❌ LLM no disponible: {test_result['error']}")
                return False
            
            # 6. RAG (opcional)
            try:
                self.rag = RAGQueryEngine()
            except Exception as e:
                logger.warning(f"⚠️ RAG no disponible: {e}")
                self.rag = None
            
            logger.success("✅ Inicialización optimizada completa")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicialización: {e}")
            return False
    
    async def start_listening(self):
        """Loop principal optimizado"""
        if not await self.initialize():
            return
        
        self.is_running = True
        logger.info("👂 Iniciando escucha optimizada...")
        
        try:
            async for audio_frame in self.audio_capture.stream_audio():
                if not self.is_running:
                    break
                
                await self._process_audio_frame(audio_frame)
                
        except KeyboardInterrupt:
            logger.info("⏹️ Deteniendo asistente...")
        finally:
            await self.stop()
    
    async def _process_audio_frame(self, audio_frame: np.ndarray):
        """
        Procesamiento frame-by-frame con timing preciso
        """
        try:
            # 1. Identificar speaker
            speaker_info = await self.speaker_id.identify_speaker(audio_frame)
            
            # 2. VAD para detectar eventos de speech
            vad_result = self.vad.process_frame(audio_frame, speaker_info)
            
            # 3. Actuar según evento detectado
            if vad_result["speech_event"] == "speech_start":
                await self._handle_speech_start(vad_result)
                
            elif vad_result["speech_event"] == "speech_end":
                await self._handle_speech_end(vad_result)
                
            elif vad_result["should_generate_suggestion"]:
                # 🎯 TIMING CLAVE: Otros terminaron de hablar
                await self._trigger_predictive_suggestion()
                
        except Exception as e:
            logger.error(f"❌ Error procesando frame: {e}")
    
    async def _handle_speech_start(self, vad_result: Dict):
        """Maneja inicio de speech"""
        speaker_type = vad_result["speaker_type"]
        
        if speaker_type == "other":
            logger.debug("👥 Otros empiezan a hablar...")
            # Cancelar cualquier sugerencia pendiente
            if self.pending_suggestion_task:
                self.pending_suggestion_task.cancel()
                self.pending_suggestion_task = None
    
    async def _handle_speech_end(self, vad_result: Dict):
        """Maneja fin de speech"""
        speaker_type = vad_result["speaker_type"]
        
        if speaker_type == "user":
            logger.debug("🎤 Terminaste de hablar")
            # Tu speech terminó, agregar al contexto si fue transcrito
            
        elif speaker_type == "other":
            logger.debug("👥 Otros terminaron de hablar")
            # Aquí es donde necesitas la sugerencia ANTES de que hables
    
    async def _trigger_predictive_suggestion(self):
        """
        Trigger para generar sugerencia predictiva
        Se llama cuando otros terminan de hablar
        """
        current_time = time.time()
        
        # Control de frecuencia
        if current_time - self.last_suggestion_time < 5.0:  # Min 5s entre sugerencias
            return
        
        # Cancelar sugerencia anterior si existe
        if self.pending_suggestion_task:
            self.pending_suggestion_task.cancel()
        
        # Crear task para sugerencia con delay
        self.pending_suggestion_task = asyncio.create_task(
            self._generate_delayed_suggestion()
        )
    
    async def _generate_delayed_suggestion(self):
        """
        Genera sugerencia con delay para evitar interrupciones
        """
        try:
            # Pequeño delay para asegurar que terminaron de hablar
            await asyncio.sleep(self.config.suggestion_trigger_delay)
            
            # Verificar que no se reanudó el speech
            if self.vad.is_speech_active:
                logger.debug("🔄 Speech resumed, canceling suggestion")
                return
            
            # Obtener contexto de conversación reciente
            recent_context = self._get_recent_context()
            
            if not recent_context or len(recent_context) == 0:
                logger.debug("ℹ️ No hay contexto para sugerencia")
                return
            
            # 🎯 Generar sugerencia basada en lo que acaban de decir
            last_statement = recent_context[-1]
            if last_statement["speaker"] != "other":
                return
            
            await self._generate_contextual_suggestion(
                others_statement=last_statement["text"],
                context=recent_context[:-1]  # Contexto previo
            )
            
            self.last_suggestion_time = time.time()
            
        except asyncio.CancelledError:
            logger.debug("🔄 Sugerencia cancelada")
        except Exception as e:
            logger.error(f"❌ Error generando sugerencia: {e}")
    
    async def _generate_contextual_suggestion(self, 
                                           others_statement: str, 
                                           context: List[Dict]):
        """
        Genera sugerencia AEIOU contextual
        """
        try:
            logger.info(f"💡 Generando sugerencia para: '{others_statement[:50]}...'")
            
            # 1. Query RAG si disponible
            rag_context = []
            if self.rag:
                try:
                    rag_context = await self._quick_rag_query(others_statement)
                except Exception as e:
                    logger.warning(f"⚠️ RAG query failed: {e}")
            
            # 2. Generar sugerencia AEIOU
            suggestion = await self.llm.generate_response_suggestion(
                others_statement=others_statement,
                conversation_context=context,
                rag_context=rag_context,
                timeout=self.config.max_suggestion_time
            )
            
            # 3. Mostrar sugerencia
            await self._display_predictive_suggestion(
                suggestion=suggestion,
                trigger_statement=others_statement,
                rag_context=rag_context
            )
            
        except Exception as e:
            logger.error(f"❌ Error en sugerencia contextual: {e}")
    
    async def _quick_rag_query(self, query: str) -> List[Dict]:
        """Query RAG optimizado para velocidad"""
        if not self.rag:
            return []
        
        try:
            # Query simple y rápido
            results = self.rag.query(
                query=f"responder a: {query}",
                n_results=self.config.rag_results
            )
            
            context = []
            if results and 'documents' in results:
                for doc in results['documents'][0]:
                    context.append({
                        'text': doc,
                        'relevance': 0.8  # Default relevance
                    })
            
            return context
            
        except Exception as e:
            logger.warning(f"⚠️ Quick RAG query failed: {e}")
            return []
    
    async def _display_predictive_suggestion(self, 
                                          suggestion, 
                                          trigger_statement: str,
                                          rag_context: List[Dict]):
        """
        Muestra sugerencia predictiva
        """
        print("\\n" + "🔮" * 80)
        print("💭 SUGERENCIA PREDICTIVA:")
        print(f"📞 Responder a: \"{trigger_statement}\"")
        print("🔮" * 80)
        print(f"💡 {suggestion.text}")
        
        if rag_context:
            print(f"📚 Basado en {len(rag_context)} ejemplos similares")
        
        print("⏰ Úsa esta sugerencia cuando hables...")
        print("🔮" * 80 + "\\n")
    
    def _get_recent_context(self) -> List[Dict[str, Any]]:
        """Obtiene contexto reciente de conversación"""
        return self.conversation_turns[-self.config.max_context_turns:]
    
    def _add_to_context(self, text: str, speaker: str):
        """Agrega al contexto de conversación"""
        self.conversation_turns.append({
            "text": text,
            "speaker": speaker,
            "timestamp": time.time()
        })
        
        # Limitar tamaño
        if len(self.conversation_turns) > self.config.max_context_turns * 2:
            self.conversation_turns = self.conversation_turns[-self.config.max_context_turns:]
    
    async def stop(self):
        """Cleanup resources"""
        self.is_running = False
        
        if self.pending_suggestion_task:
            self.pending_suggestion_task.cancel()
        
        if self.audio_capture:
            await self.audio_capture.stop()
        
        logger.info("🛑 Predictive assistant stopped")


# Factory optimizado
def create_predictive_assistant(user_profile_path: str = "data/user_voice_profile.pkl") -> PredictiveMeetingAssistant:
    """Crea asistente con configuración predictiva optimizada"""
    
    config = OptimizedConfig(
        sample_rate=16000,
        frame_duration=0.03,  # 30ms frames
        voice_threshold=0.7,  # Más estricto
        model_name="qwen2.5:0.5b",
        stt_model="tiny",  # Rápido para otros
        user_stt_model="base",  # Calidad para ti
        suggestion_trigger_delay=0.5,  # 500ms delay
        max_suggestion_time=3.0,  # 3s máximo
        min_silence_for_suggestion=0.8,  # 800ms silencio
        max_context_turns=6,
        rag_results=1  # Solo 1 resultado RAG
    )
    
    return PredictiveMeetingAssistant(config, user_profile_path)


if __name__ == "__main__":
    """Test del asistente predictivo"""
    
    async def main():
        assistant = create_predictive_assistant()
        await assistant.start_listening()
    
    asyncio.run(main())
