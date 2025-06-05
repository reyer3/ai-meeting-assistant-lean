#!/usr/bin/env python3
"""
Pipeline integrado final - Asistente lean con todas las optimizaciones
Combina: VAD preciso + LLM optimizado + Overlay UI + Timing predictivo
"""

import asyncio
import time
import signal
import sys
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

# Imports optimizados
from audio.system_capture import SystemAudioCapture, create_system_capture
from audio.lean_speaker_id import SpeakerIdentifier, create_lean_speaker_id
from audio.enhanced_vad import EnhancedVAD, create_enhanced_vad
from ai.lean_stt import WhisperSTT, create_lean_stt
from ai.optimized_llm import OptimizedLocalLLM, create_optimized_llm
from rag.query_engine import RAGQueryEngine
from ui.suggestion_overlay import OverlayManager, create_lean_overlay

@dataclass
class UltimateConfig:
    """Configuración final optimizada para máximo rendimiento"""
    
    # Audio pipeline
    sample_rate: int = 16000
    frame_duration: float = 0.03  # 30ms para detección precisa
    voice_confidence_threshold: float = 0.75  # Alto para evitar false positives
    
    # AI models
    model_name: str = "qwen2.5:0.5b"  # Modelo más rápido
    stt_model_others: str = "tiny"     # STT rápido para otros
    stt_model_user: str = "base"       # STT de calidad para ti
    
    # Timing crítico
    suggestion_delay: float = 0.6      # 600ms después que otros terminan
    max_generation_time: float = 2.5   # 2.5s máximo para sugerencias
    min_interval_between: float = 8.0  # 8s mínimo entre sugerencias
    
    # Performance optimizations
    max_context_history: int = 8       # Solo últimas 8 intervenciones
    rag_max_results: int = 1           # Un solo resultado RAG
    enable_ui_overlay: bool = True     # Usar overlay UI
    
    # System paths
    user_profile_path: str = "data/user_voice_profile.pkl"
    log_level: str = "INFO"


class UltimateLeanAssistant:
    """
    Asistente final con todas las optimizaciones integradas
    """
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        self.is_running = False
        self.startup_time = time.time()
        
        # Core components
        self.audio_capture: Optional[SystemAudioCapture] = None
        self.speaker_id: Optional[SpeakerIdentifier] = None
        self.vad: Optional[EnhancedVAD] = None
        self.stt_fast: Optional[WhisperSTT] = None      # Para otros
        self.stt_quality: Optional[WhisperSTT] = None   # Para ti
        self.llm: Optional[OptimizedLocalLLM] = None
        self.rag: Optional[RAGQueryEngine] = None
        self.overlay: Optional[OverlayManager] = None
        
        # Estado de conversación optimizado
        self.conversation_history: List[Dict[str, Any]] = []
        self.last_suggestion_time = 0
        self.pending_suggestion_task: Optional[asyncio.Task] = None
        self.current_audio_buffer = np.array([])
        
        # Métricas de performance
        self.stats = {
            "suggestions_generated": 0,
            "avg_generation_time": 0,
            "voice_identifications": 0,
            "false_positives": 0
        }
        
        logger.info("🚀 Ultimate Lean Assistant inicializado")
    
    async def initialize(self) -> bool:
        """Inicialización ultra-optimizada"""
        init_start = time.time()
        
        try:
            logger.info("⚡ Iniciando componentes ultra-optimizados...")
            
            # 1. Audio capture con buffer optimizado
            logger.debug("🎤 Configurando captura de audio...")
            self.audio_capture = create_system_capture(
                sample_rate=self.config.sample_rate,
                buffer_duration=self.config.frame_duration
            )
            
            # 2. VAD para timing preciso
            logger.debug("📊 Inicializando VAD optimizado...")
            self.vad = create_enhanced_vad(self.config.sample_rate)
            
            # 3. Speaker identification
            logger.debug("👤 Cargando perfil de voz...")
            self.speaker_id = create_lean_speaker_id(self.config.user_profile_path)
            if not await self.speaker_id.load_user_profile():
                logger.error("❌ Perfil de voz requerido")
                logger.info("💡 Ejecuta: python scripts/setup_voice_profile.py")
                return False
            
            # 4. STT dual (rápido vs calidad)
            logger.debug("🗣️ Inicializando STT dual...")
            self.stt_fast = create_lean_stt(self.config.stt_model_others, "es")
            self.stt_quality = create_lean_stt(self.config.stt_model_user, "es")
            
            # Inicializar en paralelo para velocidad
            stt_tasks = [
                self.stt_fast.initialize(),
                self.stt_quality.initialize()
            ]
            stt_results = await asyncio.gather(*stt_tasks, return_exceptions=True)
            
            if not all(result is True for result in stt_results):
                logger.error("❌ Error inicializando STT")
                return False
            
            # 5. LLM optimizado
            logger.debug("🧠 Conectando LLM optimizado...")
            self.llm = create_optimized_llm(self.config.model_name)
            
            test_result = await self.llm.test_connection()
            if not test_result["success"]:
                logger.error(f"❌ LLM no disponible: {test_result.get('error')}")
                return False
            
            # 6. RAG engine (opcional)
            logger.debug("📚 Inicializando RAG...")
            try:
                self.rag = RAGQueryEngine()
            except Exception as e:
                logger.warning(f"⚠️ RAG no disponible: {e}")
                self.rag = None
            
            # 7. UI Overlay (opcional)
            if self.config.enable_ui_overlay:
                logger.debug("🖼️ Inicializando overlay UI...")
                try:
                    self.overlay = create_lean_overlay()
                    self.overlay.start()
                    # Dar tiempo para que se inicialice
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning(f"⚠️ Overlay UI no disponible: {e}")
                    self.overlay = None
            
            init_time = time.time() - init_start
            logger.success(f"✅ Inicialización completa en {init_time:.2f}s")
            
            # Benchmark inicial del LLM
            if self.llm:
                logger.info("🏁 Ejecutando benchmark inicial...")
                benchmark = await self.llm.benchmark_speed(3)
                if "avg_time" in benchmark:
                    logger.info(f"📊 LLM benchmark: {benchmark['avg_time']:.2f}s promedio")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en inicialización: {e}")
            return False
    
    async def start_ultimate_listening(self):
        """Loop principal ultra-optimizado"""
        if not await self.initialize():
            return
        
        # Setup signal handlers para graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.is_running = True
        runtime_start = time.time()
        
        logger.success("👂 Asistente lean iniciado - Escuchando audio del sistema...")
        logger.info("🎯 Listo para generar sugerencias AEIOU cuando otros hablen")
        
        try:
            frame_count = 0
            async for audio_frame in self.audio_capture.stream_audio():
                if not self.is_running:
                    break
                
                frame_count += 1
                await self._process_optimized_frame(audio_frame, frame_count)
                
        except KeyboardInterrupt:
            logger.info("⏹️ Shutdown solicitado...")
        except Exception as e:
            logger.error(f"❌ Error en loop principal: {e}")
        finally:
            runtime = time.time() - runtime_start
            logger.info(f"⏱️ Tiempo de ejecución: {runtime:.1f}s")
            await self._graceful_shutdown()
    
    async def _process_optimized_frame(self, audio_frame: np.ndarray, frame_num: int):
        """
        Procesamiento ultra-optimizado frame por frame
        """
        try:
            # 1. Speaker identification rápida
            speaker_info = await self.speaker_id.identify_speaker(audio_frame)
            
            # 2. VAD para detección de eventos
            vad_result = self.vad.process_frame(audio_frame, speaker_info)
            
            # 3. Solo procesar si hay speech significativo
            if vad_result["frame_energy"] < 0.005:  # Skip frames muy silenciosos
                return
            
            # 4. Logging optimizado (solo cada 100 frames para no spamear)
            if frame_num % 100 == 0:
                logger.debug(f"Frame {frame_num}: speaker={vad_result.get('speaker_type', 'none')}, event={vad_result['speech_event']}")
            
            # 5. Manejar eventos de speech
            if vad_result["speech_event"] == "speech_end":
                await self._handle_speech_ended(vad_result, audio_frame)
            
            elif vad_result["should_generate_suggestion"]:
                # 🎯 MOMENTO CRÍTICO: Generar sugerencia predictiva
                await self._trigger_ultra_fast_suggestion()
            
        except Exception as e:
            logger.error(f"❌ Error en frame {frame_num}: {e}")
    
    async def _handle_speech_ended(self, vad_result: Dict, audio_frame: np.ndarray):
        """
        Maneja fin de speech con transcripción inteligente
        """
        speaker_type = vad_result.get("current_speaker")
        
        if not speaker_type:
            return
        
        try:
            # Transcribir según el speaker
            if speaker_type == "user":
                # Tu voz: STT de calidad
                transcription = await self.stt_quality.transcribe(audio_frame)
            else:
                # Otros: STT rápido
                transcription = await self.stt_fast.transcribe(audio_frame)
            
            if transcription and len(transcription.strip()) > 5:
                # Agregar al contexto
                self._add_to_optimized_context(transcription, speaker_type)
                
                logger.debug(f"📝 {speaker_type}: {transcription[:50]}...")
                
        except Exception as e:
            logger.error(f"❌ Error en transcripción: {e}")
    
    async def _trigger_ultra_fast_suggestion(self):
        """
        Trigger ultra-rápido para sugerencias predictivas
        """
        current_time = time.time()
        
        # Control de frecuencia
        if current_time - self.last_suggestion_time < self.config.min_interval_between:
            return
        
        # Cancelar tarea previa
        if self.pending_suggestion_task and not self.pending_suggestion_task.done():
            self.pending_suggestion_task.cancel()
        
        # Nueva tarea de sugerencia
        self.pending_suggestion_task = asyncio.create_task(
            self._generate_ultra_fast_suggestion()
        )
    
    async def _generate_ultra_fast_suggestion(self):
        """
        Generación ultra-rápida de sugerencias
        """
        generation_start = time.time()
        
        try:
            # Pequeño delay para estabilidad
            await asyncio.sleep(self.config.suggestion_delay)
            
            # Obtener contexto reciente
            recent_context = self._get_optimized_context()
            
            if not recent_context:
                logger.debug("ℹ️ Sin contexto para sugerencia")
                return
            
            # Última intervención de otros
            others_statements = [
                turn for turn in recent_context 
                if turn["speaker"] == "other"
            ]
            
            if not others_statements:
                return
            
            last_statement = others_statements[-1]["text"]
            conversation_context = recent_context[:-1]  # Contexto previo
            
            # Query RAG ultra-rápido si está disponible
            rag_context = []
            if self.rag:
                try:
                    rag_context = await asyncio.wait_for(
                        self._ultra_fast_rag_query(last_statement),
                        timeout=0.5  # 500ms máximo para RAG
                    )
                except asyncio.TimeoutError:
                    logger.debug("⚠️ RAG timeout")
            
            # Generar sugerencia con timeout estricto
            suggestion = await asyncio.wait_for(
                self.llm.generate_aeiou_suggestion(
                    others_statement=last_statement,
                    conversation_context=conversation_context,
                    rag_context=rag_context
                ),
                timeout=self.config.max_generation_time
            )
            
            # Mostrar sugerencia
            await self._display_ultra_fast_suggestion(
                suggestion=suggestion,
                trigger_statement=last_statement,
                rag_context=rag_context
            )
            
            # Update stats
            generation_time = time.time() - generation_start
            self._update_performance_stats(generation_time)
            self.last_suggestion_time = time.time()
            
            logger.info(f"💡 Sugerencia generada en {generation_time:.2f}s")
            
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout generando sugerencia")
        except asyncio.CancelledError:
            logger.debug("🔄 Generación de sugerencia cancelada")
        except Exception as e:
            logger.error(f"❌ Error en generación ultra-rápida: {e}")
    
    async def _ultra_fast_rag_query(self, query: str) -> List[Dict]:
        """Query RAG ultra-optimizado"""
        try:
            results = self.rag.query(
                query=f"responder: {query[:100]}",  # Query truncado
                n_results=self.config.rag_max_results
            )
            
            if results and 'documents' in results:
                return [{
                    'text': doc,
                    'relevance': 0.8
                } for doc in results['documents'][0][:1]]  # Solo 1 resultado
            
        except Exception as e:
            logger.debug(f"⚠️ RAG query error: {e}")
        
        return []
    
    async def _display_ultra_fast_suggestion(self, 
                                           suggestion: Dict,
                                           trigger_statement: str,
                                           rag_context: List[Dict]):
        """
        Display ultra-optimizado de sugerencias
        """
        suggestion_text = suggestion.get("text", "")
        confidence = suggestion.get("confidence", 0.8)
        
        # Context info para UI
        context_info = ""
        if rag_context:
            context_info = f"Basado en {len(rag_context)} ejemplos"
        
        # Mostrar en overlay si está disponible
        if self.overlay:
            try:
                self.overlay.show_suggestion(
                    text=suggestion_text,
                    context=context_info,
                    confidence=confidence
                )
            except Exception as e:
                logger.warning(f"⚠️ Error en overlay: {e}")
                # Fallback a consola
                self._display_console_suggestion(suggestion_text, trigger_statement, confidence)
        else:
            # Fallback a consola
            self._display_console_suggestion(suggestion_text, trigger_statement, confidence)
    
    def _display_console_suggestion(self, suggestion: str, trigger: str, confidence: float):
        """Fallback display en consola"""
        print("\\n" + "🔮" * 60)
        print(f"💭 SUGERENCIA (confianza: {confidence:.0%}):")
        print(f"📞 En respuesta a: \"{trigger[:60]}...\"")
        print("🔮" * 60)
        print(f"💡 {suggestion}")
        print("🔮" * 60 + "\\n")
    
    def _add_to_optimized_context(self, text: str, speaker: str):
        """Agregar al contexto optimizado"""
        self.conversation_history.append({
            "text": text,
            "speaker": speaker,
            "timestamp": time.time()
        })
        
        # Limitar tamaño para performance
        if len(self.conversation_history) > self.config.max_context_history * 2:
            self.conversation_history = self.conversation_history[-self.config.max_context_history:]
    
    def _get_optimized_context(self) -> List[Dict[str, Any]]:
        """Obtener contexto optimizado"""
        return self.conversation_history[-self.config.max_context_history:]
    
    def _update_performance_stats(self, generation_time: float):
        """Actualizar métricas de performance"""
        self.stats["suggestions_generated"] += 1
        
        # Moving average de tiempo de generación
        if self.stats["avg_generation_time"] == 0:
            self.stats["avg_generation_time"] = generation_time
        else:
            # Exponential moving average
            alpha = 0.3
            self.stats["avg_generation_time"] = (
                alpha * generation_time + 
                (1 - alpha) * self.stats["avg_generation_time"]
            )
    
    def _signal_handler(self, signum, frame):
        """Handler para shutdown graceful"""
        logger.info(f"🛑 Signal {signum} recibido")
        self.is_running = False
    
    async def _graceful_shutdown(self):
        """Shutdown graceful de todos los componentes"""
        logger.info("🛑 Iniciando shutdown...")
        
        # Cancelar tareas pendientes
        if self.pending_suggestion_task:
            self.pending_suggestion_task.cancel()
        
        # Cleanup components
        cleanup_tasks = []
        
        if self.audio_capture:
            cleanup_tasks.append(self.audio_capture.stop())
        
        if self.stt_fast:
            cleanup_tasks.append(self.stt_fast.cleanup())
            
        if self.stt_quality:
            cleanup_tasks.append(self.stt_quality.cleanup())
        
        if self.llm:
            cleanup_tasks.append(self.llm.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Cleanup UI
        if self.overlay:
            self.overlay.stop()
        
        # Performance report
        self._print_performance_report()
        
        logger.success("✅ Shutdown completo")
    
    def _print_performance_report(self):
        """Reporte de performance"""
        runtime = time.time() - self.startup_time
        
        print("\\n" + "📊" * 50)
        print("REPORTE DE PERFORMANCE")
        print("📊" * 50)
        print(f"⏱️  Tiempo total: {runtime:.1f}s")
        print(f"💡 Sugerencias generadas: {self.stats['suggestions_generated']}")
        print(f"⚡ Tiempo promedio generación: {self.stats['avg_generation_time']:.2f}s")
        print(f"🎯 Tasa de sugerencias: {self.stats['suggestions_generated']/max(runtime/60, 1):.1f}/min")
        print("📊" * 50 + "\\n")


# Factory principal
def create_ultimate_assistant(config_override: Dict[str, Any] = None) -> UltimateLeanAssistant:
    """
    Factory para crear asistente con configuración ultra-optimizada
    """
    base_config = UltimateConfig()
    
    # Apply overrides
    if config_override:
        for key, value in config_override.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
    
    return UltimateLeanAssistant(base_config)


# Entry point principal
async def main():
    """Entry point principal del asistente"""
    
    # Configuración por defecto (puedes modificar aquí)
    config_overrides = {
        # "model_name": "qwen2.5:1.5b",  # Modelo más grande si tienes RAM
        # "enable_ui_overlay": False,    # Desactivar UI si hay problemas
        # "voice_confidence_threshold": 0.65,  # Menos estricto si falla detección
    }
    
    assistant = create_ultimate_assistant(config_overrides)
    await assistant.start_ultimate_listening()


if __name__ == "__main__":
    """
    Ejecutar asistente ultra-optimizado
    
    Uso:
    python src/ultimate_assistant.py
    """
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n👋 ¡Hasta luego!")
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        sys.exit(1)
