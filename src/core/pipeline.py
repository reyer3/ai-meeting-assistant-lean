"""
Pipeline principal del AI Meeting Assistant Lean

Este módulo orquesta todos los componentes del sistema:
1. Captura de audio
2. Identificación de speaker
3. Speech-to-text
4. Consulta RAG
5. Generación de sugerencias AEIOU
6. Display en overlay
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from threading import Thread, Event

from config import config
# Imports que se implementarán en los siguientes archivos
# from ..audio.capture import AudioCapture
# from ..audio.speaker_id import SpeakerIdentifier  
# from ..rag.chroma_manager import ChromaManager
# from ..rag.query_engine import QueryEngine, ConversationContext
# from ..ai.stt import SpeechToText
# from ..ai.llm import LLMGenerator
# from ..ui.overlay import OverlayManager

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Estados del pipeline"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SUGGESTING = "suggesting"
    ERROR = "error"


@dataclass
class AudioChunk:
    """Chunk de audio procesado"""
    data: bytes
    timestamp: float
    duration: float
    speaker: str  # "USER", "OTHER", "UNKNOWN"
    confidence: float


@dataclass
class ProcessedTranscript:
    """Transcripción procesada"""
    text: str
    speaker: str
    timestamp: float
    confidence: float
    language: str


@dataclass
class Suggestion:
    """Sugerencia generada"""
    text: str
    confidence: float
    source_type: str  # "rag", "fallback", "template"
    context: Dict[str, Any]
    timestamp: float


class MeetingAssistantPipeline:
    """Pipeline principal del asistente de reuniones"""
    
    def __init__(self):
        self.state = PipelineState.IDLE
        self.is_running = False
        self.stop_event = Event()
        
        # Queues para comunicación entre componentes
        self.audio_queue = Queue(maxsize=10)
        self.transcript_queue = Queue(maxsize=5)
        self.suggestion_queue = Queue(maxsize=3)
        
        # Componentes del sistema (se inicializarán en setup)
        self.audio_capture = None
        self.speaker_identifier = None
        self.stt_engine = None
        self.rag_engine = None
        self.llm_generator = None
        self.overlay_manager = None
        
        # Contexto de conversación
        self.conversation_context = []
        self.last_user_speech_time = 0
        self.current_meeting_context = None
        
        # Métricas de performance
        self.metrics = {
            "audio_chunks_processed": 0,
            "transcriptions_made": 0,
            "suggestions_generated": 0,
            "average_latency": 0.0,
            "errors": 0
        }
        
        # Callbacks para eventos
        self.callbacks = {
            "on_transcript": [],
            "on_suggestion": [],
            "on_error": [],
            "on_state_change": []
        }
    
    async def setup(self):
        """Inicializa todos los componentes del sistema"""
        logger.info("🚀 Inicializando AI Meeting Assistant Pipeline...")
        
        try:
            # 1. Inicializar ChromaDB y RAG
            logger.info("📊 Inicializando sistema RAG...")
            # self.chroma_manager = ChromaManager(
            #     data_dir=config.rag.chroma_db_path,
            #     model_name=config.rag.embedding_model
            # )
            # self.rag_engine = QueryEngine(self.chroma_manager)
            
            # 2. Inicializar captura de audio
            logger.info("🎤 Inicializando captura de audio...")
            # self.audio_capture = AudioCapture(
            #     sample_rate=config.audio.sample_rate,
            #     chunk_size=config.audio.chunk_size,
            #     channels=config.audio.channels
            # )
            
            # 3. Inicializar identificación de speaker
            logger.info("🔍 Inicializando identificación de voz...")
            # self.speaker_identifier = SpeakerIdentifier(
            #     profile_path=config.voice_profile.profile_path,
            #     threshold=config.voice_profile.similarity_threshold
            # )
            
            # 4. Inicializar STT
            logger.info("📝 Inicializando Speech-to-Text...")
            # self.stt_engine = SpeechToText(
            #     model_size=config.stt.model_size,
            #     language=config.stt.language
            # )
            
            # 5. Inicializar LLM
            logger.info("🧠 Inicializando modelo de lenguaje...")
            # self.llm_generator = LLMGenerator(
            #     model_name=config.llm.model_name,
            #     temperature=config.llm.temperature
            # )
            
            # 6. Inicializar UI overlay
            logger.info("🖥️ Inicializando interfaz de usuario...")
            # self.overlay_manager = OverlayManager(
            #     position=config.ui.overlay_position,
            #     opacity=config.ui.overlay_opacity
            # )
            
            logger.info("✅ Pipeline inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando pipeline: {e}")
            self._set_state(PipelineState.ERROR)
            raise
    
    def add_callback(self, event_type: str, callback: Callable):
        """Agrega callback para eventos del pipeline"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def _emit_event(self, event_type: str, data: Any):
        """Emite evento a todos los callbacks suscritos"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error en callback {event_type}: {e}")
    
    def _set_state(self, new_state: PipelineState):
        """Cambia el estado del pipeline y emite evento"""
        old_state = self.state
        self.state = new_state
        logger.debug(f"Estado: {old_state.value} → {new_state.value}")
        self._emit_event("on_state_change", {"old": old_state, "new": new_state})
    
    async def start(self):
        """Inicia el pipeline de procesamiento"""
        if self.is_running:
            logger.warning("Pipeline ya está ejecutándose")
            return
        
        logger.info("🎯 Iniciando pipeline de procesamiento...")
        self.is_running = True
        self.stop_event.clear()
        self._set_state(PipelineState.LISTENING)
        
        # Iniciar threads de procesamiento
        threads = [
            Thread(target=self._audio_processing_loop, daemon=True),
            Thread(target=self._transcript_processing_loop, daemon=True),
            Thread(target=self._suggestion_processing_loop, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        # Iniciar captura de audio
        # await self.audio_capture.start_capture(self.audio_queue)
        
        logger.info("✅ Pipeline iniciado - escuchando audio...")
    
    def stop(self):
        """Detiene el pipeline"""
        logger.info("🛑 Deteniendo pipeline...")
        self.is_running = False
        self.stop_event.set()
        self._set_state(PipelineState.IDLE)
        
        # Detener captura de audio
        # if self.audio_capture:
        #     self.audio_capture.stop_capture()
        
        logger.info("✅ Pipeline detenido")
    
    def _audio_processing_loop(self):
        """Loop de procesamiento de audio (identificación de speaker)"""
        logger.info("🎤 Iniciando loop de procesamiento de audio")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Obtener chunk de audio con timeout
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get(timeout=1.0)
                    
                    start_time = time.time()
                    
                    # Identificar speaker
                    # speaker_result = self.speaker_identifier.identify_speaker(audio_data)
                    
                    # Simulación para desarrollo
                    speaker_result = {
                        "speaker": "USER" if time.time() % 2 < 1 else "OTHER",
                        "confidence": 0.85
                    }
                    
                    audio_chunk = AudioChunk(
                        data=audio_data,
                        timestamp=time.time(),
                        duration=len(audio_data) / config.audio.sample_rate,
                        speaker=speaker_result["speaker"],
                        confidence=speaker_result["confidence"]
                    )
                    
                    # Solo procesar si es el usuario hablando con suficiente confianza
                    if (audio_chunk.speaker == "USER" and 
                        audio_chunk.confidence > config.voice_profile.similarity_threshold):
                        
                        self.transcript_queue.put(audio_chunk)
                        self.last_user_speech_time = time.time()
                    
                    # Actualizar métricas
                    self.metrics["audio_chunks_processed"] += 1
                    processing_time = time.time() - start_time
                    self._update_latency_metric(processing_time)
                    
                else:
                    time.sleep(0.1)  # Evitar busy waiting
                    
            except Exception as e:
                logger.error(f"Error en procesamiento de audio: {e}")
                self.metrics["errors"] += 1
    
    def _transcript_processing_loop(self):
        """Loop de procesamiento de transcripción (STT)"""
        logger.info("📝 Iniciando loop de transcripción")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                if not self.transcript_queue.empty():
                    audio_chunk = self.transcript_queue.get(timeout=1.0)
                    
                    start_time = time.time()
                    
                    # Transcribir audio
                    # transcript_result = self.stt_engine.transcribe(audio_chunk.data)
                    
                    # Simulación para desarrollo
                    transcript_result = {
                        "text": "Esta es una transcripción de prueba del usuario",
                        "confidence": 0.89,
                        "language": "es"
                    }
                    
                    processed_transcript = ProcessedTranscript(
                        text=transcript_result["text"],
                        speaker=audio_chunk.speaker,
                        timestamp=audio_chunk.timestamp,
                        confidence=transcript_result["confidence"],
                        language=transcript_result["language"]
                    )
                    
                    # Agregar al contexto de conversación
                    self.conversation_context.append(processed_transcript)
                    
                    # Mantener solo las últimas 10 intervenciones
                    if len(self.conversation_context) > 10:
                        self.conversation_context.pop(0)
                    
                    # Emitir evento de transcripción
                    self._emit_event("on_transcript", processed_transcript)
                    
                    # Enviar para procesamiento de sugerencias
                    self.suggestion_queue.put(processed_transcript)
                    
                    # Actualizar métricas
                    self.metrics["transcriptions_made"] += 1
                    processing_time = time.time() - start_time
                    self._update_latency_metric(processing_time)
                    
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error en transcripción: {e}")
                self.metrics["errors"] += 1
    
    def _suggestion_processing_loop(self):
        """Loop de procesamiento de sugerencias (RAG + LLM)"""
        logger.info("💡 Iniciando loop de sugerencias")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                if not self.suggestion_queue.empty():
                    transcript = self.suggestion_queue.get(timeout=1.0)
                    
                    start_time = time.time()
                    self._set_state(PipelineState.PROCESSING)
                    
                    # Construir contexto para RAG
                    conversation_text = " ".join([
                        t.text for t in self.conversation_context[-3:]  # Últimas 3 intervenciones
                    ])
                    
                    # context = ConversationContext(
                    #     current_speaker=transcript.speaker,
                    #     recent_transcript=conversation_text,
                    #     meeting_type=self.current_meeting_context
                    # )
                    
                    # Consultar RAG
                    # rag_response = self.rag_engine.get_contextual_suggestions(context)
                    
                    # Simulación de respuesta RAG
                    rag_response = {
                        "suggested_response": "Entiendo tu perspectiva (A). Yo valoro tu punto de vista (E). ¿Podríamos explorar esto juntos? (I) Mi objetivo es que lleguemos a un entendimiento mutuo (O). ¿Qué sería más útil para ti? (U)",
                        "confidence_score": 0.85,
                        "conflict_type": "communication_breakdown",
                        "reasoning": "Basado en patrón de comunicación similar"
                    }
                    
                    # Crear sugerencia
                    suggestion = Suggestion(
                        text=rag_response["suggested_response"],
                        confidence=rag_response["confidence_score"],
                        source_type="rag",
                        context={
                            "conflict_type": rag_response["conflict_type"],
                            "reasoning": rag_response["reasoning"],
                            "transcript": transcript.text
                        },
                        timestamp=time.time()
                    )
                    
                    # Solo mostrar sugerencias con confianza suficiente
                    if suggestion.confidence > 0.7:
                        self._set_state(PipelineState.SUGGESTING)
                        
                        # Mostrar en overlay
                        # self.overlay_manager.show_suggestion(suggestion)
                        
                        # Emitir evento
                        self._emit_event("on_suggestion", suggestion)
                        
                        self.metrics["suggestions_generated"] += 1
                    
                    processing_time = time.time() - start_time
                    self._update_latency_metric(processing_time)
                    
                    self._set_state(PipelineState.LISTENING)
                    
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error generando sugerencias: {e}")
                self.metrics["errors"] += 1
                self._set_state(PipelineState.LISTENING)
    
    def _update_latency_metric(self, processing_time: float):
        """Actualiza la métrica de latencia promedio"""
        current_avg = self.metrics["average_latency"]
        total_operations = (self.metrics["audio_chunks_processed"] + 
                          self.metrics["transcriptions_made"] + 
                          self.metrics["suggestions_generated"])
        
        if total_operations > 0:
            self.metrics["average_latency"] = (
                (current_avg * (total_operations - 1) + processing_time) / total_operations
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas actuales del pipeline"""
        return {
            **self.metrics,
            "state": self.state.value,
            "is_running": self.is_running,
            "conversation_length": len(self.conversation_context),
            "last_user_speech": self.last_user_speech_time
        }
    
    def set_meeting_context(self, context: str):
        """Establece el contexto de la reunión actual"""
        self.current_meeting_context = context
        logger.info(f"Contexto de reunión establecido: {context}")
    
    def export_conversation(self, file_path: str):
        """Exporta la conversación actual a un archivo"""
        import json
        
        export_data = {
            "timestamp": time.time(),
            "meeting_context": self.current_meeting_context,
            "conversation": [
                {
                    "speaker": t.speaker,
                    "text": t.text,
                    "timestamp": t.timestamp,
                    "confidence": t.confidence
                }
                for t in self.conversation_context
            ],
            "metrics": self.get_metrics()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversación exportada a: {file_path}")


if __name__ == "__main__":
    # Test básico del pipeline
    import asyncio
    
    async def test_pipeline():
        pipeline = MeetingAssistantPipeline()
        
        # Agregar callbacks de prueba
        def on_transcript(transcript):
            print(f"📝 Transcripción: {transcript.text} ({transcript.speaker})")
        
        def on_suggestion(suggestion):
            print(f"💡 Sugerencia: {suggestion.text[:50]}... (confianza: {suggestion.confidence:.2f})")
        
        def on_state_change(state_data):
            print(f"🔄 Estado: {state_data['old'].value} → {state_data['new'].value}")
        
        pipeline.add_callback("on_transcript", on_transcript)
        pipeline.add_callback("on_suggestion", on_suggestion)
        pipeline.add_callback("on_state_change", on_state_change)
        
        # Inicializar y ejecutar por 10 segundos
        await pipeline.setup()
        await pipeline.start()
        
        await asyncio.sleep(10)
        
        pipeline.stop()
        
        # Mostrar métricas finales
        metrics = pipeline.get_metrics()
        print("\n📊 Métricas finales:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    # Ejecutar test
    asyncio.run(test_pipeline())
