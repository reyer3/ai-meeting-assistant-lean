#!/usr/bin/env python3
"""
Pipeline optimizado para desarrollo lean - Asistente de reuniones IA
Enfoque: App de escritorio que captura audio del sistema sin unirse a reuniones
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from audio.capture import SystemAudioCapture
from audio.speaker_id import SpeakerIdentifier
from ai.stt import WhisperSTT
from ai.llm import LocalLLM
from rag.query_engine import RAGQueryEngine


@dataclass
class ProcessingConfig:
    """Configuración optimizada para rendimiento lean"""
    
    # Audio settings
    sample_rate: int = 16000
    buffer_duration: float = 2.0  # segundos de audio por chunk
    voice_threshold: float = 0.7  # umbral de confianza para identificación
    
    # AI settings  
    model_name: str = "qwen2.5:0.5b"  # Modelo pequeño sin GPU
    max_context_length: int = 2048
    temperature: float = 0.3  # Respuestas más determinísticas
    
    # RAG settings
    max_rag_results: int = 3
    similarity_threshold: float = 0.75
    
    # Performance
    processing_timeout: float = 6.0  # Max tiempo total por sugerencia
    min_suggestion_interval: float = 10.0  # Evitar spam de sugerencias


class LeanMeetingAssistant:
    """Pipeline principal optimizado para desarrollo lean"""
    
    def __init__(self, config: ProcessingConfig, user_profile_path: str):
        self.config = config
        self.user_profile_path = user_profile_path
        self.is_running = False
        self.last_suggestion_time = 0
        
        # Componentes principales
        self.audio_capture: Optional[SystemAudioCapture] = None
        self.speaker_id: Optional[SpeakerIdentifier] = None
        self.stt: Optional[WhisperSTT] = None
        self.llm: Optional[LocalLLM] = None
        self.rag: Optional[RAGQueryEngine] = None
        
        # Estado de conversación
        self.conversation_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 20
        
        logger.info("🚀 Lean Meeting Assistant inicializado")
    
    async def initialize(self) -> bool:
        """Inicialización rápida de componentes esenciales"""
        try:
            start_time = time.time()
            logger.info("⚡ Iniciando componentes lean...")
            
            # 1. Audio capture (crítico)
            logger.info("🎤 Configurando captura de audio del sistema...")
            self.audio_capture = SystemAudioCapture(
                sample_rate=self.config.sample_rate,
                buffer_duration=self.config.buffer_duration
            )
            
            # 2. Speaker identification (crítico para diferenciación)
            logger.info("👤 Cargando perfil de voz personal...")
            self.speaker_id = SpeakerIdentifier()
            if not await self.speaker_id.load_user_profile(self.user_profile_path):
                logger.error("❌ No se pudo cargar perfil de voz. Ejecuta setup_voice_profile.py")
                return False
            
            # 3. STT local (crítico)
            logger.info("🗣️ Inicializando Whisper local...")
            self.stt = WhisperSTT(model_size="base")  # Balance velocidad/calidad
            
            # 4. LLM pequeño local (crítico) 
            logger.info("🧠 Conectando con modelo local...")
            self.llm = LocalLLM(
                model_name=self.config.model_name,
                max_context_length=self.config.max_context_length,
                temperature=self.config.temperature
            )
            
            # 5. RAG engine (para contexto)
            logger.info("📚 Inicializando sistema RAG...")
            self.rag = RAGQueryEngine()
            
            init_time = time.time() - start_time
            logger.success(f"✅ Inicialización completa en {init_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en inicialización: {e}")
            return False
    
    async def start_listening(self):
        """Inicia el loop principal de escucha y procesamiento"""
        if not await self.initialize():
            return
        
        self.is_running = True
        logger.info("👂 Iniciando escucha de audio del sistema...")
        
        try:
            async for audio_chunk in self.audio_capture.stream_audio():
                if not self.is_running:
                    break
                
                await self._process_audio_chunk(audio_chunk)
                
        except KeyboardInterrupt:
            logger.info("⏹️ Deteniendo asistente...")
        except Exception as e:
            logger.error(f"❌ Error en loop principal: {e}")
        finally:
            await self.stop()
    
    async def _process_audio_chunk(self, audio_data: np.ndarray):
        """Procesamiento lean de chunk de audio"""
        processing_start = time.time()
        
        try:
            # 1. Identificación rápida de voz
            speaker_info = await self.speaker_id.identify_speaker(audio_data)
            
            if speaker_info['is_user_voice'] and speaker_info['confidence'] > self.config.voice_threshold:
                logger.debug(f"🎯 Tu voz detectada (confianza: {speaker_info['confidence']:.2f})")
                
                # 2. Transcripción solo de tu voz
                transcription = await self.stt.transcribe(audio_data)
                
                if transcription and len(transcription.strip()) > 10:
                    # 3. Análisis de contexto y generación de sugerencia
                    await self._analyze_and_suggest(transcription, processing_start)
            
            else:
                # Solo capturar contexto de otros hablantes (sin transcribir)
                if speaker_info['confidence'] > 0.5:
                    logger.debug("👥 Voz de otro participante detectada")
                    # Agregar info de contexto sin transcripción completa
                    self._add_context_marker("other_speaker_active")
        
        except Exception as e:
            logger.error(f"❌ Error procesando audio: {e}")
    
    async def _analyze_and_suggest(self, user_speech: str, start_time: float):
        """Análisis contextual y generación de sugerencias AEIOU"""
        
        # Control de frecuencia de sugerencias
        current_time = time.time()
        if current_time - self.last_suggestion_time < self.config.min_suggestion_interval:
            return
        
        try:
            # 1. Detectar si necesita sugerencia (tensión, pregunta, etc.)
            needs_suggestion = await self._detect_suggestion_trigger(user_speech)
            
            if not needs_suggestion:
                logger.debug("ℹ️ No se detectó necesidad de sugerencia")
                self._add_to_conversation_buffer(user_speech, "user")
                return
            
            # 2. Query RAG para contexto similar
            logger.debug("🔍 Buscando contexto en base de conocimiento...")
            rag_context = await self.rag.query_similar_situations(
                user_speech, 
                max_results=self.config.max_rag_results,
                threshold=self.config.similarity_threshold
            )
            
            # 3. Generar sugerencia AEIOU enriquecida
            logger.debug("💡 Generando sugerencia AEIOU...")
            suggestion = await self.llm.generate_aeiou_suggestion(
                user_speech=user_speech,
                conversation_context=self.conversation_buffer[-5:],  # Últimas 5 intervenciones
                rag_context=rag_context
            )
            
            # 4. Mostrar sugerencia 
            await self._display_suggestion(suggestion, rag_context)
            
            # 5. Update estado
            self.last_suggestion_time = current_time
            self._add_to_conversation_buffer(user_speech, "user")
            
            processing_time = time.time() - start_time
            logger.info(f"⚡ Sugerencia generada en {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error generando sugerencia: {e}")
    
    async def _detect_suggestion_trigger(self, text: str) -> bool:
        """Detecta si el texto requiere una sugerencia AEIOU"""
        
        # Palabras clave que indican tensión o necesidad de diplomacia
        tension_keywords = [
            "no entiendes", "no funciona", "problema", "error", "mal", 
            "imposible", "no puedo", "difícil", "complicado", "bloqueado",
            "retraso", "deadline", "presión", "urgente", "crítico"
        ]
        
        # Patrones de preguntas difíciles
        difficult_patterns = [
            "por qué no", "cómo es posible", "no estás de acuerdo",
            "qué vamos a hacer", "cómo solucionamos", "qué propones"
        ]
        
        text_lower = text.lower()
        
        # Check keywords
        has_tension = any(keyword in text_lower for keyword in tension_keywords)
        has_difficult_pattern = any(pattern in text_lower for pattern in difficult_patterns)
        
        # Check si termina en pregunta
        is_question = text.strip().endswith('?')
        
        # Check longitud (conversaciones largas suelen necesitar estructura)
        is_long_statement = len(text.split()) > 15
        
        trigger = has_tension or has_difficult_pattern or (is_question and is_long_statement)
        
        if trigger:
            logger.debug(f"🎯 Trigger detectado: tensión={has_tension}, patrón={has_difficult_pattern}, pregunta_larga={is_question and is_long_statement}")
        
        return trigger
    
    async def _display_suggestion(self, suggestion: Dict[str, Any], rag_context: List[Dict]):
        """Muestra la sugerencia en overlay UI"""
        
        # Formato de salida para desarrollo lean (consola por ahora)
        print("\n" + "="*80)
        print("💡 SUGERENCIA AEIOU")
        print("="*80)
        print(f"📝 {suggestion['response']}")
        
        if rag_context:
            print(f"\n📊 Basado en {len(rag_context)} situaciones similares")
            print(f"🎯 Efectividad promedio: {suggestion.get('confidence', 0.8):.0%}")
        
        print("="*80 + "\n")
        
        # TODO: Integrar con overlay UI real en siguiente iteración
    
    def _add_to_conversation_buffer(self, text: str, speaker_type: str):
        """Mantiene buffer de conversación para contexto"""
        self.conversation_buffer.append({
            "text": text,
            "speaker": speaker_type,
            