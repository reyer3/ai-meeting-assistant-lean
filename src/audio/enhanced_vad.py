#!/usr/bin/env python3
"""
Voice Activity Detection mejorado para timing preciso
Detecta CUÁNDO otros terminan de hablar para trigger sugerencias
"""

import numpy as np
import time
from typing import Dict, Any, Optional, List
from loguru import logger
from collections import deque

class EnhancedVAD:
    """
    VAD optimizado para detectar pausas y cambios de speaker
    Clave para saber CUÁNDO generar sugerencias
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_duration: float = 0.03,  # 30ms frames
                 energy_threshold: float = 0.01,
                 silence_duration: float = 1.0):  # 1s de silencio = fin de speech
        
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        
        # Estado del VAD
        self.is_speech_active = False
        self.last_speech_end = 0
        self.silence_start = None
        
        # Buffer de energía para análisis
        self.energy_buffer = deque(maxlen=int(2.0 / frame_duration))  # 2 segundos
        
        # Estados para diferentes speakers
        self.current_speaker = None  # "user" | "other" | None
        self.last_speaker_change = 0
        
        logger.info("🎙️ Enhanced VAD inicializado")
    
    def process_frame(self, 
                     audio_frame: np.ndarray, 
                     speaker_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa frame de audio y detecta eventos de speech
        
        Returns:
            - speech_event: "speech_start" | "speech_end" | "silence" | "continuing"
            - speaker_type: "user" | "other" | None
            - should_generate_suggestion: bool
            - silence_duration: float
        """
        
        current_time = time.time()
        
        # Calcular energía del frame
        if len(audio_frame) < self.frame_size:
            audio_frame = np.pad(audio_frame, (0, self.frame_size - len(audio_frame)))
        elif len(audio_frame) > self.frame_size:
            audio_frame = audio_frame[:self.frame_size]
        
        frame_energy = np.sqrt(np.mean(audio_frame ** 2))
        self.energy_buffer.append(frame_energy)
        
        # Determinar si hay speech activo
        has_speech = frame_energy > self.energy_threshold
        speaker_type = "user" if speaker_info.get('is_user_voice', False) else "other"
        
        # Detectar cambios de estado
        speech_event = "continuing"
        should_generate_suggestion = False
        
        if has_speech:
            if not self.is_speech_active:
                # Speech started
                speech_event = "speech_start"
                self.is_speech_active = True
                self.silence_start = None
                
                # Detectar cambio de speaker
                if self.current_speaker != speaker_type:
                    self.last_speaker_change = current_time
                    self.current_speaker = speaker_type
                    
                    logger.debug(f"🔄 Speaker change: {self.current_speaker}")
            
            # Reset silence tracking
            self.silence_start = None
        
        else:
            # No speech in current frame
            if self.is_speech_active:
                # Might be transitioning to silence
                if self.silence_start is None:
                    self.silence_start = current_time
                
                silence_duration = current_time - self.silence_start
                
                if silence_duration >= self.silence_duration:
                    # Speech definitively ended
                    speech_event = "speech_end"
                    self.is_speech_active = False
                    self.last_speech_end = current_time
                    
                    # 🎯 CLAVE: Generar sugerencia si otros acaban de hablar
                    if self.current_speaker == "other":
                        should_generate_suggestion = True
                        logger.debug("💡 Others finished speaking - trigger suggestion")
                    
                    self.current_speaker = None
        
        # Calcular métricas
        silence_duration = (current_time - self.last_speech_end) if not self.is_speech_active else 0
        
        return {
            "speech_event": speech_event,
            "speaker_type": speaker_type if has_speech else None,
            "should_generate_suggestion": should_generate_suggestion,
            "silence_duration": silence_duration,
            "frame_energy": frame_energy,
            "is_speech_active": self.is_speech_active,
            "current_speaker": self.current_speaker
        }
    
    def get_adaptive_threshold(self) -> float:
        """
        Calcula threshold adaptativo basado en ruido de fondo
        """
        if len(self.energy_buffer) < 10:
            return self.energy_threshold
        
        # Usar percentil 20 como estimación de ruido de fondo
        energies = list(self.energy_buffer)
        noise_floor = np.percentile(energies, 20)
        
        # Threshold = 3x ruido de fondo
        adaptive_threshold = max(self.energy_threshold, noise_floor * 3)
        
        return adaptive_threshold
    
    def update_thresholds(self):
        """
        Actualiza thresholds dinámicamente
        """
        new_threshold = self.get_adaptive_threshold()
        if abs(new_threshold - self.energy_threshold) > 0.001:
            self.energy_threshold = new_threshold
            logger.debug(f"🎚️ Threshold adaptado: {self.energy_threshold:.4f}")


def create_enhanced_vad(sample_rate: int = 16000) -> EnhancedVAD:
    """Factory para VAD optimizado"""
    return EnhancedVAD(
        sample_rate=sample_rate,
        frame_duration=0.03,  # 30ms = buen balance
        energy_threshold=0.01,  # Ajustable dinámicamente  
        silence_duration=0.8  # 800ms = pausa natural en conversación
    )
