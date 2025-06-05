"""
Speech-to-Text optimizado para desarrollo lean
Enfoque: Transcripción local rápida usando Whisper small/base
"""

import asyncio
import time
import tempfile
import os
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
import whisper
from loguru import logger
import soundfile as sf


class WhisperSTT:
    """
    STT lean usando Whisper local
    Optimizado para velocidad con modelos pequeños
    """
    
    def __init__(self, model_size: str = "base", language: str = "es"):
        self.model_size = model_size
        self.language = language
        self.model = None
        self.sample_rate = 16000
        
        # Configuración lean para velocidad
        self.transcribe_options = {
            "language": language,
            "task": "transcribe",
            "verbose": False,
            "condition_on_previous_text": False,  # Más rápido
            "temperature": 0.0,  # Determinístico
            "compression_ratio_threshold": 2.4,
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0
        }
        
        logger.info(f"🗣️ WhisperSTT lean inicializado: modelo {model_size}")
    
    async def initialize(self) -> bool:
        """Inicializa el modelo Whisper"""
        try:
            logger.info(f"📥 Cargando modelo Whisper {self.model_size}...")
            
            # Cargar en thread pool para no bloquear async
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                whisper.load_model, 
                self.model_size
            )
            
            logger.success(f"✅ Modelo Whisper {self.model_size} cargado")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando Whisper: {e}")
            logger.info("💡 Instalar con: pip install openai-whisper")
            return False
    
    async def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe audio usando Whisper
        
        Args:
            audio_data: Array de audio (16kHz, mono)
            
        Returns:
            Texto transcrito o None si falla
        """
        
        if self.model is None:
            if not await self.initialize():
                return None
        
        start_time = time.time()
        
        try:
            # Verificar que el audio tenga contenido
            if len(audio_data) == 0:
                return None
            
            # Normalizar audio
            audio_data = self._preprocess_audio(audio_data)
            
            # Verificar nivel de audio (evitar transcribir silencio)
            rms_level = np.sqrt(np.mean(audio_data**2))
            if rms_level < 0.01:  # Umbral de silencio
                logger.debug("🔇 Audio demasiado bajo, omitiendo transcripción")
                return None
            
            # Transcribir en thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_data
            )
            
            processing_time = time.time() - start_time
            
            if result and result["text"].strip():
                text = result["text"].strip()
                logger.debug(f"📝 Transcrito en {processing_time:.2f}s: {text[:50]}...")
                return text
            else:
                logger.debug("🔇 No se detectó habla")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error en transcripción: {e}")
            return None
    
    def _transcribe_sync(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Transcripción síncrona para ejecutar en thread pool"""
        return self.model.transcribe(audio_data, **self.transcribe_options)
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocesa audio para optimizar transcripción"""
        
        # Asegurar formato correcto
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        # Convertir a float32 si es necesario
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalizar al rango [-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95
        
        # Pad o trim a múltiplo de 16000 (1 segundo mínimo)
        min_length = self.sample_rate  # 1 segundo
        if len(audio_data) < min_length:
            # Pad con ceros
            audio_data = np.pad(audio_data, (0, min_length - len(audio_data)))
        elif len(audio_data) > self.sample_rate * 30:  # Máximo 30 segundos
            # Trim audio muy largo
            audio_data = audio_data[:self.sample_rate * 30]
        
        return audio_data
    
    async def transcribe_file(self, file_path: str) -> Optional[str]:
        """Transcribe archivo de audio"""
        
        try:
            # Cargar archivo
            audio_data, sr = sf.read(file_path)
            
            # Resampleas si es necesario
            if sr != self.sample_rate:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            
            return await self.transcribe(audio_data)
            
        except Exception as e:
            logger.error(f"❌ Error transcribiendo archivo {file_path}: {e}")
            return None
    
    async def test_transcription(self, test_text: str = "Hola, esto es una prueba de transcripción") -> Dict[str, Any]:
        """Test de transcripción con audio sintético"""
        
        try:
            # Generar audio de prueba (silencio con un poco de ruido)
            duration = 3  # segundos
            test_audio = np.random.normal(0, 0.1, self.sample_rate * duration).astype(np.float32)
            
            # Agregar "habla" sintética (ondas sinusoidales)
            t = np.linspace(0, duration, self.sample_rate * duration)
            speech_signal = (
                0.3 * np.sin(2 * np.pi * 200 * t) +  # Fundamental
                0.2 * np.sin(2 * np.pi * 400 * t) +  # Primer armónico
                0.1 * np.sin(2 * np.pi * 600 * t)    # Segundo armónico
            )
            
            # Mezclar con ruido
            test_audio = 0.7 * speech_signal + 0.3 * test_audio
            
            start_time = time.time()
            result = await self.transcribe(test_audio)
            processing_time = time.time() - start_time
            
            return {
                "success": result is not None,
                "transcribed_text": result,
                "processing_time": processing_time,
                "audio_duration": duration,
                "model_size": self.model_size
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_size": self.model_size
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo"""
        
        model_info = {
            "model_size": self.model_size,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "model_loaded": self.model is not None
        }
        
        if self.model is not None:
            # Información adicional si el modelo está cargado
            model_info.update({
                "model_dimensions": getattr(self.model, 'dims', None),
                "transcribe_options": self.transcribe_options
            })
        
        return model_info
    
    async def cleanup(self):
        """Limpia recursos del modelo"""
        if self.model is not None:
            # Whisper no requiere limpieza específica
            self.model = None
            logger.info("🧹 Modelo Whisper limpiado")


# Factory function para configuración lean
def create_lean_stt(model_size: str = "base", language: str = "es") -> WhisperSTT:
    """
    Factory para crear STT con configuración lean
    
    Modelos disponibles por tamaño y velocidad:
    - tiny: ~32MB, ~5x faster, menor precisión
    - base: ~74MB, ~3x faster, buena precisión (RECOMENDADO para lean)
    - small: ~244MB, ~2x faster, muy buena precisión
    - medium: ~769MB, ~1.5x faster, excelente precisión
    """
    return WhisperSTT(model_size, language)


if __name__ == "__main__":
    """Test del STT lean"""
    
    async def test_lean_stt():
        logger.info("🧪 Iniciando test de STT lean...")
        
        # Crear STT con modelo base (recomendado para lean)
        stt = create_lean_stt("base", "es")
        
        # Mostrar info del modelo
        model_info = stt.get_model_info()
        logger.info(f"📊 Info del modelo:")
        for key, value in model_info.items():
            logger.info(f"   {key}: {value}")
        
        # Test de transcripción
        logger.info("🎤 Ejecutando test de transcripción...")
        
        test_result = await stt.test_transcription()
        
        logger.info("📋 Resultado del test:")
        for key, value in test_result.items():
            logger.info(f"   {key}: {value}")
        
        if test_result["success"]:
            logger.success("✅ STT funcionando correctamente")
        else:
            logger.error("❌ STT falló en el test")
        
        # Cleanup
        await stt.cleanup()
        
        logger.success("✅ Test completado")
    
    asyncio.run(test_lean_stt())
