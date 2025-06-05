"""
Sistema de captura de audio optimizado para desarrollo lean
Enfoque: Captura audio del sistema sin unirse a reuniones
"""

import asyncio
import time
import platform
from typing import AsyncGenerator, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from loguru import logger


@dataclass
class AudioChunk:
    """Chunk de audio simplificado para procesamiento lean"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    duration: float
    
    @property
    def rms_level(self) -> float:
        """Calcula nivel RMS para detección de actividad"""
        return float(np.sqrt(np.mean(self.data**2)))


class SystemAudioCapture:
    """
    Capturador de audio del sistema optimizado para desarrollo lean
    Funciona sin necesidad de unirse a reuniones
    """
    
    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 2.0):
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.chunk_size = int(sample_rate * buffer_duration)
        
        self.is_running = False
        self.device_id = None
        self.stream = None
        
        # Buffer para acumular audio
        self.audio_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        logger.info(f"🎤 Sistema de captura lean: {sample_rate}Hz, chunks de {buffer_duration}s")
    
    async def _setup_system_audio_device(self) -> bool:
        """Detecta y configura el dispositivo de audio del sistema"""
        try:
            devices = sd.query_devices()
            
            # Buscar dispositivo de captura del sistema por plataforma
            if platform.system() == "Windows":
                # Buscar "Stereo Mix", "What U Hear", o "Mezcla estéreo"
                target_names = ["stereo mix", "what u hear", "mezcla", "wave out mix"]
                
                for i, device in enumerate(devices):
                    if (device['max_input_channels'] > 0 and 
                        any(name in device['name'].lower() for name in target_names)):
                        self.device_id = i
                        logger.success(f"✅ Dispositivo sistema encontrado: {device['name']}")
                        return True
                
                # Fallback: usar dispositivo default con warning
                logger.warning("⚠️ No se encontró dispositivo de sistema. Usando micrófono default.")
                logger.info("💡 Para capturar audio del sistema en Windows:")
                logger.info("   1. Panel de Control > Sonido > Grabación")
                logger.info("   2. Click derecho > Mostrar dispositivos deshabilitados")
                logger.info("   3. Habilitar 'Mezcla estéreo' o 'Stereo Mix'")
                
                self.device_id = sd.default.device[0]
                return True
                
            elif platform.system() == "Darwin":  # macOS
                # En macOS necesitamos BlackHole o SoundFlower
                target_names = ["blackhole", "soundflower", "loopback"]
                
                for i, device in enumerate(devices):
                    if (device['max_input_channels'] > 0 and 
                        any(name in device['name'].lower() for name in target_names)):
                        self.device_id = i
                        logger.success(f"✅ Dispositivo sistema encontrado: {device['name']}")
                        return True
                
                logger.warning("⚠️ No se encontró dispositivo de sistema en macOS.")
                logger.info("💡 Para capturar audio del sistema en macOS:")
                logger.info("   1. Instalar BlackHole: https://github.com/ExistentialAudio/BlackHole")
                logger.info("   2. O usar Loopback de Rogue Amoeba")
                
                self.device_id = sd.default.device[0]
                return True
                
            else:  # Linux
                # Buscar PulseAudio monitor
                target_names = ["monitor", "loopback"]
                
                for i, device in enumerate(devices):
                    if (device['max_input_channels'] > 0 and 
                        any(name in device['name'].lower() for name in target_names)):
                        self.device_id = i
                        logger.success(f"✅ Dispositivo sistema encontrado: {device['name']}")
                        return True
                
                logger.warning("⚠️ No se encontró monitor de audio en Linux.")
                logger.info("💡 Para capturar audio del sistema en Linux:")
                logger.info("   1. Usar: pactl load-module module-loopback")
                logger.info("   2. O configurar monitor en PulseAudio")
                
                self.device_id = sd.default.device[0]
                return True
        
        except Exception as e:
            logger.error(f"❌ Error configurando dispositivo de audio: {e}")
            return False
    
    async def stream_audio(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Generator que produce chunks de audio del sistema en tiempo real
        Optimizado para procesamiento lean
        """
        if not await self._setup_system_audio_device():
            logger.error("❌ No se pudo configurar captura de audio")
            return
        
        logger.info("🎧 Iniciando captura de audio del sistema...")
        self.is_running = True
        
        try:
            # Configurar stream de audio
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                
                # Convertir a mono si es estéreo
                if indata.shape[1] > 1:
                    audio_data = np.mean(indata, axis=1)
                else:
                    audio_data = indata.flatten()
                
                # Normalizar y filtrar ruido básico
                audio_data = self._preprocess_audio(audio_data)
                
                # Agregar al buffer
                asyncio.create_task(self._add_to_buffer(audio_data))
            
            # Crear stream
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=1,  # Mono para simplificar
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=audio_callback,
                dtype=np.float32
            )
            
            # Iniciar stream
            self.stream.start()
            
            # Loop principal - yield chunks cuando estén listos
            while self.is_running:
                async with self.buffer_lock:
                    if self.audio_buffer:
                        chunk_data = self.audio_buffer.pop(0)
                        
                        # Crear AudioChunk
                        chunk = AudioChunk(
                            data=chunk_data,
                            timestamp=time.time(),
                            sample_rate=self.sample_rate,
                            duration=len(chunk_data) / self.sample_rate
                        )
                        
                        yield chunk_data
                
                # Sleep corto para no saturar CPU
                await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"❌ Error en stream de audio: {e}")
        finally:
            await self.stop()
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocesamiento básico y eficiente de audio"""
        
        # Normalizar
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
        
        # Gate de ruido simple
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < 0.01:  # Umbral de ruido
            audio_data = np.zeros_like(audio_data)
        
        return audio_data
    
    async def _add_to_buffer(self, audio_data: np.ndarray):
        """Agrega audio al buffer de manera thread-safe"""
        async with self.buffer_lock:
            self.audio_buffer.append(audio_data.copy())
            
            # Limitar tamaño del buffer (máximo 10 chunks)
            if len(self.audio_buffer) > 10:
                self.audio_buffer.pop(0)
    
    async def stop(self):
        """Detiene la captura de audio"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        logger.info("🛑 Captura de audio detenida")
    
    def get_available_devices(self) -> list:
        """Lista dispositivos de audio disponibles para depuración"""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0:
                devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        return devices


# Factory function para configuración rápida
def create_system_capture(sample_rate: int = 16000, buffer_duration: float = 2.0) -> SystemAudioCapture:
    """Factory para crear capturador con configuración lean"""
    return SystemAudioCapture(sample_rate, buffer_duration)


if __name__ == "__main__":
    """Test básico del sistema de captura"""
    
    async def test_capture():
        capture = create_system_capture()
        
        logger.info("📱 Dispositivos disponibles:")
        for device in capture.get_available_devices():
            logger.info(f"  {device['id']}: {device['name']}")
        
        logger.info("\n🎤 Iniciando test de captura (5 segundos)...")
        
        chunk_count = 0
        async for audio_chunk in capture.stream_audio():
            chunk_count += 1
            rms = np.sqrt(np.mean(audio_chunk**2))
            logger.info(f"📊 Chunk {chunk_count}: RMS={rms:.4f}, Shape={audio_chunk.shape}")
            
            if chunk_count >= 3:  # Test corto
                break
        
        logger.info("✅ Test completado")
    
    asyncio.run(test_capture())
