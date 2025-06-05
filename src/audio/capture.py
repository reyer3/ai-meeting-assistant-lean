"""
Audio Capture Module para Call Centers

Este módulo maneja la captura de audio en tiempo real desde:
- Dispositivos de audio del sistema
- Líneas telefónicas
- Softphones
- Audio del sistema (What U Hear)
"""

import threading
import time
import logging
from typing import Optional, Callable, Dict, Any
from queue import Queue, Full
from dataclasses import dataclass
from enum import Enum

import sounddevice as sd
import numpy as np
from scipy import signal

from ..core.config import config

logger = logging.getLogger(__name__)


class AudioSource(Enum):
    """Tipos de fuente de audio"""
    MICROPHONE = "microphone"
    SYSTEM_AUDIO = "system_audio"  # What U Hear
    LINE_IN = "line_in"
    USB_DEVICE = "usb_device"
    VIRTUAL_CABLE = "virtual_cable"


@dataclass
class AudioChunk:
    """Chunk de audio capturado"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    channels: int
    duration: float
    source: AudioSource
    rms_level: float  # Nivel RMS para detección de actividad vocal


class AudioCapture:
    """Capturador de audio en tiempo real para call centers"""
    
    def __init__(self, 
                 sample_rate: int = None,
                 chunk_size: int = None,
                 channels: int = None,
                 device_name: str = None,
                 source_type: AudioSource = AudioSource.SYSTEM_AUDIO):
        
        # Configuración de audio
        self.sample_rate = sample_rate or config.audio.sample_rate
        self.chunk_size = chunk_size or config.audio.chunk_size
        self.channels = channels or config.audio.channels
        self.device_name = device_name or config.audio.device_name
        self.source_type = source_type
        
        # Estado del capturador
        self.is_capturing = False
        self.device_info = None
        self.stream = None
        self.capture_thread = None
        
        # Buffer de audio
        self.audio_buffer = Queue(maxsize=50)  # Buffer para ~2.5 segundos
        
        # Configuración de procesamiento
        self.noise_gate_threshold = 0.01  # Umbral para gate de ruido
        self.auto_gain_enabled = True
        self.gain_factor = 1.0
        
        # Callbacks
        self.on_audio_chunk: Optional[Callable[[AudioChunk], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Inicializar dispositivo
        self._initialize_device()
    
    def _initialize_device(self):
        """Inicializa y configura el dispositivo de audio"""
        try:
            # Listar dispositivos disponibles
            devices = sd.query_devices()
            logger.info(f"Dispositivos de audio disponibles: {len(devices)}")
            
            # Encontrar dispositivo específico o usar default
            if self.device_name:
                device_id = self._find_device_by_name(self.device_name)
                if device_id is None:
                    logger.warning(f"Dispositivo '{self.device_name}' no encontrado, usando default")
                    device_id = sd.default.device[0]  # Input device
            else:
                device_id = sd.default.device[0]
            
            # Obtener información del dispositivo
            self.device_info = sd.query_devices(device_id)
            logger.info(f"Usando dispositivo: {self.device_info['name']}")
            
            # Verificar capacidades
            self._verify_device_capabilities(device_id)
            
            # Configurar stream
            self.device_id = device_id
            
        except Exception as e:
            logger.error(f"Error inicializando dispositivo de audio: {e}")
            raise
    
    def _find_device_by_name(self, name: str) -> Optional[int]:
        """Encuentra dispositivo por nombre"""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if name.lower() in device['name'].lower() and device['max_input_channels'] > 0:
                return i
        return None
    
    def _verify_device_capabilities(self, device_id: int):
        """Verifica que el dispositivo soporte la configuración requerida"""
        try:
            sd.check_input_settings(
                device=device_id,
                channels=self.channels,
                samplerate=self.sample_rate
            )
            logger.info("Configuración de audio verificada exitosamente")
        except sd.PortAudioError as e:
            logger.error(f"Dispositivo no soporta configuración requerida: {e}")
            raise
    
    def start_capture(self, output_queue: Optional[Queue] = None) -> bool:
        """Inicia la captura de audio"""
        if self.is_capturing:
            logger.warning("Captura ya está en curso")
            return False
        
        try:
            # Configurar queue de salida
            if output_queue:
                self.audio_buffer = output_queue
            
            # Crear y configurar stream
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
                finished_callback=self._stream_finished_callback
            )
            
            # Iniciar stream
            self.stream.start()
            self.is_capturing = True
            
            logger.info("🎤 Captura de audio iniciada")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando captura: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    def stop_capture(self):
        """Detiene la captura de audio"""
        if not self.is_capturing:
            return
        
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            self.is_capturing = False
            logger.info("🛑 Captura de audio detenida")
            
        except Exception as e:
            logger.error(f"Error deteniendo captura: {e}")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback llamado por cada chunk de audio"""
        if status:
            logger.warning(f"Estado de audio: {status}")
        
        try:
            # Procesar audio
            processed_audio = self._process_audio_chunk(indata.copy())
            
            # Calcular nivel RMS
            rms_level = np.sqrt(np.mean(processed_audio**2))
            
            # Aplicar noise gate
            if rms_level < self.noise_gate_threshold:
                return  # Ignorar audio por debajo del umbral
            
            # Crear chunk de audio
            audio_chunk = AudioChunk(
                data=processed_audio,
                timestamp=time.time(),
                sample_rate=self.sample_rate,
                channels=self.channels,
                duration=frames / self.sample_rate,
                source=self.source_type,
                rms_level=rms_level
            )
            
            # Enviar a queue
            try:
                self.audio_buffer.put_nowait(audio_chunk)
            except Full:
                # Buffer lleno, descartar chunk más antiguo
                try:
                    self.audio_buffer.get_nowait()
                    self.audio_buffer.put_nowait(audio_chunk)
                except:
                    pass  # Si falla, simplemente descartar este chunk
            
            # Llamar callback si existe
            if self.on_audio_chunk:
                self.on_audio_chunk(audio_chunk)
                
        except Exception as e:
            logger.error(f"Error en callback de audio: {e}")
    
    def _process_audio_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        """Procesa chunk de audio (filtros, gain, etc.)"""
        # Convertir a float si es necesario
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Aplicar auto-gain si está habilitado
        if self.auto_gain_enabled:
            audio_data = self._apply_auto_gain(audio_data)
        
        # Aplicar filtro anti-aliasing básico
        audio_data = self._apply_low_pass_filter(audio_data)
        
        return audio_data
    
    def _apply_auto_gain(self, audio_data: np.ndarray) -> np.ndarray:
        """Aplica control automático de ganancia"""
        current_rms = np.sqrt(np.mean(audio_data**2))
        
        if current_rms > 0:
            # Target RMS level
            target_rms = 0.1
            
            # Ajustar gain gradualmente
            desired_gain = target_rms / current_rms
            
            # Limitar cambios bruscos de gain
            max_gain_change = 1.2
            min_gain_change = 0.8
            
            desired_gain = np.clip(desired_gain, min_gain_change, max_gain_change)
            
            # Suavizar cambios de gain
            alpha = 0.1  # Factor de suavizado
            self.gain_factor = (1 - alpha) * self.gain_factor + alpha * desired_gain
            
            # Aplicar gain
            audio_data *= self.gain_factor
        
        return audio_data
    
    def _apply_low_pass_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """Aplica filtro pasa-bajos para reducir ruido de alta frecuencia"""
        # Filtro Butterworth de 6to orden, frecuencia de corte a 8kHz
        nyquist = self.sample_rate / 2
        cutoff = min(8000, nyquist * 0.8)  # Evitar frecuencias muy cercanas a Nyquist
        
        try:
            # Diseñar filtro
            b, a = signal.butter(6, cutoff / nyquist, btype='low')
            
            # Aplicar filtro
            filtered_data = signal.filtfilt(b, a, audio_data, axis=0)
            return filtered_data.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Error aplicando filtro: {e}")
            return audio_data  # Retornar audio sin filtrar
    
    def _stream_finished_callback(self):
        """Callback llamado cuando el stream termina"""
        logger.info("Stream de audio terminado")
        self.is_capturing = False
    
    def get_audio_levels(self) -> Dict[str, float]:
        """Obtiene niveles de audio actuales"""
        if not self.is_capturing or self.audio_buffer.empty():
            return {"rms": 0.0, "peak": 0.0, "gain": self.gain_factor}
        
        # Obtener último chunk sin removerlo del buffer
        try:
            # Crear una copia temporal del buffer para peek
            temp_chunks = []
            while not self.audio_buffer.empty():
                chunk = self.audio_buffer.get_nowait()
                temp_chunks.append(chunk)
            
            # Restaurar buffer
            for chunk in temp_chunks:
                self.audio_buffer.put_nowait(chunk)
            
            if temp_chunks:
                last_chunk = temp_chunks[-1]
                peak_level = np.max(np.abs(last_chunk.data))
                
                return {
                    "rms": last_chunk.rms_level,
                    "peak": peak_level,
                    "gain": self.gain_factor
                }
                
        except Exception as e:
            logger.debug(f"Error obteniendo niveles de audio: {e}")
        
        return {"rms": 0.0, "peak": 0.0, "gain": self.gain_factor}
    
    def list_available_devices(self) -> List[Dict[str, Any]]:
        """Lista todos los dispositivos de audio disponibles"""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0:  # Solo dispositivos de input
                devices.append({
                    "id": i,
                    "name": device['name'],
                    "channels": device['max_input_channels'],
                    "sample_rate": device['default_samplerate'],
                    "hostapi": sd.query_hostapis(device['hostapi'])['name']
                })
        return devices
    
    def test_device(self, device_id: int, duration: float = 2.0) -> Dict[str, Any]:
        """Prueba un dispositivo específico"""
        test_results = {
            "success": False,
            "average_level": 0.0,
            "peak_level": 0.0,
            "error": None
        }
        
        try:
            # Configurar stream de prueba
            test_data = []
            
            def test_callback(indata, frames, time, status):
                test_data.append(indata.copy())
            
            with sd.InputStream(
                device=device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=test_callback
            ):
                sd.sleep(int(duration * 1000))  # Convertir a ms
            
            # Analizar datos capturados
            if test_data:
                all_data = np.concatenate(test_data)
                test_results.update({
                    "success": True,
                    "average_level": np.sqrt(np.mean(all_data**2)),
                    "peak_level": np.max(np.abs(all_data)),
                    "samples_captured": len(all_data)
                })
            
        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Error probando dispositivo {device_id}: {e}")
        
        return test_results
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene estado actual del capturador"""
        return {
            "is_capturing": self.is_capturing,
            "device_name": self.device_info['name'] if self.device_info else None,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size,
            "buffer_size": self.audio_buffer.qsize(),
            "gain_factor": self.gain_factor,
            "source_type": self.source_type.value
        }


class CallCenterAudioCapture(AudioCapture):
    """Capturador especializado para call centers"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Configuración específica para call centers
        self.dual_channel_mode = False  # Canal 1: Agente, Canal 2: Cliente
        self.channel_separation_enabled = False
        
        # Buffers separados para cada canal
        self.agent_buffer = Queue(maxsize=25)
        self.client_buffer = Queue(maxsize=25)
    
    def enable_dual_channel_mode(self, channels: int = 2):
        """Habilita modo dual canal para separar agente y cliente"""
        if channels != 2:
            raise ValueError("Modo dual canal requiere exactamente 2 canales")
        
        self.channels = 2
        self.dual_channel_mode = True
        self.channel_separation_enabled = True
        
        logger.info("Modo dual canal habilitado (Canal 1: Agente, Canal 2: Cliente)")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback especializado para call centers"""
        if status:
            logger.warning(f"Estado de audio: {status}")
        
        try:
            if self.dual_channel_mode and indata.shape[1] >= 2:
                # Separar canales
                agent_audio = indata[:, 0]  # Canal izquierdo (agente)
                client_audio = indata[:, 1]  # Canal derecho (cliente)
                
                # Procesar cada canal por separado
                agent_processed = self._process_audio_chunk(agent_audio.reshape(-1, 1))
                client_processed = self._process_audio_chunk(client_audio.reshape(-1, 1))
                
                # Crear chunks separados
                agent_chunk = AudioChunk(
                    data=agent_processed,
                    timestamp=time.time(),
                    sample_rate=self.sample_rate,
                    channels=1,
                    duration=frames / self.sample_rate,
                    source=AudioSource.MICROPHONE,  # Agente
                    rms_level=np.sqrt(np.mean(agent_processed**2))
                )
                
                client_chunk = AudioChunk(
                    data=client_processed,
                    timestamp=time.time(),
                    sample_rate=self.sample_rate,
                    channels=1,
                    duration=frames / self.sample_rate,
                    source=AudioSource.LINE_IN,  # Cliente
                    rms_level=np.sqrt(np.mean(client_processed**2))
                )
                
                # Enviar a buffers correspondientes
                try:
                    self.agent_buffer.put_nowait(agent_chunk)
                    self.client_buffer.put_nowait(client_chunk)
                except Full:
                    # Manejar buffers llenos
                    pass
                
            else:
                # Modo single channel - usar callback padre
                super()._audio_callback(indata, frames, time_info, status)
                
        except Exception as e:
            logger.error(f"Error en callback de call center: {e}")
    
    def get_separated_audio(self) -> Dict[str, Optional[AudioChunk]]:
        """Obtiene audio separado por canal"""
        result = {"agent": None, "client": None}
        
        try:
            if not self.agent_buffer.empty():
                result["agent"] = self.agent_buffer.get_nowait()
            
            if not self.client_buffer.empty():
                result["client"] = self.client_buffer.get_nowait()
                
        except Exception as e:
            logger.debug(f"Error obteniendo audio separado: {e}")
        
        return result


if __name__ == "__main__":
    # Test del audio capture
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\n🛑 Deteniendo captura...")
        capture.stop_capture()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== Test Audio Capture para Call Centers ===")
    
    # Listar dispositivos disponibles
    capture = AudioCapture()
    devices = capture.list_available_devices()
    
    print("\n📱 Dispositivos disponibles:")
    for device in devices:
        print(f"  {device['id']}: {device['name']} ({device['channels']} canales)")
    
    # Test de captura
    print("\n🎤 Iniciando captura de audio (Ctrl+C para detener)...")
    
    def on_audio_chunk(chunk: AudioChunk):
        print(f"📊 Audio: RMS={chunk.rms_level:.3f}, Peak={np.max(np.abs(chunk.data)):.3f}, Duration={chunk.duration:.2f}s")
    
    capture.on_audio_chunk = on_audio_chunk
    
    if capture.start_capture():
        print("✅ Captura iniciada. Habla al micrófono...")
        
        try:
            while True:
                time.sleep(1)
                status = capture.get_status()
                levels = capture.get_audio_levels()
                print(f"📈 Buffer: {status['buffer_size']}, RMS: {levels['rms']:.3f}, Gain: {levels['gain']:.2f}")
                
        except KeyboardInterrupt:
            pass
    else:
        print("❌ Error iniciando captura")
    
    capture.stop_capture()
    print("🏁 Test completado")
