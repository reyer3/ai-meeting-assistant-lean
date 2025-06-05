"""
Speech-to-Text Local Engine

Este módulo maneja la transcripción de audio en tiempo real usando:
- Whisper.cpp (optimizado para CPU)
- Modelos locales sin dependencias de nube
- Vocabulario especializado para cobranza
"""

import os
import logging
import tempfile
import subprocess
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import json
from queue import Queue, Empty
from threading import Thread, Event
import time

import numpy as np
import soundfile as sf

from ..core.config import config
from ..audio.capture import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Resultado de transcripción"""
    text: str
    confidence: float
    language: str
    duration: float
    timestamp: float
    segments: List[Dict[str, Any]] = None
    processing_time: float = 0.0


class WhisperSTT:
    """Speech-to-Text usando Whisper.cpp local"""
    
    def __init__(self, 
                 model_path: str = None,
                 language: str = None,
                 model_size: str = None):
        
        self.model_path = model_path or self._get_default_model_path()
        self.language = language or config.stt.language
        self.model_size = model_size or config.stt.model_size.value
        
        # Verificar que el modelo existe
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Modelo Whisper no encontrado: {self.model_path}")
        
        # Configuración de transcripción
        self.whisper_executable = self._find_whisper_executable()
        self.temp_dir = Path(tempfile.gettempdir()) / "ai_collections_stt"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Vocabulario especializado para cobranza
        self.collections_vocabulary = self._load_collections_vocabulary()
        
        # Cache de transcripciones recientes
        self.transcription_cache = {}
        self.cache_max_size = 100
        
        # Métricas
        self.metrics = {
            "total_transcriptions": 0,
            "average_processing_time": 0.0,
            "average_confidence": 0.0,
            "errors": 0
        }
        
        logger.info(f"WhisperSTT inicializado con modelo {self.model_size} ({self.language})")
    
    def _get_default_model_path(self) -> str:
        """Obtiene la ruta por defecto del modelo Whisper"""
        models_dir = Path(config.base_dir) / "models" / "whisper"
        model_file = f"ggml-{config.stt.model_size.value}.bin"
        return str(models_dir / model_file)
    
    def _find_whisper_executable(self) -> str:
        """Encuentra el ejecutable de whisper.cpp"""
        # Buscar en ubicaciones comunes
        possible_paths = [
            "whisper",  # En PATH
            "./whisper",  # Directorio actual
            "../whisper/main",  # Directorio padre
            "/usr/local/bin/whisper",  # Unix
            "C:\\whisper\\main.exe"  # Windows
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "--version"], 
                                       capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info(f"Whisper encontrado en: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        # Si no se encuentra, intentar con Python whisper como fallback
        logger.warning("Whisper.cpp no encontrado, usando fallback a openai-whisper")
        return "python -m whisper"
    
    def _load_collections_vocabulary(self) -> List[str]:
        """Carga vocabulario especializado para cobranza"""
        vocabulary = [
            # Términos financieros
            "deuda", "balance", "pago", "cuota", "interés", "principal",
            "settlement", "arreglo", "plan de pagos", "financiamiento",
            
            # Términos legales
            "validación", "disputa", "crédito", "reporte", "bureau",
            "abogado", "demanda", "corte", "juzgado",
            
            # Objeciones comunes
            "desempleo", "enfermedad", "divorcio", "emergencia",
            "pandemia", "COVID", "hospitalización",
            
            # Términos de cobranza
            "collector", "agencia", "cobranza", "recuperación",
            "promesa", "compromiso", "acuerdo"
        ]
        
        # Cargar vocabulario personalizado si existe
        vocab_file = config.get_data_path("collections_vocabulary.txt")
        if vocab_file.exists():
            try:
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    custom_vocab = [line.strip() for line in f if line.strip()]
                vocabulary.extend(custom_vocab)
                logger.info(f"Vocabulario personalizado cargado: {len(custom_vocab)} términos")
            except Exception as e:
                logger.warning(f"Error cargando vocabulario personalizado: {e}")
        
        return vocabulary
    
    def transcribe(self, audio_chunk: AudioChunk) -> Optional[TranscriptionResult]:
        """Transcribe un chunk de audio"""
        start_time = time.time()
        
        try:
            # Verificar calidad mínima del audio
            if audio_chunk.rms_level < 0.01:
                return None  # Audio demasiado silencioso
            
            # Verificar duración mínima
            if audio_chunk.duration < 0.5:
                return None  # Audio demasiado corto
            
            # Verificar cache
            audio_hash = self._hash_audio(audio_chunk.data)
            if audio_hash in self.transcription_cache:
                cached_result = self.transcription_cache[audio_hash]
                logger.debug("Transcripción obtenida de cache")
                return cached_result
            
            # Guardar audio temporal
            temp_audio_file = self._save_temp_audio(audio_chunk)
            
            # Ejecutar Whisper
            result = self._run_whisper(temp_audio_file)
            
            # Limpiar archivo temporal
            self._cleanup_temp_file(temp_audio_file)
            
            # Procesar resultado
            if result:
                processing_time = time.time() - start_time
                result.processing_time = processing_time
                result.timestamp = audio_chunk.timestamp
                
                # Agregar a cache
                self._add_to_cache(audio_hash, result)
                
                # Actualizar métricas
                self._update_metrics(result)
                
                logger.debug(f"Transcripción: '{result.text}' (confianza: {result.confidence:.2f})")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error en transcripción: {e}")
            self.metrics["errors"] += 1
            return None
    
    def _hash_audio(self, audio_data: np.ndarray) -> str:
        """Genera hash del audio para cache"""
        return str(hash(audio_data.tobytes()))
    
    def _save_temp_audio(self, audio_chunk: AudioChunk) -> str:
        """Guarda audio en archivo temporal"""
        temp_file = self.temp_dir / f"audio_{int(time.time()*1000)}.wav"
        
        # Convertir a formato compatible
        audio_data = audio_chunk.data.flatten().astype(np.float32)
        
        # Normalizar si es necesario
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Guardar como WAV
        sf.write(temp_file, audio_data, audio_chunk.sample_rate)
        
        return str(temp_file)
    
    def _run_whisper(self, audio_file: str) -> Optional[TranscriptionResult]:
        """Ejecuta Whisper.cpp en el archivo de audio"""
        try:
            # Construir comando
            cmd = [
                self.whisper_executable,
                "-m", self.model_path,
                "-f", audio_file,
                "-l", self.language,
                "--output-json",
                "--no-timestamps"  # Para transcripción más rápida
            ]
            
            # Agregar vocabulario si está soportado
            if self.collections_vocabulary and "--prompt" in self._get_whisper_help():
                vocab_prompt = ", ".join(self.collections_vocabulary[:20])  # Primeras 20 palabras
                cmd.extend(["--prompt", vocab_prompt])
            
            # Ejecutar comando
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,  # Timeout de 30 segundos
                cwd=self.temp_dir
            )
            
            if result.returncode == 0:
                return self._parse_whisper_output(result.stdout)
            else:
                logger.error(f"Error ejecutando Whisper: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout ejecutando Whisper")
            return None
        except Exception as e:
            logger.error(f"Error ejecutando Whisper: {e}")
            return None
    
    def _get_whisper_help(self) -> str:
        """Obtiene ayuda de Whisper para verificar opciones disponibles"""
        try:
            result = subprocess.run(
                [self.whisper_executable, "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout
        except:
            return ""
    
    def _parse_whisper_output(self, output: str) -> Optional[TranscriptionResult]:
        """Parsea la salida JSON de Whisper"""
        try:
            # Buscar línea JSON válida
            lines = output.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)
                        
                        # Extraer texto principal
                        text = data.get('text', '').strip()
                        
                        if not text:
                            continue
                        
                        # Estimar confianza basado en coherencia
                        confidence = self._estimate_confidence(text, data)
                        
                        # Detectar idioma
                        language = data.get('language', self.language)
                        
                        return TranscriptionResult(
                            text=text,
                            confidence=confidence,
                            language=language,
                            duration=data.get('duration', 0.0),
                            timestamp=0.0,  # Se establecerá en transcribe()
                            segments=data.get('segments', [])
                        )
                        
                    except json.JSONDecodeError:
                        continue
            
            # Si no hay JSON válido, intentar extraer texto plano
            text = self._extract_plain_text(output)
            if text:
                return TranscriptionResult(
                    text=text,
                    confidence=0.5,  # Confianza media para texto plano
                    language=self.language,
                    duration=0.0,
                    timestamp=0.0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error parseando salida de Whisper: {e}")
            return None
    
    def _extract_plain_text(self, output: str) -> str:
        """Extrae texto plano de la salida de Whisper"""
        lines = output.strip().split('\n')
        
        # Buscar líneas que parezcan transcripción
        for line in lines:
            line = line.strip()
            
            # Filtrar líneas de log/debug
            if any(skip in line.lower() for skip in 
                   ['whisper', 'model', 'loading', 'processing', 'error', '[']):
                continue
            
            # Si la línea tiene contenido sustancial
            if len(line) > 5 and any(c.isalpha() for c in line):
                return line
        
        return ""
    
    def _estimate_confidence(self, text: str, whisper_data: Dict) -> float:
        """Estima la confianza de la transcripción"""
        confidence = 0.5  # Base
        
        # Factor 1: Longitud del texto (más texto = más confianza)
        if len(text) > 20:
            confidence += 0.2
        elif len(text) > 10:
            confidence += 0.1
        
        # Factor 2: Presencia de palabras del vocabulario de cobranza
        collections_words = sum(1 for word in self.collections_vocabulary 
                              if word.lower() in text.lower())
        if collections_words > 0:
            confidence += min(0.3, collections_words * 0.1)
        
        # Factor 3: Coherencia del texto (sin repeticiones excesivas)
        words = text.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            uniqueness_ratio = len(unique_words) / len(words)
            if uniqueness_ratio > 0.7:
                confidence += 0.1
        
        # Factor 4: Ausencia de caracteres extraños
        if all(c.isalnum() or c.isspace() or c in '.,!?¿¡' for c in text):
            confidence += 0.1
        
        # Factor 5: Información de segmentos si está disponible
        segments = whisper_data.get('segments', [])
        if segments:
            avg_prob = np.mean([seg.get('avg_logprob', -1.0) for seg in segments])
            # Convertir log prob a probabilidad aproximada
            if avg_prob > -0.5:
                confidence += 0.2
            elif avg_prob > -1.0:
                confidence += 0.1
        
        return min(1.0, confidence)
    
    def _cleanup_temp_file(self, file_path: str):
        """Elimina archivo temporal"""
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Error eliminando archivo temporal: {e}")
    
    def _add_to_cache(self, audio_hash: str, result: TranscriptionResult):
        """Agrega resultado a cache"""
        if len(self.transcription_cache) >= self.cache_max_size:
            # Remover entrada más antigua
            oldest_key = next(iter(self.transcription_cache))
            del self.transcription_cache[oldest_key]
        
        self.transcription_cache[audio_hash] = result
    
    def _update_metrics(self, result: TranscriptionResult):
        """Actualiza métricas de transcripción"""
        self.metrics["total_transcriptions"] += 1
        
        # Promedio móvil de tiempo de procesamiento
        alpha = 0.1
        self.metrics["average_processing_time"] = (
            (1 - alpha) * self.metrics["average_processing_time"] + 
            alpha * result.processing_time
        )
        
        # Promedio móvil de confianza
        self.metrics["average_confidence"] = (
            (1 - alpha) * self.metrics["average_confidence"] + 
            alpha * result.confidence
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de transcripción"""
        return {
            **self.metrics,
            "model_size": self.model_size,
            "language": self.language,
            "vocabulary_size": len(self.collections_vocabulary),
            "cache_size": len(self.transcription_cache)
        }
    
    def cleanup(self):
        """Limpia recursos y archivos temporales"""
        try:
            # Limpiar directorio temporal
            for temp_file in self.temp_dir.glob("audio_*.wav"):
                temp_file.unlink(missing_ok=True)
            
            logger.info("Recursos de STT limpiados")
            
        except Exception as e:
            logger.error(f"Error limpiando recursos STT: {e}")


class StreamingSTT:
    """STT streaming para transcripción en tiempo real"""
    
    def __init__(self, whisper_stt: WhisperSTT):
        self.stt_engine = whisper_stt
        self.audio_buffer = Queue(maxsize=10)
        self.is_processing = False
        self.processing_thread = None
        self.stop_event = Event()
        
        # Callbacks
        self.on_transcription = None
        self.on_error = None
        
        # Buffer para acumular audio corto
        self.accumulated_audio = []
        self.max_accumulation_duration = 3.0  # segundos
        self.current_accumulation_duration = 0.0
    
    def start_streaming(self):
        """Inicia transcripción streaming"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.stop_event.clear()
        
        self.processing_thread = Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("📋 Transcripción streaming iniciada")
    
    def stop_streaming(self):
        """Detiene transcripción streaming"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        self.stop_event.set()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        logger.info("🛑 Transcripción streaming detenida")
    
    def add_audio_chunk(self, audio_chunk: AudioChunk):
        """Agrega chunk de audio para transcripción"""
        try:
            self.audio_buffer.put_nowait(audio_chunk)
        except:
            # Buffer lleno, descartar chunk más antiguo
            try:
                self.audio_buffer.get_nowait()
                self.audio_buffer.put_nowait(audio_chunk)
            except:
                pass
    
    def _processing_loop(self):
        """Loop principal de procesamiento"""
        while self.is_processing and not self.stop_event.is_set():
            try:
                # Obtener chunk de audio con timeout
                try:
                    chunk = self.audio_buffer.get(timeout=1.0)
                except Empty:
                    continue
                
                # Acumular audio corto para transcripción más efectiva
                self.accumulated_audio.append(chunk)
                self.current_accumulation_duration += chunk.duration
                
                # Procesar cuando tengamos suficiente audio
                if (self.current_accumulation_duration >= self.max_accumulation_duration or
                    len(self.accumulated_audio) >= 5):
                    
                    self._process_accumulated_audio()
                
            except Exception as e:
                logger.error(f"Error en loop de procesamiento STT: {e}")
                if self.on_error:
                    self.on_error(e)
    
    def _process_accumulated_audio(self):
        """Procesa audio acumulado"""
        if not self.accumulated_audio:
            return
        
        try:
            # Combinar chunks de audio
            combined_chunk = self._combine_audio_chunks(self.accumulated_audio)
            
            # Transcribir
            result = self.stt_engine.transcribe(combined_chunk)
            
            if result and result.text.strip():
                # Emitir resultado
                if self.on_transcription:
                    self.on_transcription(result)
            
            # Limpiar buffer
            self.accumulated_audio = []
            self.current_accumulation_duration = 0.0
            
        except Exception as e:
            logger.error(f"Error procesando audio acumulado: {e}")
    
    def _combine_audio_chunks(self, chunks: List[AudioChunk]) -> AudioChunk:
        """Combina múltiples chunks en uno solo"""
        if not chunks:
            raise ValueError("No hay chunks para combinar")
        
        # Verificar que todos los chunks tienen la misma configuración
        first_chunk = chunks[0]
        
        combined_data = []
        total_duration = 0.0
        earliest_timestamp = min(chunk.timestamp for chunk in chunks)
        max_rms = 0.0
        
        for chunk in chunks:
            combined_data.append(chunk.data)
            total_duration += chunk.duration
            max_rms = max(max_rms, chunk.rms_level)
        
        # Concatenar datos de audio
        combined_audio = np.concatenate(combined_data, axis=0)
        
        return AudioChunk(
            data=combined_audio,
            timestamp=earliest_timestamp,
            sample_rate=first_chunk.sample_rate,
            channels=first_chunk.channels,
            duration=total_duration,
            source=first_chunk.source,
            rms_level=max_rms
        )


if __name__ == "__main__":
    # Test del STT
    import time
    from ..audio.capture import AudioCapture
    
    print("=== Test Speech-to-Text Local ===")
    
    # Verificar modelo Whisper
    try:
        stt = WhisperSTT()
        print(f"✅ Whisper STT inicializado: {stt.model_size}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPara instalar Whisper:")
        print("1. Descargar modelo: wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin")
        print("2. Colocar en: ./models/whisper/ggml-small.bin")
        exit(1)
    
    # Test con audio en vivo
    capture = AudioCapture()
    streaming = StreamingSTT(stt)
    
    def on_transcription(result):
        print(f"📋 '{result.text}' (confianza: {result.confidence:.2f}, tiempo: {result.processing_time:.1f}s)")
    
    streaming.on_transcription = on_transcription
    
    print("\n🎤 Iniciando transcripción en tiempo real (Ctrl+C para detener)...")
    
    try:
        # Iniciar componentes
        capture.start_capture()
        streaming.start_streaming()
        
        # Conectar audio capture con STT streaming
        def on_audio_chunk(chunk):
            streaming.add_audio_chunk(chunk)
        
        capture.on_audio_chunk = on_audio_chunk
        
        print("✅ Sistema iniciado. Habla para probar transcripción...")
        
        # Mantener activo y mostrar métricas
        while True:
            time.sleep(5)
            metrics = stt.get_metrics()
            print(f"📈 Transcripciones: {metrics['total_transcriptions']}, "
                  f"Tiempo promedio: {metrics['average_processing_time']:.1f}s, "
                  f"Confianza promedio: {metrics['average_confidence']:.2f}")
    
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo sistema...")
    
    finally:
        streaming.stop_streaming()
        capture.stop_capture()
        stt.cleanup()
        
        # Métricas finales
        final_metrics = stt.get_metrics()
        print(f"\n📈 Métricas finales:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value}")
    
    print("🏁 Test completado")
