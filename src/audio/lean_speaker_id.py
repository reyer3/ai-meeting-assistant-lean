"""
Identificación de voz optimizada para desarrollo lean
Enfoque: Diferencia tu voz de otros participantes usando embeddings simples pero efectivos
"""

import asyncio
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
from loguru import logger


class SpeakerIdentifier:
    """
    Identificador de voz lean usando embeddings simples
    Optimizado para diferenciación rápida sin modelos complejos
    """
    
    def __init__(self, profile_path: str = "data/user_voice_profile.pkl"):
        self.profile_path = Path(profile_path)
        self.user_embeddings: Optional[np.ndarray] = None
        self.sample_rate = 16000
        
        # Configuración lean
        self.similarity_threshold = 0.7  # Umbral de similitud
        self.confidence_factor = 1.2  # Factor de ajuste de confianza
        
        logger.info(f"🎯 SpeakerIdentifier lean inicializado")
    
    async def load_user_profile(self, profile_path: Optional[str] = None) -> bool:
        """Carga el perfil de voz del usuario"""
        
        if profile_path:
            self.profile_path = Path(profile_path)
        
        if not self.profile_path.exists():
            logger.error(f"❌ Perfil de voz no encontrado: {self.profile_path}")
            logger.info("💡 Ejecuta: python scripts/setup_voice_profile.py")
            return False
        
        try:
            with open(self.profile_path, 'rb') as f:
                profile_data = pickle.load(f)
            
            # Convertir embeddings a numpy array
            embeddings = np.array(profile_data['embeddings'])
            
            if len(embeddings) == 0:
                logger.error("❌ Perfil de voz vacío")
                return False
            
            # Calcular embedding promedio del usuario
            self.user_embeddings = np.mean(embeddings, axis=0)
            
            logger.success(f"✅ Perfil cargado: {len(embeddings)} muestras")
            logger.info(f"📊 Dimensiones del embedding: {self.user_embeddings.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando perfil: {e}")
            return False
    
    async def identify_speaker(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Identifica si el audio corresponde al usuario
        
        Returns:
            Dict con información de identificación:
            - is_user_voice: bool
            - confidence: float (0-1)
            - embedding_distance: float
            - processing_time: float
        """
        
        start_time = time.time()
        
        try:
            if self.user_embeddings is None:
                logger.warning("⚠️ Perfil de usuario no cargado")
                return {
                    "is_user_voice": False,
                    "confidence": 0.0,
                    "embedding_distance": float('inf'),
                    "processing_time": time.time() - start_time,
                    "error": "profile_not_loaded"
                }
            
            # Crear embedding del audio actual
            current_embedding = self._create_voice_embedding(audio_data)
            
            # Calcular similitud con perfil del usuario
            distance = self._calculate_embedding_distance(
                current_embedding, 
                self.user_embeddings
            )
            
            # Determinar si es el usuario basado en umbral
            is_user = distance <= self.similarity_threshold
            
            # Calcular confianza (inversa de la distancia normalizada)
            confidence = max(0.0, min(1.0, 
                (self.similarity_threshold - distance) / self.similarity_threshold * self.confidence_factor
            ))
            
            processing_time = time.time() - start_time
            
            result = {
                "is_user_voice": is_user,
                "confidence": confidence,
                "embedding_distance": distance,
                "processing_time": processing_time
            }
            
            if is_user:
                logger.debug(f"👤 Usuario detectado (confianza: {confidence:.3f}, distancia: {distance:.3f})")
            else:
                logger.debug(f"👥 Otra persona (confianza: {confidence:.3f}, distancia: {distance:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en identificación: {e}")
            return {
                "is_user_voice": False,
                "confidence": 0.0,
                "embedding_distance": float('inf'),
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _create_voice_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Crea embedding de voz usando características simples pero efectivas
        Misma implementación que en setup_voice_profile.py para consistencia
        """
        
        # Asegurar que el audio tenga la forma correcta
        if len(audio.shape) > 1:
            audio = audio.flatten()
        
        # Filtrar silencio
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.01:  # Audio muy bajo
            return np.zeros(13, dtype=np.float32)  # Embedding vacío
        
        features = []
        
        # 1. Características espectrales
        fft = np.fft.fft(audio)
        magnitude_spectrum = np.abs(fft[:len(fft)//2])
        
        # Energía en diferentes bandas de frecuencia (8 bandas)
        bands = np.array_split(magnitude_spectrum, 8)
        band_energies = [np.sum(band) for band in bands]
        features.extend(band_energies)
        
        # 2. Características temporales
        features.append(np.mean(audio))  # DC component
        features.append(np.std(audio))   # Variabilidad
        features.append(np.max(audio))   # Peak level
        
        # 3. Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        features.append(zero_crossings / len(audio))
        
        # 4. Características de pitch (fundamental frequency)
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Encontrar picos en autocorrelación
        if len(autocorr) > 50:
            peak_idx = np.argmax(autocorr[20:200]) + 20  # Evitar DC
            fundamental_freq = self.sample_rate / peak_idx if peak_idx > 0 else 0
            features.append(fundamental_freq)
        else:
            features.append(0)
        
        # Normalizar características
        features = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(features)
        if norm > 1e-8:  # Evitar división por cero
            features = features / norm
        
        return features
    
    def _calculate_embedding_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calcula distancia entre embeddings"""
        
        # Asegurar que los embeddings tengan la misma dimensión
        if embedding1.shape != embedding2.shape:
            logger.warning(f"⚠️ Dimensiones diferentes: {embedding1.shape} vs {embedding2.shape}")
            return float('inf')
        
        # Usar distancia coseno (más robusta que euclidiana para embeddings)
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return float('inf')
        
        cosine_similarity = dot_product / (norm1 * norm2)
        cosine_distance = 1 - cosine_similarity  # Convertir similitud a distancia
        
        return float(cosine_distance)
    
    def update_threshold(self, new_threshold: float):
        """Actualiza el umbral de similitud dinámicamente"""
        old_threshold = self.similarity_threshold
        self.similarity_threshold = max(0.1, min(0.9, new_threshold))
        
        logger.info(f"🎯 Umbral actualizado: {old_threshold:.3f} → {self.similarity_threshold:.3f}")
    
    def get_profile_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del perfil cargado"""
        
        if self.user_embeddings is None:
            return {"profile_loaded": False}
        
        try:
            with open(self.profile_path, 'rb') as f:
                profile_data = pickle.load(f)
            
            return {
                "profile_loaded": True,
                "profile_path": str(self.profile_path),
                "samples_count": len(profile_data['embeddings']),
                "embedding_dimensions": len(self.user_embeddings),
                "sample_rate": profile_data.get('sample_rate', self.sample_rate),
                "created_at": profile_data.get('created_at', 0),
                "version": profile_data.get('version', 'unknown'),
                "current_threshold": self.similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo stats: {e}")
            return {"profile_loaded": True, "error": str(e)}
    
    async def test_identification(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Prueba la identificación con audio de test"""
        
        result = await self.identify_speaker(audio_data)
        
        # Agregar información adicional para debugging
        if self.user_embeddings is not None:
            current_embedding = self._create_voice_embedding(audio_data)
            
            result.update({
                "current_embedding_norm": float(np.linalg.norm(current_embedding)),
                "user_embedding_norm": float(np.linalg.norm(self.user_embeddings)),
                "threshold_used": self.similarity_threshold,
                "audio_rms": float(np.sqrt(np.mean(audio_data**2))),
                "audio_length": len(audio_data)
            })
        
        return result


# Factory function para configuración lean
def create_lean_speaker_id(profile_path: str = "data/user_voice_profile.pkl") -> SpeakerIdentifier:
    """Factory para crear identificador con configuración lean"""
    return SpeakerIdentifier(profile_path)


if __name__ == "__main__":
    """Test del identificador de voz"""
    
    async def test_speaker_identification():
        logger.info("🧪 Iniciando test de identificación de voz...")
        
        # Crear identificador
        speaker_id = create_lean_speaker_id()
        
        # Cargar perfil
        if not await speaker_id.load_user_profile():
            logger.error("❌ No se pudo cargar el perfil de voz")
            logger.info("💡 Ejecuta: python scripts/setup_voice_profile.py")
            return
        
        # Mostrar stats del perfil
        stats = speaker_id.get_profile_stats()
        logger.info(f"📊 Stats del perfil:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")
        
        # Test con audio sintético
        logger.info("🎤 Generando audio de prueba...")
        
        # Simular audio de prueba (ruido blanco)
        test_audio = np.random.normal(0, 0.1, 16000 * 2).astype(np.float32)  # 2 segundos
        
        # Test de identificación
        result = await speaker_id.test_identification(test_audio)
        
        logger.info(f"🎯 Resultado de identificación:")
        for key, value in result.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.4f}")
            else:
                logger.info(f"   {key}: {value}")
        
        logger.success("✅ Test completado")
    
    asyncio.run(test_speaker_identification())
