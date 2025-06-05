"""
Speaker Identification Module

Este módulo identifica quién está hablando en tiempo real:
- Agente de cobranza vs Cliente
- Múltiples agentes en un call center
- Perfiles de voz personalizados
"""

import os
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from ..core.config import config
from .capture import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """Perfil de voz de un speaker"""
    speaker_id: str
    name: str
    embeddings: List[np.ndarray]
    mean_embedding: np.ndarray
    confidence_threshold: float
    created_at: datetime
    last_updated: datetime
    usage_count: int = 0
    accuracy_score: float = 0.0
    role: str = "unknown"  # "agent", "client", "supervisor"


@dataclass
class SpeakerResult:
    """Resultado de identificación de speaker"""
    speaker_id: str
    speaker_name: str
    confidence: float
    is_agent: bool
    embedding: np.ndarray
    audio_quality: float


class SpeakerIdentifier:
    """Identificador de speakers usando embeddings de voz"""
    
    def __init__(self, profile_path: str = None, threshold: float = None):
        self.profile_path = Path(profile_path or config.voice_profile.profile_path)
        self.threshold = threshold or config.voice_profile.similarity_threshold
        
        # Crear directorio de perfiles si no existe
        self.profile_path.mkdir(parents=True, exist_ok=True)
        
        # Inicializar encoder de voz
        logger.info("Cargando modelo Resemblyzer...")
        self.encoder = VoiceEncoder()
        
        # Cargar perfiles existentes
        self.profiles: Dict[str, VoiceProfile] = {}
        self._load_existing_profiles()
        
        # Configuración de clustering para speakers desconocidos
        self.unknown_embeddings = []
        self.clustering_enabled = True
        self.min_samples_for_clustering = 10
        
        # Cache de embeddings recientes para optimización
        self.embedding_cache = {}
        self.cache_max_size = 100
        
        logger.info(f"SpeakerIdentifier inicializado con {len(self.profiles)} perfiles")
    
    def _load_existing_profiles(self):
        """Carga perfiles de voz existentes"""
        try:
            for profile_file in self.profile_path.glob("*.json"):
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                
                # Cargar embeddings desde archivo separado
                embeddings_file = self.profile_path / f"{profile_file.stem}_embeddings.pkl"
                if embeddings_file.exists():
                    with open(embeddings_file, 'rb') as f:
                        embeddings_data = pickle.load(f)
                    
                    profile = VoiceProfile(
                        speaker_id=profile_data['speaker_id'],
                        name=profile_data['name'],
                        embeddings=embeddings_data['embeddings'],
                        mean_embedding=embeddings_data['mean_embedding'],
                        confidence_threshold=profile_data['confidence_threshold'],
                        created_at=datetime.fromisoformat(profile_data['created_at']),
                        last_updated=datetime.fromisoformat(profile_data['last_updated']),
                        usage_count=profile_data.get('usage_count', 0),
                        accuracy_score=profile_data.get('accuracy_score', 0.0),
                        role=profile_data.get('role', 'unknown')
                    )
                    
                    self.profiles[profile.speaker_id] = profile
                    logger.info(f"Perfil cargado: {profile.name} ({profile.speaker_id})")
                
        except Exception as e:
            logger.error(f"Error cargando perfiles: {e}")
    
    def create_voice_profile(self, 
                           speaker_id: str, 
                           speaker_name: str, 
                           audio_samples: List[AudioChunk],
                           role: str = "agent") -> bool:
        """Crea un nuevo perfil de voz"""
        try:
            logger.info(f"Creando perfil para {speaker_name} ({speaker_id})...")
            
            # Procesar muestras de audio
            embeddings = []
            for chunk in audio_samples:
                embedding = self._compute_embedding(chunk)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if len(embeddings) < 3:
                logger.error(f"Insuficientes muestras de audio válidas: {len(embeddings)}")
                return False
            
            # Calcular embedding promedio
            mean_embedding = np.mean(embeddings, axis=0)
            
            # Calcular threshold adaptivo basado en variabilidad interna
            internal_distances = []
            for emb in embeddings:
                distance = cosine(emb, mean_embedding)
                internal_distances.append(distance)
            
            # Threshold = media + 2 * desviación estándar
            mean_distance = np.mean(internal_distances)
            std_distance = np.std(internal_distances)
            adaptive_threshold = min(self.threshold, mean_distance + 2 * std_distance)
            
            # Crear perfil
            profile = VoiceProfile(
                speaker_id=speaker_id,
                name=speaker_name,
                embeddings=embeddings,
                mean_embedding=mean_embedding,
                confidence_threshold=adaptive_threshold,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                role=role
            )
            
            # Guardar perfil
            self._save_profile(profile)
            self.profiles[speaker_id] = profile
            
            logger.info(f"✅ Perfil creado para {speaker_name} (threshold: {adaptive_threshold:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Error creando perfil: {e}")
            return False
    
    def identify_speaker(self, audio_chunk: AudioChunk) -> SpeakerResult:
        """Identifica el speaker en un chunk de audio"""
        try:
            # Calcular embedding del audio
            embedding = self._compute_embedding(audio_chunk)
            if embedding is None:
                return self._unknown_speaker_result()
            
            # Buscar mejor match entre perfiles existentes
            best_match = None
            best_distance = float('inf')
            
            for profile in self.profiles.values():
                distance = cosine(embedding, profile.mean_embedding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = profile
            
            # Verificar si el match supera el threshold
            if best_match and best_distance < best_match.confidence_threshold:
                confidence = 1.0 - best_distance
                
                # Actualizar estadísticas del perfil
                self._update_profile_stats(best_match, embedding, confidence)
                
                return SpeakerResult(
                    speaker_id=best_match.speaker_id,
                    speaker_name=best_match.name,
                    confidence=confidence,
                    is_agent=best_match.role == "agent",
                    embedding=embedding,
                    audio_quality=self._assess_audio_quality(audio_chunk)
                )
            
            # Speaker desconocido
            if self.clustering_enabled:
                self._add_unknown_embedding(embedding)
            
            return self._unknown_speaker_result(embedding, audio_chunk)
            
        except Exception as e:
            logger.error(f"Error identificando speaker: {e}")
            return self._unknown_speaker_result()
    
    def _compute_embedding(self, audio_chunk: AudioChunk) -> Optional[np.ndarray]:
        """Calcula embedding de voz para un chunk de audio"""
        try:
            # Convertir audio a formato esperado por Resemblyzer
            audio_data = audio_chunk.data.flatten()
            
            # Verificar duración mínima (al menos 0.5 segundos)
            if len(audio_data) < audio_chunk.sample_rate * 0.5:
                return None
            
            # Verificar nivel de audio (evitar silencio)
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 0.01:  # Muy bajo nivel
                return None
            
            # Preprocesar audio
            preprocessed = preprocess_wav(audio_data, audio_chunk.sample_rate)
            
            # Verificar cache
            audio_hash = hash(audio_data.tobytes())
            if audio_hash in self.embedding_cache:
                return self.embedding_cache[audio_hash]
            
            # Calcular embedding
            embedding = self.encoder.embed_utterance(preprocessed)
            
            # Agregar a cache
            if len(self.embedding_cache) >= self.cache_max_size:
                # Remover entrada más antigua
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
            
            self.embedding_cache[audio_hash] = embedding
            
            return embedding
            
        except Exception as e:
            logger.debug(f"Error computando embedding: {e}")
            return None
    
    def _assess_audio_quality(self, audio_chunk: AudioChunk) -> float:
        """Evalúa la calidad del audio para identificación"""
        try:
            audio_data = audio_chunk.data.flatten()
            
            # Métricas de calidad
            rms = np.sqrt(np.mean(audio_data**2))
            peak = np.max(np.abs(audio_data))
            
            # Relación señal-ruido estimada
            signal_power = np.mean(audio_data**2)
            noise_floor = np.percentile(np.abs(audio_data), 10)**2
            snr = 10 * np.log10(signal_power / max(noise_floor, 1e-10))
            
            # Score de calidad (0.0 a 1.0)
            quality_score = 0.0
            
            # Factor 1: Nivel de señal apropiado
            if 0.01 <= rms <= 0.5:
                quality_score += 0.3
            
            # Factor 2: No hay clipping
            if peak < 0.95:
                quality_score += 0.2
            
            # Factor 3: SNR decente
            if snr > 10:
                quality_score += 0.3
            elif snr > 5:
                quality_score += 0.15
            
            # Factor 4: Duración suficiente
            if audio_chunk.duration >= 1.0:
                quality_score += 0.2
            elif audio_chunk.duration >= 0.5:
                quality_score += 0.1
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.debug(f"Error evaluando calidad de audio: {e}")
            return 0.5  # Calidad media por defecto
    
    def _update_profile_stats(self, profile: VoiceProfile, embedding: np.ndarray, confidence: float):
        """Actualiza estadísticas del perfil basado en uso"""
        try:
            profile.usage_count += 1
            
            # Actualizar accuracy score (promedio móvil)
            alpha = 0.1
            profile.accuracy_score = (1 - alpha) * profile.accuracy_score + alpha * confidence
            
            # Actualizar embeddings si la confianza es alta
            if confidence > 0.9 and len(profile.embeddings) < 20:
                profile.embeddings.append(embedding)
                # Recalcular embedding promedio
                profile.mean_embedding = np.mean(profile.embeddings, axis=0)
                profile.last_updated = datetime.now()
            
            # Guardar actualización cada 10 usos
            if profile.usage_count % 10 == 0:
                self._save_profile(profile)
                
        except Exception as e:
            logger.debug(f"Error actualizando estadísticas de perfil: {e}")
    
    def _add_unknown_embedding(self, embedding: np.ndarray):
        """Agrega embedding de speaker desconocido para clustering futuro"""
        self.unknown_embeddings.append(embedding)
        
        # Intentar clustering si tenemos suficientes muestras
        if len(self.unknown_embeddings) >= self.min_samples_for_clustering:
            self._cluster_unknown_speakers()
    
    def _cluster_unknown_speakers(self):
        """Agrupa speakers desconocidos usando clustering"""
        try:
            if len(self.unknown_embeddings) < self.min_samples_for_clustering:
                return
            
            logger.info(f"Intentando clustering de {len(self.unknown_embeddings)} embeddings desconocidos")
            
            # Reducir dimensionalidad para clustering más eficiente
            pca = PCA(n_components=50)
            embeddings_array = np.array(self.unknown_embeddings)
            embeddings_reduced = pca.fit_transform(embeddings_array)
            
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.3, min_samples=3, metric='cosine')
            clusters = dbscan.fit_predict(embeddings_reduced)
            
            # Analizar clusters encontrados
            unique_clusters = set(clusters) - {-1}  # Excluir ruido (-1)
            
            if unique_clusters:
                logger.info(f"Encontrados {len(unique_clusters)} posibles speakers nuevos")
                
                for cluster_id in unique_clusters:
                    cluster_indices = np.where(clusters == cluster_id)[0]
                    cluster_embeddings = [self.unknown_embeddings[i] for i in cluster_indices]
                    
                    if len(cluster_embeddings) >= 3:
                        # Crear perfil temporal para speaker desconocido
                        speaker_id = f"unknown_{cluster_id}_{int(datetime.now().timestamp())}"
                        logger.info(f"Creando perfil temporal para speaker: {speaker_id}")
                        
                        # Calcular embedding promedio
                        mean_embedding = np.mean(cluster_embeddings, axis=0)
                        
                        profile = VoiceProfile(
                            speaker_id=speaker_id,
                            name=f"Speaker Desconocido {cluster_id}",
                            embeddings=cluster_embeddings,
                            mean_embedding=mean_embedding,
                            confidence_threshold=self.threshold,
                            created_at=datetime.now(),
                            last_updated=datetime.now(),
                            role="unknown"
                        )
                        
                        self.profiles[speaker_id] = profile
                
                # Limpiar embeddings procesados
                self.unknown_embeddings = []
                
        except Exception as e:
            logger.error(f"Error en clustering de speakers: {e}")
    
    def _unknown_speaker_result(self, embedding: np.ndarray = None, audio_chunk: AudioChunk = None) -> SpeakerResult:
        """Crea resultado para speaker desconocido"""
        audio_quality = 0.0
        if audio_chunk:
            audio_quality = self._assess_audio_quality(audio_chunk)
        
        return SpeakerResult(
            speaker_id="unknown",
            speaker_name="Speaker Desconocido",
            confidence=0.0,
            is_agent=False,  # Asumir que es cliente por defecto
            embedding=embedding or np.zeros(256),
            audio_quality=audio_quality
        )
    
    def _save_profile(self, profile: VoiceProfile):
        """Guarda perfil de voz en disco"""
        try:
            # Guardar metadatos en JSON
            metadata = {
                'speaker_id': profile.speaker_id,
                'name': profile.name,
                'confidence_threshold': profile.confidence_threshold,
                'created_at': profile.created_at.isoformat(),
                'last_updated': profile.last_updated.isoformat(),
                'usage_count': profile.usage_count,
                'accuracy_score': profile.accuracy_score,
                'role': profile.role
            }
            
            metadata_file = self.profile_path / f"{profile.speaker_id}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Guardar embeddings en pickle
            embeddings_data = {
                'embeddings': profile.embeddings,
                'mean_embedding': profile.mean_embedding
            }
            
            embeddings_file = self.profile_path / f"{profile.speaker_id}_embeddings.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings_data, f)
                
        except Exception as e:
            logger.error(f"Error guardando perfil: {e}")
    
    def get_profile_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de todos los perfiles"""
        stats = {
            "total_profiles": len(self.profiles),
            "agents": len([p for p in self.profiles.values() if p.role == "agent"]),
            "clients": len([p for p in self.profiles.values() if p.role == "client"]),
            "unknown": len([p for p in self.profiles.values() if p.role == "unknown"]),
            "profiles": []
        }
        
        for profile in self.profiles.values():
            stats["profiles"].append({
                "speaker_id": profile.speaker_id,
                "name": profile.name,
                "role": profile.role,
                "usage_count": profile.usage_count,
                "accuracy_score": profile.accuracy_score,
                "embeddings_count": len(profile.embeddings),
                "threshold": profile.confidence_threshold
            })
        
        return stats
    
    def delete_profile(self, speaker_id: str) -> bool:
        """Elimina un perfil de voz"""
        try:
            if speaker_id not in self.profiles:
                return False
            
            # Eliminar archivos
            metadata_file = self.profile_path / f"{speaker_id}.json"
            embeddings_file = self.profile_path / f"{speaker_id}_embeddings.pkl"
            
            if metadata_file.exists():
                metadata_file.unlink()
            
            if embeddings_file.exists():
                embeddings_file.unlink()
            
            # Eliminar de memoria
            del self.profiles[speaker_id]
            
            logger.info(f"Perfil {speaker_id} eliminado")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando perfil: {e}")
            return False


if __name__ == "__main__":
    # Test del speaker identifier
    import time
    from .capture import AudioCapture
    
    print("=== Test Speaker Identification ===")
    
    # Inicializar componentes
    identifier = SpeakerIdentifier()
    capture = AudioCapture()
    
    # Mostrar perfiles existentes
    stats = identifier.get_profile_stats()
    print(f"\n📊 Perfiles existentes: {stats['total_profiles']}")
    for profile in stats['profiles']:
        print(f"  - {profile['name']} ({profile['role']}) - Usos: {profile['usage_count']}")
    
    print("\n🎤 Test de identificación en tiempo real (Ctrl+C para detener)...")
    
    def on_audio_chunk(chunk):
        result = identifier.identify_speaker(chunk)
        if result.confidence > 0.5:
            agent_indicator = "🧑‍💼" if result.is_agent else "👤"
            print(f"{agent_indicator} {result.speaker_name}: {result.confidence:.2f} (calidad: {result.audio_quality:.2f})")
    
    capture.on_audio_chunk = on_audio_chunk
    
    try:
        if capture.start_capture():
            print("✅ Iniciado. Habla para probar identificación...")
            
            while True:
                time.sleep(1)
        else:
            print("❌ Error iniciando captura")
            
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo...")
    finally:
        capture.stop_capture()
        
        # Mostrar estadísticas finales
        final_stats = identifier.get_profile_stats()
        print(f"\n📈 Estadísticas finales:")
        for profile in final_stats['profiles']:
            if profile['usage_count'] > 0:
                print(f"  {profile['name']}: {profile['usage_count']} usos, precisión: {profile['accuracy_score']:.2f}")
    
    print("🏁 Test completado")
