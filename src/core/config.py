"""
Configuración global del AI Meeting Assistant Lean

Centraliza toda la configuración del sistema incluyendo:
- Parámetros de modelos
- Configuración de RAG
- Settings de audio
- Configuración de UI
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class ModelSize(Enum):
    """Tamaños de modelo disponibles"""
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class AudioConfig:
    """Configuración de captura de audio"""
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    buffer_duration: float = 2.0  # segundos
    device_name: str = None  # Auto-detect
    

@dataclass 
class RAGConfig:
    """Configuración del sistema RAG"""
    embedding_model: str = "all-MiniLM-L6-v2"  # Modelo de embeddings
    chroma_db_path: str = "./data/chroma"  # Path de ChromaDB
    max_query_results: int = 3  # Resultados máximos por query
    similarity_threshold: float = 0.7  # Umbral de similitud
    auto_update_effectiveness: bool = True  # Auto-actualizar scores
    

@dataclass
class LLMConfig:
    """Configuración del modelo de lenguaje"""
    model_name: str = "qwen2.5:0.5b"  # Modelo Ollama
    temperature: float = 0.3  # Creatividad vs consistencia
    max_tokens: int = 150  # Máximo tokens en respuesta
    timeout: int = 10  # Timeout en segundos
    

@dataclass
class STTConfig:
    """Configuración de Speech-to-Text"""
    model_size: ModelSize = ModelSize.SMALL
    language: str = "es"  # Idioma principal
    energy_threshold: int = 300  # Umbral de energía para voz
    pause_threshold: float = 0.8  # Pausa mínima entre frases
    

@dataclass
class UIConfig:
    """Configuración de interfaz de usuario"""
    overlay_position: str = "top-right"  # Posición del overlay
    overlay_opacity: float = 0.9  # Opacidad del overlay
    suggestion_timeout: int = 8  # Segundos antes de ocultar sugerencia
    theme: str = "dark"  # Tema de la UI
    font_size: int = 12  # Tamaño de fuente
    

@dataclass
class VoiceProfileConfig:
    """Configuración del perfil de voz"""
    profile_path: str = "./data/voice_profiles"  # Path de perfiles
    calibration_duration: int = 180  # Segundos de calibración
    similarity_threshold: float = 0.85  # Umbral para identificar usuario
    update_frequency: int = 10  # Actualizar perfil cada N usos
    

class AppConfig:
    """Configuración principal de la aplicación"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        
        # Crear directorios si no existen
        for directory in [self.data_dir, self.models_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Configuraciones por componente
        self.audio = AudioConfig()
        self.rag = RAGConfig()
        self.llm = LLMConfig()
        self.stt = STTConfig()
        self.ui = UIConfig()
        self.voice_profile = VoiceProfileConfig()
        
        # Configuración general
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.performance_monitoring = True
        
        # Configuración de características
        self.features = {
            "voice_recognition": True,
            "real_time_suggestions": True,
            "rag_contextual_search": True,
            "effectiveness_tracking": True,
            "auto_learning": True,
            "export_suggestions": True
        }
        
        # Load from environment or config file if exists
        self._load_from_env()
        
    def _load_from_env(self):
        """Carga configuración desde variables de entorno"""
        # Audio
        self.audio.sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", self.audio.sample_rate))
        self.audio.chunk_size = int(os.getenv("AUDIO_CHUNK_SIZE", self.audio.chunk_size))
        
        # RAG
        self.rag.embedding_model = os.getenv("RAG_EMBEDDING_MODEL", self.rag.embedding_model)
        self.rag.chroma_db_path = os.getenv("CHROMA_DB_PATH", self.rag.chroma_db_path)
        
        # LLM
        self.llm.model_name = os.getenv("LLM_MODEL_NAME", self.llm.model_name)
        self.llm.temperature = float(os.getenv("LLM_TEMPERATURE", self.llm.temperature))
        
        # STT
        stt_model_env = os.getenv("STT_MODEL_SIZE", self.stt.model_size.value)
        try:
            self.stt.model_size = ModelSize(stt_model_env)
        except ValueError:
            pass  # Keep default
            
        # Voice Profile
        self.voice_profile.similarity_threshold = float(
            os.getenv("VOICE_SIMILARITY_THRESHOLD", self.voice_profile.similarity_threshold)
        )
    
    def get_model_path(self, model_type: str, model_name: str) -> Path:
        """Obtiene la ruta completa de un modelo"""
        return self.models_dir / model_type / model_name
    
    def get_data_path(self, data_type: str) -> Path:
        """Obtiene la ruta completa de un directorio de datos"""
        return self.data_dir / data_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a diccionario"""
        return {
            "audio": self.audio.__dict__,
            "rag": self.rag.__dict__,
            "llm": self.llm.__dict__,
            "stt": {**self.stt.__dict__, "model_size": self.stt.model_size.value},
            "ui": self.ui.__dict__,
            "voice_profile": self.voice_profile.__dict__,
            "debug_mode": self.debug_mode,
            "log_level": self.log_level,
            "features": self.features
        }
    
    def save_to_file(self, config_path: str = None):
        """Guarda la configuración a un archivo JSON"""
        import json
        
        if not config_path:
            config_path = self.data_dir / "config.json"
            
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, config_path: str):
        """Carga configuración desde un archivo JSON"""
        import json
        
        config = cls()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Actualizar configuraciones
            for section, values in data.items():
                if hasattr(config, section) and isinstance(values, dict):
                    section_config = getattr(config, section)
                    for key, value in values.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
                elif hasattr(config, section):
                    setattr(config, section, values)
                    
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default configuration.")
        
        return config


# Instancia global de configuración
config = AppConfig()


if __name__ == "__main__":
    # Test de configuración
    print("=== AI Meeting Assistant Lean - Configuration ===")
    print(f"Base directory: {config.base_dir}")
    print(f"Data directory: {config.data_dir}")
    print(f"Models directory: {config.models_dir}")
    print(f"Debug mode: {config.debug_mode}")
    
    print("\n=== Audio Config ===")
    print(f"Sample rate: {config.audio.sample_rate}")
    print(f"Chunk size: {config.audio.chunk_size}")
    
    print("\n=== RAG Config ===")
    print(f"Embedding model: {config.rag.embedding_model}")
    print(f"ChromaDB path: {config.rag.chroma_db_path}")
    print(f"Max query results: {config.rag.max_query_results}")
    
    print("\n=== LLM Config ===")
    print(f"Model: {config.llm.model_name}")
    print(f"Temperature: {config.llm.temperature}")
    
    print("\n=== Features ===")
    for feature, enabled in config.features.items():
        status = "✅" if enabled else "❌"
        print(f"{status} {feature}")
    
    # Guardar configuración de ejemplo
    config.save_to_file()
    print(f"\n📁 Configuración guardada en: {config.data_dir / 'config.json'}")
