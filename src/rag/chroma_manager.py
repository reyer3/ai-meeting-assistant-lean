"""
ChromaDB Manager - Gestión de la base de datos vectorial local

Este módulo maneja todas las operaciones con ChromaDB incluyendo:
- Inicialización de la base de datos
- Inserción de documentos con embeddings
- Consultas vectoriales
- Gestión de metadatos
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Esquema de metadatos para documentos en ChromaDB"""
    category: str
    effectiveness_score: float
    context_type: str
    aeiou_components: List[str]
    industry: Optional[str] = None
    team_size: Optional[str] = None
    created_at: str
    usage_count: int = 0


class ChromaManager:
    """Gestor de ChromaDB para el knowledge base local"""
    
    def __init__(self, data_dir: str = "./data/chroma", model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el gestor de ChromaDB
        
        Args:
            data_dir: Directorio donde se almacenará la base de datos
            model_name: Modelo de sentence-transformers para embeddings
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar modelo de embeddings local
        logger.info(f"Cargando modelo de embeddings: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Configurar ChromaDB en modo persistente local
        self.client = chromadb.PersistentClient(
            path=str(self.data_dir),
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Colecciones principales
        self.collections = {}
        self._init_collections()
        
    def _init_collections(self):
        """Inicializa las colecciones principales de ChromaDB"""
        collection_configs = {
            "aeiou_examples": "Ejemplos de respuestas AEIOU exitosas",
            "conflict_patterns": "Patrones de conflicto y resoluciones",
            "communication_styles": "Estilos de comunicación efectiva",
            "meeting_contexts": "Contextos específicos de reuniones",
            "feedback_loops": "Casos de feedback constructivo"
        }
        
        for name, description in collection_configs.items():
            try:
                collection = self.client.get_or_create_collection(
                    name=name,
                    metadata={"description": description}
                )
                self.collections[name] = collection
                logger.info(f"Colección '{name}' inicializada")
            except Exception as e:
                logger.error(f"Error inicializando colección {name}: {e}")
    
    def add_document(
        self, 
        collection_name: str,
        content: str, 
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None
    ) -> str:
        """
        Añade un documento a la colección especificada
        
        Args:
            collection_name: Nombre de la colección
            content: Contenido del documento
            metadata: Metadatos del documento
            doc_id: ID único del documento (opcional)
            
        Returns:
            ID del documento añadido
        """
        if collection_name not in self.collections:
            raise ValueError(f"Colección {collection_name} no existe")
            
        # Generar embedding del contenido
        embedding = self.embedding_model.encode(content).tolist()
        
        # Generar ID si no se proporciona
        if not doc_id:
            doc_id = f"{collection_name}_{len(self.collections[collection_name].get()['ids']) + 1:04d}"
        
        # Añadir documento
        self.collections[collection_name].add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        logger.info(f"Documento {doc_id} añadido a {collection_name}")
        return doc_id
    
    def query_similar(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 3,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Busca documentos similares en la colección
        
        Args:
            collection_name: Nombre de la colección
            query_text: Texto de consulta
            n_results: Número de resultados a devolver
            where_filter: Filtros adicionales de metadatos
            
        Returns:
            Resultados de la consulta
        """
        if collection_name not in self.collections:
            raise ValueError(f"Colección {collection_name} no existe")
        
        # Generar embedding de la consulta
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Realizar consulta
        results = self.collections[collection_name].query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        return {
            "query": query_text,
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }
    
    def update_effectiveness_score(self, collection_name: str, doc_id: str, new_score: float):
        """
        Actualiza el score de efectividad de un documento
        
        Args:
            collection_name: Nombre de la colección
            doc_id: ID del documento
            new_score: Nuevo score de efectividad
        """
        try:
            # Obtener documento actual
            result = self.collections[collection_name].get(ids=[doc_id])
            
            if not result["metadatas"]:
                logger.warning(f"Documento {doc_id} no encontrado")
                return
                
            # Actualizar metadata
            current_metadata = result["metadatas"][0]
            current_metadata["effectiveness_score"] = new_score
            current_metadata["usage_count"] = current_metadata.get("usage_count", 0) + 1
            
            # Actualizar en ChromaDB
            self.collections[collection_name].update(
                ids=[doc_id],
                metadatas=[current_metadata]
            )
            
            logger.info(f"Score actualizado para {doc_id}: {new_score}")
            
        except Exception as e:
            logger.error(f"Error actualizando score de {doc_id}: {e}")
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas de una colección
        
        Args:
            collection_name: Nombre de la colección
            
        Returns:
            Estadísticas de la colección
        """
        if collection_name not in self.collections:
            return {"error": f"Colección {collection_name} no existe"}
            
        collection = self.collections[collection_name]
        all_data = collection.get()
        
        total_docs = len(all_data["ids"])
        
        # Calcular estadísticas de efectividad
        effectiveness_scores = [
            metadata.get("effectiveness_score", 0.0) 
            for metadata in all_data["metadatas"]
        ]
        
        avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0
        
        # Categorías más comunes
        categories = [metadata.get("category", "unknown") for metadata in all_data["metadatas"]]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "total_documents": total_docs,
            "average_effectiveness": round(avg_effectiveness, 3),
            "categories": category_counts,
            "collection_name": collection_name
        }
    
    def export_collection(self, collection_name: str, output_path: str):
        """
        Exporta una colección a un archivo JSON
        
        Args:
            collection_name: Nombre de la colección
            output_path: Ruta del archivo de salida
        """
        import json
        
        if collection_name not in self.collections:
            raise ValueError(f"Colección {collection_name} no existe")
            
        data = self.collections[collection_name].get()
        
        export_data = {
            "collection_name": collection_name,
            "total_documents": len(data["ids"]),
            "documents": []
        }
        
        for i, doc_id in enumerate(data["ids"]):
            export_data["documents"].append({
                "id": doc_id,
                "content": data["documents"][i],
                "metadata": data["metadatas"][i]
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Colección {collection_name} exportada a {output_path}")


if __name__ == "__main__":
    # Test básico
    chroma = ChromaManager()
    
    # Ejemplo de uso
    sample_metadata = {
        "category": "deadline_conflicts",
        "effectiveness_score": 0.85,
        "context_type": "technical_meeting",
        "aeiou_components": ["acknowledge", "express"],
        "created_at": "2024-06-05"
    }
    
    doc_id = chroma.add_document(
        "aeiou_examples",
        "Entiendo que sientes presión por el deadline. Yo percibo que necesitamos priorizar.",
        sample_metadata
    )
    
    # Consulta de prueba
    results = chroma.query_similar(
        "aeiou_examples",
        "deadline pressure meeting",
        n_results=1
    )
    
    print(f"Resultado: {results}")
