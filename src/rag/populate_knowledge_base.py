"""
Población inicial de la Knowledge Base

Este script puebla la base de datos ChromaDB con ejemplos iniciales de:
- Situaciones AEIOU exitosas
- Patrones de conflicto comunes
- Respuestas efectivas categorizadas
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from chroma_manager import ChromaManager
from rich.progress import Progress, TaskID
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def get_initial_aeiou_examples():
    """Devuelve ejemplos iniciales de respuestas AEIOU"""
    return [
        {
            "content": "Entiendo que sientes que no estoy captando el punto principal (A). Yo percibo que podríamos estar enfocándonos en aspectos diferentes (E). ¿Podrías ayudarme a entender qué aspecto específico te preocupa más? (I) Mi objetivo es que ambos estemos alineados en la dirección del proyecto (O). ¿Qué información adicional necesitas de mi parte? (U)",
            "metadata": {
                "category": "communication_breakdown",
                "effectiveness_score": 0.92,
                "context_type": "technical_meeting",
                "aeiou_components": ["acknowledge", "express", "identify", "outcome", "understanding"],
                "industry": "software_development",
                "team_size": "small",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                "situation": "Malentendido sobre objetivos del proyecto"
            }
        },
        {
            "content": "Reconozco que el deadline está generando presión en el equipo (A). Yo siento que necesitamos ser realistas sobre lo que podemos entregar (E). Propongo que hagamos una sesión de re-priorización para identificar lo crítico (I). El resultado que busco es entregar calidad sin burnout del equipo (O). ¿Cómo ves que podríamos ajustar el scope? (U)",
            "metadata": {
                "category": "deadline_conflicts",
                "effectiveness_score": 0.89,
                "context_type": "project_management",
                "aeiou_components": ["acknowledge", "express", "identify", "outcome", "understanding"],
                "industry": "software_development",
                "team_size": "medium",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                "situation": "Presión por deadline poco realista"
            }
        },
        {
            "content": "Entiendo que tienes una perspectiva diferente sobre la implementación técnica (A). Yo creo que ambos enfoques tienen mérito (E). ¿Qué te parece si hacemos un spike de 2 horas para probar ambas opciones? (I) Mi objetivo es que tomemos la mejor decisión basada en datos (O). ¿Qué métricas crees que deberíamos evaluar? (U)",
            "metadata": {
                "category": "technical_disagreement",
                "effectiveness_score": 0.94,
                "context_type": "technical_discussion",
                "aeiou_components": ["acknowledge", "express", "identify", "outcome", "understanding"],
                "industry": "software_development",
                "team_size": "small",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                "situation": "Desacuerdo sobre enfoque técnico"
            }
        },
        {
            "content": "Reconozco que el feedback que recibiste fue difícil de procesar (A). Yo entiendo que puede sentirse abrumador (E). Sugiero que nos enfoquemos en 1-2 puntos específicos para mejorar gradualmente (I). Mi objetivo es que te sientas apoyado en tu crecimiento (O). ¿Con cuál área te gustaría empezar? (U)",
            "metadata": {
                "category": "difficult_feedback",
                "effectiveness_score": 0.87,
                "context_type": "performance_review",
                "aeiou_components": ["acknowledge", "express", "identify", "outcome", "understanding"],
                "industry": "general",
                "team_size": "one_on_one",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                "situation": "Dar feedback constructivo"
            }
        },
        {
            "content": "Entiendo que sientes que tu opinión no está siendo escuchada en las reuniones (A). Yo valoro tu perspectiva y quiero asegurarme de que tengas espacio (E). Propongo que asignemos tiempo específico para tu input en la agenda (I). Mi objetivo es que todos se sientan incluidos en las decisiones (O). ¿Qué formato te ayudaría a sentirte más cómodo participando? (U)",
            "metadata": {
                "category": "team_dynamics",
                "effectiveness_score": 0.91,
                "context_type": "team_meeting",
                "aeiou_components": ["acknowledge", "express", "identify", "outcome", "understanding"],
                "industry": "general",
                "team_size": "medium",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                "situation": "Miembro del equipo se siente excluido"
            }
        }
    ]


def get_conflict_patterns():
    """Devuelve patrones comunes de conflicto y sus características"""
    return [
        {
            "content": "Patrón: Interrupciones frecuentes durante presentaciones. Señales: Voz elevada, hablar por encima, gestos de impaciencia. Resolución efectiva: Establecer reglas claras de participación y usar técnica de 'parking lot' para ideas.",
            "metadata": {
                "category": "interruption_pattern",
                "effectiveness_score": 0.85,
                "context_type": "presentation",
                "triggers": ["interruption", "overlapping_speech", "impatience"],
                "resolution_techniques": ["ground_rules", "parking_lot", "facilitation"],
                "created_at": datetime.now().isoformat(),
                "usage_count": 0
            }
        },
        {
            "content": "Patrón: Desacuerdo sobre prioridades entre stakeholders. Señales: Uso de palabras absolutas ('siempre', 'nunca'), referencias a decisiones pasadas. Resolución efectiva: Facilitar sesión de alineación con criterios objetivos.",
            "metadata": {
                "category": "priority_conflict",
                "effectiveness_score": 0.88,
                "context_type": "stakeholder_meeting",
                "triggers": ["absolute_words", "past_references", "conflicting_priorities"],
                "resolution_techniques": ["objective_criteria", "facilitated_alignment", "stakeholder_mapping"],
                "created_at": datetime.now().isoformat(),
                "usage_count": 0
            }
        },
        {
            "content": "Patrón: Frustración por falta de progreso en decisiones. Señales: Suspiros, referencias repetidas al tiempo perdido, propuestas de 'simplemente decidir'. Resolución efectiva: Clarificar proceso de toma de decisiones y roles.",
            "metadata": {
                "category": "decision_paralysis",
                "effectiveness_score": 0.83,
                "context_type": "decision_meeting",
                "triggers": ["time_pressure", "repeated_discussions", "unclear_process"],
                "resolution_techniques": ["decision_framework", "role_clarification", "timeline_setting"],
                "created_at": datetime.now().isoformat(),
                "usage_count": 0
            }
        }
    ]


def get_communication_styles():
    """Devuelve ejemplos de estilos de comunicación efectiva"""
    return [
        {
            "content": "Estilo Asertivo-Colaborativo: Expresar necesidades claramente mientras se busca solución mutua. Ejemplo: 'Necesito que tengamos claridad en los roles (necesidad) y me gustaría explorar cómo podemos definirlos juntos (colaboración)'.",
            "metadata": {
                "category": "assertive_collaborative",
                "effectiveness_score": 0.92,
                "context_type": "general",
                "characteristics": ["clear_needs", "mutual_solutions", "respectful_tone"],
                "best_for": ["role_clarification", "boundary_setting", "problem_solving"],
                "created_at": datetime.now().isoformat(),
                "usage_count": 0
            }
        },
        {
            "content": "Estilo Empático-Investigativo: Primero entender la perspectiva del otro antes de presentar la propia. Ejemplo: 'Help me understand your perspective on this... (pausa para escuchar) ... desde mi punto de vista...'.",
            "metadata": {
                "category": "empathetic_investigative",
                "effectiveness_score": 0.89,
                "context_type": "conflict_resolution",
                "characteristics": ["active_listening", "curiosity", "delayed_judgment"],
                "best_for": ["misunderstandings", "emotional_situations", "perspective_gaps"],
                "created_at": datetime.now().isoformat(),
                "usage_count": 0
            }
        }
    ]


def populate_knowledge_base():
    """Función principal para poblar la knowledge base"""
    console.print("[bold blue]🔧 Inicializando Knowledge Base Local[/bold blue]")
    
    # Inicializar ChromaManager
    chroma = ChromaManager()
    
    with Progress(console=console) as progress:
        # Task para ejemplos AEIOU
        aeiou_task = progress.add_task("[green]Ejemplos AEIOU...", total=len(get_initial_aeiou_examples()))
        
        for example in get_initial_aeiou_examples():
            doc_id = chroma.add_document(
                "aeiou_examples",
                example["content"],
                example["metadata"]
            )
            progress.advance(aeiou_task)
            
        # Task para patrones de conflicto
        conflict_task = progress.add_task("[red]Patrones de conflicto...", total=len(get_conflict_patterns()))
        
        for pattern in get_conflict_patterns():
            doc_id = chroma.add_document(
                "conflict_patterns",
                pattern["content"],
                pattern["metadata"]
            )
            progress.advance(conflict_task)
            
        # Task para estilos de comunicación
        style_task = progress.add_task("[yellow]Estilos de comunicación...", total=len(get_communication_styles()))
        
        for style in get_communication_styles():
            doc_id = chroma.add_document(
                "communication_styles",
                style["content"],
                style["metadata"]
            )
            progress.advance(style_task)
    
    # Mostrar estadísticas
    console.print("\n[bold green]✅ Knowledge Base poblada exitosamente![/bold green]")
    
    for collection_name in ["aeiou_examples", "conflict_patterns", "communication_styles"]:
        stats = chroma.get_collection_stats(collection_name)
        console.print(
            f"[cyan]{collection_name}[/cyan]: {stats['total_documents']} documentos, "
            f"efectividad promedio: {stats['average_effectiveness']:.2f}"
        )
    
    # Test de consulta
    console.print("\n[bold yellow]🔍 Test de consulta RAG:[/bold yellow]")
    test_query = "deadline pressure in technical meeting"
    results = chroma.query_similar("aeiou_examples", test_query, n_results=2)
    
    console.print(f"[dim]Consulta:[/dim] {test_query}")
    for i, doc in enumerate(results["documents"]):
        score = results["metadatas"][i].get("effectiveness_score", 0)
        console.print(f"[green]Resultado {i+1}[/green] (score: {score}): {doc[:100]}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    populate_knowledge_base()
