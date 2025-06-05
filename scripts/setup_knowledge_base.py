#!/usr/bin/env python3
"""
Setup de base de conocimiento para asistente lean
Inicializa ChromaDB con ejemplos AEIOU para reuniones de trabajo
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from loguru import logger

# Imports del sistema
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.chroma_manager import ChromaManager

console = Console()

class KnowledgeBaseSetup:
    """Setup de base de conocimiento AEIOU para reuniones"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.chroma_dir = self.data_dir / "chroma_db"
        self.collection_name = "aeiou_examples"
        
        self.chroma_manager = None
        
        # Ejemplos AEIOU categorizados por situación
        self.knowledge_examples = self._create_knowledge_examples()
    
    def _create_knowledge_examples(self) -> List[Dict[str, Any]]:
        """Crea ejemplos de conocimiento AEIOU para reuniones"""
        
        examples = [
            # Conflictos de implementación técnica
            {
                "id": "tech_disagreement_001",
                "situation": "Desacuerdo sobre arquitectura de software",
                "trigger_text": "No estoy de acuerdo con esta implementación",
                "aeiou_response": "Entiendo que tienes reservas sobre este enfoque (A). Yo también quiero asegurarme de que elijamos la mejor solución (E). ¿Podríamos revisar las alternativas que tienes en mente? (I) Mi objetivo es que lleguemos a una decisión técnica sólida (O). ¿Qué aspectos específicos te preocupan más? (U)",
                "category": "technical_dispute",
                "effectiveness_score": 0.89,
                "context": {
                    "meeting_type": "technical_review",
                    "urgency": "medium",
                    "team_size": "small"
                }
            },
            {
                "id": "tech_disagreement_002", 
                "situation": "Propuesta de cambio de herramienta rechazada",
                "trigger_text": "Esta herramienta no va a resolver el problema",
                "aeiou_response": "Reconozco que ves limitaciones en esta herramienta (A). Yo pienso que vale la pena explorar todas las opciones (E). ¿Qué tal si evaluamos pros y contras de diferentes alternativas? (I) Queremos encontrar la herramienta que realmente funcione para el equipo (O). ¿Qué criterios consideras más importantes? (U)",
                "category": "technical_dispute",
                "effectiveness_score": 0.92,
                "context": {
                    "meeting_type": "tool_selection",
                    "urgency": "low", 
                    "team_size": "medium"
                }
            },
            
            # Presión de deadlines
            {
                "id": "deadline_pressure_001",
                "situation": "Imposibilidad de cumplir fecha límite",
                "trigger_text": "Es imposible terminar esto para el deadline",
                "aeiou_response": "Entiendo que el timeline se siente muy ajustado (A). Yo también veo que hay muchas tareas por completar (E). ¿Podríamos priorizar las funcionalidades más críticas? (I) Mi objetivo es entregar algo valioso aunque no sea la versión completa (O). ¿Qué consideras absolutamente esencial para la primera entrega? (U)",
                "category": "deadline_pressure", 
                "effectiveness_score": 0.87,
                "context": {
                    "meeting_type": "sprint_planning",
                    "urgency": "high",
                    "team_size": "large"
                }
            },
            {
                "id": "deadline_pressure_002",
                "situation": "Solicitud de extensión de proyecto",
                "trigger_text": "Necesitamos más tiempo, no va a estar listo",
                "aeiou_response": "Veo que necesitas más tiempo para hacer un trabajo de calidad (A). Yo quiero balancear la calidad con los compromisos del negocio (E). ¿Podríamos definir qué partes son posibles entregar a tiempo? (I) Busco una solución que mantenga la calidad sin comprometer demasiado el cronograma (O). ¿Cuánto tiempo adicional necesitarías realmente? (U)",
                "category": "deadline_pressure",
                "effectiveness_score": 0.85,
                "context": {
                    "meeting_type": "project_review",
                    "urgency": "high",
                    "team_size": "small"
                }
            },
            
            # Problemas de comunicación
            {
                "id": "communication_001",
                "situation": "Malentendido sobre requerimientos",
                "trigger_text": "No entiendes lo que estoy pidiendo",
                "aeiou_response": "Reconozco que no estoy captando completamente tu punto (A). Yo quiero asegurarme de entenderte bien (E). ¿Podrías explicarme con un ejemplo específico? (I) Mi objetivo es que ambos tengamos claridad total sobre los requerimientos (O). ¿Qué aspecto sientes que no estoy comprendiendo? (U)",
                "category": "communication_breakdown",
                "effectiveness_score": 0.94,
                "context": {
                    "meeting_type": "requirements_review",
                    "urgency": "medium",
                    "team_size": "small"
                }
            },
            {
                "id": "communication_002",
                "situation": "Interrupción constante en reunión",
                "trigger_text": "Déjame terminar de explicar",
                "aeiou_response": "Tienes razón, necesitas espacio para completar tu idea (A). Yo estoy ansioso por contribuir pero reconozco que interrumpí (E). ¿Puedo escucharte completamente primero? (I) Quiero que tengamos una discusión productiva donde todos se sientan escuchados (O). ¿Prefieres que tome notas y comente al final? (U)",
                "category": "communication_breakdown",
                "effectiveness_score": 0.91,
                "context": {
                    "meeting_type": "brainstorming",
                    "urgency": "low",
                    "team_size": "large"
                }
            },
            
            # Conflictos de recursos
            {
                "id": "resource_conflict_001",
                "situation": "Falta de presupuesto para implementación",
                "trigger_text": "No tenemos presupuesto para eso",
                "aeiou_response": "Entiendo que el presupuesto es una limitación real (A). Yo también quiero ser responsable con los recursos (E). ¿Podríamos explorar alternativas más económicas o una implementación por fases? (I) Mi objetivo es encontrar una manera viable de avanzar dentro del presupuesto (O). ¿Qué alternativas estarías dispuesto a considerar? (U)",
                "category": "resource_conflict",
                "effectiveness_score": 0.88,
                "context": {
                    "meeting_type": "budget_review",
                    "urgency": "medium",
                    "team_size": "small"
                }
            },
            {
                "id": "resource_conflict_002",
                "situation": "Asignación de personal insuficiente",
                "trigger_text": "Mi equipo no tiene capacidad para esto",
                "aeiou_response": "Reconozco que tu equipo ya tiene mucha carga de trabajo (A). Yo también veo que necesitamos más manos para esto (E). ¿Podríamos analizar qué tareas podrían redistribuirse o postergarse? (I) Busco una forma de avanzar sin sobrecargar a nadie (O). ¿Qué apoyo necesitarías para hacer esto factible? (U)",
                "category": "resource_conflict",
                "effectiveness_score": 0.86,
                "context": {
                    "meeting_type": "resource_planning",
                    "urgency": "high",
                    "team_size": "large"
                }
            },
            
            # Feedback difícil
            {
                "id": "difficult_feedback_001",
                "situation": "Crítica a trabajo realizado",
                "trigger_text": "Este trabajo no cumple con los estándares",
                "aeiou_response": "Entiendo que el resultado no está a la altura de las expectativas (A). Yo también quiero entregar trabajo de alta calidad (E). ¿Podríamos revisar específicamente qué aspectos necesitan mejora? (I) Mi objetivo es corregir esto y establecer un proceso para evitar problemas similares (O). ¿Qué estándares específicos deberías estar cumpliendo? (U)",
                "category": "difficult_feedback",
                "effectiveness_score": 0.90,
                "context": {
                    "meeting_type": "performance_review",
                    "urgency": "medium",
                    "team_size": "small"
                }
            },
            {
                "id": "difficult_feedback_002",
                "situation": "Cuestionamiento de decisión tomada",
                "trigger_text": "No debimos haber tomado esa decisión",
                "aeiou_response": "Veo que tienes dudas sobre la decisión que tomamos (A). Yo también quiero evaluar si fue la correcta (E). ¿Podríamos analizar qué información nueva tenemos ahora? (I) Mi objetivo es aprender de esto y mejorar nuestro proceso de decisión (O). ¿Qué factores crees que no consideramos adecuadamente? (U)",
                "category": "difficult_feedback",
                "effectiveness_score": 0.89,
                "context": {
                    "meeting_type": "retrospective",
                    "urgency": "low",
                    "team_size": "medium"
                }
            },
            
            # Dinámicas de equipo
            {
                "id": "team_dynamics_001",
                "situation": "Conflicto entre miembros del equipo",
                "trigger_text": "Siempre tenemos problemas de coordinación",
                "aeiou_response": "Reconozco que hemos tenido desafíos de coordinación (A). Yo también noto que necesitamos mejorar nuestra colaboración (E). ¿Podríamos establecer algunas reglas claras de comunicación y seguimiento? (I) Mi objetivo es que el equipo funcione de manera más fluida (O). ¿Qué cambios crees que tendrían el mayor impacto positivo? (U)",
                "category": "team_dynamics",
                "effectiveness_score": 0.87,
                "context": {
                    "meeting_type": "team_building",
                    "urgency": "medium",
                    "team_size": "large"
                }
            },
            {
                "id": "team_dynamics_002",
                "situation": "Falta de participación en reuniones",
                "trigger_text": "Nadie participa en estas reuniones",
                "aeiou_response": "Entiendo tu frustración por la falta de participación (A). Yo también quiero que estas reuniones sean más productivas (E). ¿Podríamos probar un formato diferente que invite más a la participación? (I) Mi objetivo es que todos se sientan cómodos contribuyendo (O). ¿Qué crees que está inhibiendo la participación del equipo? (U)",
                "category": "team_dynamics",
                "effectiveness_score": 0.85,
                "context": {
                    "meeting_type": "team_meeting",
                    "urgency": "low",
                    "team_size": "large"
                }
            }
        ]
        
        return examples
    
    def show_intro(self):
        """Muestra introducción del setup"""
        console.print(Panel.fit(
            "[bold blue]📚 Setup de Base de Conocimiento AEIOU[/bold blue]\\n\\n"
            "Este proceso inicializará la base de conocimiento con:\\n"
            "• Ejemplos de situaciones comunes en reuniones\\n"
            "• Respuestas AEIOU efectivas y probadas\\n" 
            "• Contexto para generar sugerencias inteligentes\\n\\n"
            f"[dim]Se crearán {len(self.knowledge_examples)} ejemplos en ChromaDB[/dim]",
            border_style="blue"
        ))
    
    def setup_directories(self):
        """Crea directorios necesarios"""
        console.print("\\n[cyan]📁 Creando directorios...[/cyan]")
        
        self.data_dir.mkdir(exist_ok=True)
        self.chroma_dir.mkdir(exist_ok=True)
        
        console.print(f"[green]✅ Directorios creados en: {self.data_dir}[/green]")
    
    def initialize_chroma(self) -> bool:
        """Inicializa ChromaDB"""
        console.print("\\n[cyan]🗄️ Inicializando ChromaDB...[/cyan]")
        
        try:
            self.chroma_manager = ChromaManager(
                data_dir=str(self.chroma_dir),
                model_name="sentence-transformers/all-MiniLM-L6-v2"  # Modelo ligero
            )
            
            console.print("[green]✅ ChromaDB inicializado[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error inicializando ChromaDB: {e}[/red]")
            return False
    
    def populate_knowledge_base(self):
        """Puebla la base de conocimiento con ejemplos"""
        console.print("\\n[cyan]📝 Poblando base de conocimiento...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "Agregando ejemplos AEIOU...", 
                total=len(self.knowledge_examples)
            )
            
            # Crear o limpiar colección
            try:
                self.chroma_manager.delete_collection(self.collection_name)
            except:
                pass  # Colección no existe
            
            collection = self.chroma_manager.create_collection(self.collection_name)
            
            # Agregar ejemplos
            for example in self.knowledge_examples:
                
                # Preparar documento
                document_text = f"Situación: {example['situation']}. Trigger: {example['trigger_text']}. Respuesta AEIOU: {example['aeiou_response']}"
                
                # Preparar metadatos
                metadata = {
                    "category": example['category'],
                    "effectiveness_score": example['effectiveness_score'],
                    "meeting_type": example['context']['meeting_type'],
                    "urgency": example['context']['urgency'],
                    "team_size": example['context']['team_size'],
                    "created_at": time.time()
                }
                
                # Agregar a ChromaDB
                collection.add(
                    documents=[document_text],
                    metadatas=[metadata],
                    ids=[example['id']]
                )
                
                progress.update(task, advance=1)
                time.sleep(0.1)  # Pequeña pausa para mostrar progreso
        
        console.print("[green]✅ Base de conocimiento poblada[/green]")
    
    def verify_knowledge_base(self) -> bool:
        """Verifica que la base de conocimiento esté correcta"""
        console.print("\\n[cyan]🔍 Verificando base de conocimiento...[/cyan]")
        
        try:
            # Verificar colección
            collection = self.chroma_manager.get_collection(self.collection_name)
            count = collection.count()
            
            if count != len(self.knowledge_examples):
                console.print(f"[yellow]⚠️ Conteo incorrecto: {count} vs {len(self.knowledge_examples)}[/yellow]")
                return False
            
            # Hacer query de prueba
            test_results = collection.query(
                query_texts=["problemas de comunicación en reuniones"],
                n_results=3
            )
            
            if not test_results['documents']:
                console.print("[yellow]⚠️ Query de prueba no devolvió resultados[/yellow]")
                return False
            
            console.print(f"[green]✅ Base de conocimiento verificada ({count} ejemplos)[/green]")
            
            # Mostrar estadísticas
            categories = {}
            for example in self.knowledge_examples:
                cat = example['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            console.print("\\n[dim]Distribución por categoría:[/dim]")
            for category, count in categories.items():
                console.print(f"  {category}: {count} ejemplos")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error verificando: {e}[/red]")
            return False
    
    def create_config_file(self):
        """Crea archivo de configuración"""
        config_data = {
            "knowledge_base": {
                "version": "1.0",
                "created_at": time.time(),
                "total_examples": len(self.knowledge_examples),
                "collection_name": self.collection_name,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "categories": list(set(ex['category'] for ex in self.knowledge_examples))
        }
        
        config_path = self.data_dir / "knowledge_base_config.json"
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✅ Configuración guardada en: {config_path}[/green]")
    
    def run_setup(self) -> bool:
        """Ejecuta el setup completo"""
        
        self.show_intro()
        
        # Confirmación
        if not console.input("\\n[yellow]¿Continuar con el setup? (y/n): [/yellow]").lower().startswith('y'):
            console.print("[yellow]Setup cancelado[/yellow]")
            return False
        
        try:
            # 1. Crear directorios
            self.setup_directories()
            
            # 2. Inicializar ChromaDB
            if not self.initialize_chroma():
                return False
            
            # 3. Poblar base de conocimiento
            self.populate_knowledge_base()
            
            # 4. Verificar
            if not self.verify_knowledge_base():
                return False
            
            # 5. Crear config
            self.create_config_file()
            
            # Mostrar resumen
            console.print("\\n" + "="*60)
            console.print("[bold green]🎉 ¡Base de conocimiento creada exitosamente![/bold green]")
            console.print("="*60)
            console.print(f"📊 Ejemplos creados: {len(self.knowledge_examples)}")
            console.print(f"🗄️ ChromaDB: {self.chroma_dir}")
            console.print(f"📁 Configuración: {self.data_dir}/knowledge_base_config.json")
            console.print("\\n[cyan]🚀 Ya puedes ejecutar el asistente principal:[/cyan]")
            console.print("[bold]python src/main.py[/bold]")
            
            return True
            
        except Exception as e:
            console.print(f"\\n[red]❌ Error durante el setup: {e}[/red]")
            logger.exception("Error en setup de knowledge base")
            return False


def main():
    """Función principal"""
    
    # Configurar logging
    logger.remove()
    logger.add(sys.stderr, level="INFO",
              format="<green>{time:HH:mm:ss}</green> | {message}")
    
    setup = KnowledgeBaseSetup()
    
    try:
        success = setup.run_setup()
        exit_code = 0 if success else 1
        
    except KeyboardInterrupt:
        console.print("\\n[yellow]⏹️ Setup interrumpido por el usuario[/yellow]")
        exit_code = 1
        
    except Exception as e:
        console.print(f"\\n[red]❌ Error inesperado: {e}[/red]")
        logger.exception("Error en setup de knowledge base")
        exit_code = 1
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
