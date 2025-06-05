"""
Knowledge Base especializada para Cobranza

Este módulo contiene ejemplos específicos de técnicas de cobranza,
manejo de objeciones, compliance y estrategias probadas para
maximar tasas de recuperación.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any

from chroma_manager import ChromaManager
from rich.progress import Progress
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def get_collections_objections_responses():
    """Respuestas AEIOU para objeciones comunes en cobranza"""
    return [
        {
            "content": "Entiendo que crees que ya pagaste esta cuenta (A). Quiero asegurarme de que tengamos la información correcta en nuestros registros (E). ¿Podrías ayudarme con la fecha exacta y el método de pago que utilizaste? (I) Mi objetivo es resolver cualquier discrepancia rápidamente (O). ¿Tienes el número de confirmación o recibo disponible? (U)",
            "metadata": {
                "category": "dispute_already_paid",
                "objection_type": "dispute",
                "effectiveness_score": 0.84,
                "context_type": "account_verification",
                "aeiou_components": ["acknowledge", "express", "identify", "outcome", "understanding"],
                "compliance_notes": "Requests documentation, non-confrontational",
                "industry": "general_collections",
                "urgency_level": "medium",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                "objection_phrase": "Ya pagué esa deuda"
            }
        },
        {
            "content": "Reconozco que la situación económica está muy difícil en este momento (A). Nosotros también queremos encontrar una solución que funcione para tu presupuesto (E). ¿Qué tal si estructuramos un plan pequeño de $25 por quincena? (I) Mi objetivo es resolver esto sin que afecte más tu historial crediticio (O). ¿Te parece razonable empezar el próximo viernes de pago? (U)",
            "metadata": {
                "category": "financial_hardship",
                "objection_type": "inability_to_pay",
                "effectiveness_score": 0.89,
                "context_type": "payment_arrangement",
                "aeiou_components": ["acknowledge", "express", "identify", "outcome", "understanding"],
                "compliance_notes": "Empathetic, offers reasonable plan, no coercion",
                "industry": "general_collections",
                "urgency_level": "low",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                "objection_phrase": "No tengo dinero para pagar nada"
            }
        },
        {
            "content": "Entiendo que ahora no es el mejor momento para hablar (A). Yo también prefiero llamar cuando sea conveniente para ti (E). ¿Qué día específico de la próxima semana sería mejor para una conversación de 5 minutos? (I) Mi objetivo es resolver esto en esa llamada programada (O). ¿Te parece el martes a las 2 PM o preferirías otra hora? (U)",
            "metadata": {
                "category": "call_avoidance",
                "objection_type": "stalling",
                "effectiveness_score": 0.76,
                "context_type": "appointment_setting",
                "aeiou_components": ["acknowledge", "express", "identify", "outcome", "understanding"],
                "compliance_notes": "Respectful, specific appointment, time limit mentioned",
                "industry": "general_collections",
                "urgency_level": "medium",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                "objection_phrase": "Llamen la próxima semana"
            }
        },
        {
            "content": "Entiendo completamente que no reconoces esta deuda (A). Es importante que verifiquemos toda la información correctamente (E). Voy a enviarte por correo toda la documentación de validación dentro de 5 días hábiles (I). Mi objetivo es que tengas toda la información necesaria para revisar la cuenta (O). ¿Cuál es la mejor dirección postal para enviarte estos documentos? (U)",
            "metadata": {
                "category": "debt_validation",
                "objection_type": "dispute",
                "effectiveness_score": 0.92,
                "context_type": "legal_compliance",
                "aeiou_components": ["acknowledge", "express", "identify", "outcome", "understanding"],
                "compliance_notes": "FDCPA compliant, offers validation, no pressure",
                "industry": "general_collections",
                "urgency_level": "high",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                "objection_phrase": "Esa deuda no es mía"
            }
        },
        {
            "content": "Reconozco que has tenido dificultades para cumplir con los pagos acordados anteriormente (A). Entiendo que pueden surgir circunstancias imprevistas (E). ¿Qué tal si esta vez empezamos con un compromiso más pequeño de $50 esta semana? (I) Mi objetivo es ayudarte a reconstruir un historial de pagos consistente (O). ¿Puedes confirmar que el miércoles podrás hacer este pago inicial? (U)",
            "metadata": {
                "category": "broken_promises",
                "objection_type": "credibility_issue",
                "effectiveness_score": 0.71,
                "context_type": "trust_rebuilding",
                "aeiou_components": ["acknowledge", "express", "identify", "outcome", "understanding"],
                "compliance_notes": "Non-judgmental, smaller commitment, specific date",
                "industry": "general_collections",
                "urgency_level": "medium",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                "objection_phrase": "Ya les dije que iba a pagar y no pude"
            }
        }
    ]


def get_collections_techniques():
    """Técnicas específicas de cobranza y mejores prácticas"""
    return [
        {
            "content": "Técnica de Apertura 'Rapport + Agenda': Saludo personalizado, verificación de identidad, establecimiento de agenda clara. 'Buenos días Juan, soy María de ABC Collections. Te llamo respecto a tu cuenta con XYZ Company. ¿Tienes 5 minutos para revisar algunas opciones para resolver esto?'",
            "metadata": {
                "category": "opening_techniques",
                "technique_type": "rapport_building",
                "effectiveness_score": 0.87,
                "context_type": "call_opening",
                "best_for": ["first_contact", "cooperative_debtor"],
                "avoid_if": ["hostile_debtor", "dispute_case"],
                "compliance_notes": "Identifies self and company, states purpose clearly",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0
            }
        },
        {
            "content": "Técnica de Cierre 'Assumptive Close': Una vez que el deudor acepta un plan, asumir la venta y moverse a detalles logísticos. 'Perfecto, entonces el primer pago de $75 será el viernes 12. ¿Prefieres que lo procesemos desde tu cuenta corriente o de ahorros? Te enviaré confirmación por email.'",
            "metadata": {
                "category": "closing_techniques",
                "technique_type": "assumptive_close",
                "effectiveness_score": 0.82,
                "context_type": "payment_commitment",
                "best_for": ["agreed_payment_plan", "cooperative_debtor"],
                "avoid_if": ["hesitant_debtor", "dispute_case"],
                "compliance_notes": "Confirms agreement, provides written confirmation",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0
            }
        },
        {
            "content": "Técnica de Negociación 'Anchor and Adjust': Empezar con el monto total, luego ajustar basándose en la respuesta del deudor. 'El balance total es $1,200. Entiendo que eso puede ser difícil ahora mismo. ¿Qué cantidad podrías manejar mensualmente para resolver esto en 6 meses?'",
            "metadata": {
                "category": "negotiation_techniques",
                "technique_type": "anchoring",
                "effectiveness_score": 0.79,
                "context_type": "payment_negotiation",
                "best_for": ["financial_hardship", "large_balances"],
                "avoid_if": ["dispute_case", "validation_request"],
                "compliance_notes": "Provides full disclosure, allows debtor input",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0
            }
        }
    ]


def get_compliance_rules():
    """Reglas de compliance y best practices legales"""
    return [
        {
            "content": "FDCPA Sección 806: Prohibición de Acoso. Nunca usar lenguaje obsceno, amenazar violencia, publicar listas de deudores, o llamar repetidamente con intención de acosar. Límite: 7 intentos de contacto en 7 días. Documentar todos los intentos.",
            "metadata": {
                "category": "fdcpa_compliance",
                "rule_type": "harassment_prevention",
                "jurisdiction": "USA_federal",
                "penalty_level": "high",
                "monitoring_required": True,
                "auto_enforcement": True,
                "created_at": datetime.now().isoformat()
            }
        },
        {
            "content": "FDCPA Sección 807: Prácticas Engañosas Prohibidas. No falsificar identidad como abogado, amenazar con acciones legales no autorizadas, o crear urgencia falsa. Siempre identificarse como collector, proporcionar información precisa sobre la deuda.",
            "metadata": {
                "category": "fdcpa_compliance",
                "rule_type": "false_representation",
                "jurisdiction": "USA_federal",
                "penalty_level": "high",
                "monitoring_required": True,
                "auto_enforcement": True,
                "created_at": datetime.now().isoformat()
            }
        },
        {
            "content": "FDCPA Sección 808: Prácticas Injustas Prohibidas. No cobrar montos no autorizados, no amenazar con embargar propiedades sin autoridad legal, no comunicarse por postal excepto para proceso legal. Respetar horarios: 8 AM - 9 PM en zona horaria del deudor.",
            "metadata": {
                "category": "fdcpa_compliance",
                "rule_type": "unfair_practices",
                "jurisdiction": "USA_federal",
                "penalty_level": "medium",
                "monitoring_required": True,
                "auto_enforcement": True,
                "created_at": datetime.now().isoformat()
            }
        }
    ]


def get_escalation_protocols():
    """Protocolos de escalación para diferentes situaciones"""
    return [
        {
            "content": "Escalación a Supervisor - Cliente Hostil: Si el deudor usa profanidad repetida, amenaza al agente, o se vuelve verbalmente abusivo, transferir inmediatamente. Script: 'Entiendo que estás frustrado. Voy a conectarte con mi supervisor quien puede tener más opciones disponibles.'",
            "metadata": {
                "category": "escalation_protocol",
                "trigger_type": "hostile_debtor",
                "escalation_level": "supervisor",
                "urgency": "immediate",
                "documentation_required": True,
                "compliance_notes": "Protects agent, defuses situation",
                "created_at": datetime.now().isoformat()
            }
        },
        {
            "content": "Escalación a Legal - Solicitud de Validación: Cuando el deudor solicita validación de deuda por escrito o disputa la deuda completamente, cesar intentos de cobranza y transferir a departamento legal. No continuar cobranza hasta que se envíe validación.",
            "metadata": {
                "category": "escalation_protocol",
                "trigger_type": "debt_validation",
                "escalation_level": "legal_department",
                "urgency": "high",
                "documentation_required": True,
                "compliance_notes": "FDCPA required, must cease collection",
                "created_at": datetime.now().isoformat()
            }
        }
    ]


def populate_collections_knowledge_base(chroma_manager: ChromaManager):
    """Puebla la knowledge base con contenido especializado de cobranza"""
    console.print("[bold blue]🏦 Poblando Knowledge Base para Cobranza[/bold blue]")
    
    with Progress(console=console) as progress:
        # Objeciones y respuestas
        objections_task = progress.add_task("[red]Objeciones comunes...", total=len(get_collections_objections_responses()))
        
        for objection in get_collections_objections_responses():
            doc_id = chroma_manager.add_document(
                "aeiou_examples",
                objection["content"],
                objection["metadata"]
            )
            progress.advance(objections_task)
        
        # Técnicas de cobranza
        techniques_task = progress.add_task("[green]Técnicas de cobranza...", total=len(get_collections_techniques()))
        
        for technique in get_collections_techniques():
            doc_id = chroma_manager.add_document(
                "communication_styles",
                technique["content"],
                technique["metadata"]
            )
            progress.advance(techniques_task)
        
        # Reglas de compliance
        compliance_task = progress.add_task("[yellow]Reglas de compliance...", total=len(get_compliance_rules()))
        
        for rule in get_compliance_rules():
            doc_id = chroma_manager.add_document(
                "conflict_patterns",  # Usando esta colección para compliance
                rule["content"],
                rule["metadata"]
            )
            progress.advance(compliance_task)
        
        # Protocolos de escalación
        escalation_task = progress.add_task("[cyan]Protocolos de escalación...", total=len(get_escalation_protocols()))
        
        for protocol in get_escalation_protocols():
            doc_id = chroma_manager.add_document(
                "meeting_contexts",  # Usando esta colección para protocolos
                protocol["content"],
                protocol["metadata"]
            )
            progress.advance(escalation_task)
    
    console.print("\n[bold green]✅ Knowledge Base de cobranza poblada exitosamente![/bold green]")
    
    # Test de consulta especializada
    console.print("\n[bold yellow]🔍 Test de consulta especializada:[/bold yellow]")
    test_queries = [
        "cliente dice que ya pagó la deuda",
        "no tengo dinero para pagar",
        "cliente hostil y agresivo"
    ]
    
    for query in test_queries:
        results = chroma_manager.query_similar("aeiou_examples", query, n_results=1)
        if results["documents"]:
            effectiveness = results["metadatas"][0].get("effectiveness_score", 0)
            console.print(f"[dim]Query:[/dim] {query}")
            console.print(f"[green]Respuesta[/green] (efectividad: {effectiveness:.0%}): {results['documents'][0][:80]}...\n")


if __name__ == "__main__":
    # Test de la knowledge base de cobranza
    from chroma_manager import ChromaManager
    
    chroma = ChromaManager(data_dir="./data/chroma_collections")
    populate_collections_knowledge_base(chroma)
    
    # Mostrar estadísticas finales
    for collection_name in ["aeiou_examples", "communication_styles", "conflict_patterns", "meeting_contexts"]:
        stats = chroma.get_collection_stats(collection_name)
        console.print(
            f"[cyan]{collection_name}[/cyan]: {stats['total_documents']} documentos"
        )
