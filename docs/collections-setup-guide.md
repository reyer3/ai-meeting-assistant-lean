# 🎯 AI Collections Assistant - Setup Guide

> Guía completa para implementar el asistente de IA especializado en cobranza

## 📋 Pre-requisitos

### Requerimientos del Sistema
- **RAM**: 6GB mínimo (8GB recomendado)
- **CPU**: Intel i5 2018+ o AMD Ryzen 5 equivalente  
- **Almacenamiento**: 3GB espacio libre
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 20.04+
- **Audio**: Dispositivo de captura de audio activo

### Requerimientos Legales
- Licencia de call center de cobranza vigente
- Cumplimiento con regulaciones locales (FDCPA, etc.)
- Políticas de grabación de llamadas establecidas
- Consentimiento de agentes para asistencia de IA

## 🚀 Instalación Rápida

### 1. Clonar Repositorio
```bash
git clone https://github.com/reyer3/ai-meeting-assistant-lean.git
cd ai-meeting-assistant-lean
```

### 2. Configuración del Entorno
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Setup Inicial para Cobranza
```bash
# Configuración automática
python setup.py --collections

# Poblar knowledge base de cobranza
python src/rag/collections_knowledge_base.py

# Verificar compliance engine
python src/compliance/compliance_engine.py
```

### 4. Configuración de Perfil de Voz
```bash
# Crear perfil de voz del agente
python src/voice_profile_setup.py --agent-mode
```

## ⚙️ Configuración Detallada

### Audio Setup para Call Centers

#### Opción 1: Captura Directa del Sistema
```python
# Configurar en collections_config.json
{
  "audio": {
    "device_name": "Stereo Mix",  # Windows
    "sample_rate": 16000,
    "noise_reduction": true,
    "auto_gain_control": true
  }
}
```

#### Opción 2: Integración con Softphone
```python
# Para integración con softphones
{
  "audio": {
    "source_type": "line_in",
    "monitoring_mode": true,
    "dual_channel": true  # Canal 1: Agente, Canal 2: Cliente
  }
}
```

### Knowledge Base Personalizada

#### Agregar Objeciones Específicas
```python
# src/rag/custom_objections.py
custom_objections = [
    {
        "objection": "No tengo trabajo desde la pandemia",
        "response": "Entiendo que la situación laboral ha sido muy desafiante (A)...",
        "effectiveness_score": 0.87,
        "industry": "credit_cards"
    }
]
```

#### Configurar por Vertical de Cobranza
```python
# Configuraciones por tipo de deuda
VERTICALS = {
    "credit_cards": {
        "max_settlement": 0.7,
        "min_payment_plan": 25,
        "typical_objections": ["high_interest", "minimum_payments"]
    },
    "medical_debt": {
        "max_settlement": 0.5,
        "min_payment_plan": 15,
        "typical_objections": ["insurance_dispute", "medical_hardship"]
    }
}
```

### Compliance Configuration

#### FDCPA Settings
```json
{
  "compliance": {
    "fdcpa": {
      "enabled": true,
      "jurisdiction": "USA",
      "auto_terminate_violations": true,
      "prohibited_phrases": [
        "te vamos a demandar",
        "soy abogado",
        "tienes 24 horas"
      ],
      "required_disclosures": [
        "debt_collector_identification",
        "validation_rights",
        "recording_notification"
      ]
    }
  }
}
```

#### State-Specific Regulations
```json
{
  "state_regulations": {
    "california": {
      "max_calls_per_week": 5,
      "required_breaks_between_calls": 7,
      "additional_disclosures": ["california_rosenthal_act"]
    },
    "texas": {
      "max_calls_per_week": 7,
      "homestead_exemption_notice": true
    }
  }
}
```

## 🔗 Integraciones CRM

### TOTALS Integration
```python
# src/integrations/totals_connector.py
class TOTALSConnector:
    def __init__(self, api_endpoint, auth_token):
        self.endpoint = api_endpoint
        self.token = auth_token
    
    def get_account_info(self, debtor_id):
        # Obtener información de la cuenta
        return {
            "balance": 1250.00,
            "days_past_due": 45,
            "last_payment": "2024-01-15",
            "payment_history": [...]
        }
    
    def update_call_outcome(self, call_id, outcome_data):
        # Actualizar resultado de la llamada
        pass
```

### GECCO Integration
```python
# src/integrations/gecco_connector.py
class GECCOConnector:
    def __init__(self, database_connection):
        self.db = database_connection
    
    def sync_account_data(self, account_number):
        # Sincronizar datos de cuenta
        pass
```

## 📊 Dashboard y Reportes

### Métricas en Tiempo Real
```python
# src/dashboard/real_time_metrics.py
class CollectionsDashboard:
    def get_agent_performance(self, agent_id):
        return {
            "calls_today": 45,
            "promises_obtained": 12,
            "average_call_time": "4:32",
            "compliance_score": 0.94,
            "recovery_rate": 0.23
        }
    
    def get_team_metrics(self, team_id):
        return {
            "total_calls": 450,
            "total_recovery": 15750.00,
            "average_compliance": 0.91,
            "escalation_rate": 0.08
        }
```

### Reportes Automáticos
```python
# Configurar reportes diarios
{
  "reporting": {
    "daily_reports": {
      "enabled": true,
      "delivery_time": "09:00",
      "recipients": ["manager@callcenter.com"],
      "include_metrics": [
        "recovery_rate",
        "compliance_violations",
        "agent_performance",
        "escalation_summary"
      ]
    }
  }
}
```

## 🎓 Entrenamiento de Agentes

### Sesión de Onboarding
```markdown
### Día 1: Introducción al Sistema
- Overview de funcionalidades
- Configuración de perfil de voz personal
- Práctica con objeciones básicas
- Compliance 101

### Día 2: Técnicas Avanzadas
- Manejo de objeciones complejas
- Interpretación de sugerencias del sistema
- Escalación apropiada
- Métricas personales

### Día 3: Práctica Supervisada
- Llamadas reales con supervisor
- Feedback en tiempo real
- Ajustes de configuración
- Evaluación de compliance
```

### Material de Entrenamiento
```python
# src/training/agent_training.py
class AgentTraining:
    def generate_practice_scenarios(self, difficulty_level):
        scenarios = {
            "beginner": [
                "Cliente cooperativo, situación financiera estable",
                "Objeción simple: 'No tengo dinero ahora'"
            ],
            "intermediate": [
                "Cliente evasivo, historial de promesas rotas",
                "Disputa del monto adeudado"
            ],
            "advanced": [
                "Cliente hostil, amenazas legales",
                "Solicitud formal de validación de deuda"
            ]
        }
        return scenarios[difficulty_level]
```

## 🔧 Troubleshooting

### Problemas Comunes

#### Audio No Se Captura
```bash
# Verificar dispositivos de audio
python -c "import sounddevice as sd; print(sd.query_devices())"

# Configurar dispositivo específico
# En collections_config.json:
{
  "audio": {
    "device_name": "Microphone (USB Audio Device)"
  }
}
```

#### Knowledge Base Vacía
```bash
# Re-poblar knowledge base
python src/rag/collections_knowledge_base.py

# Verificar ChromaDB
python -c "from src.rag.chroma_manager import ChromaManager; cm = ChromaManager(); print(cm.get_collection_stats('aeiou_examples'))"
```

#### Compliance Engine No Funciona
```bash
# Test del compliance engine
python src/compliance/compliance_engine.py

# Verificar configuración
grep -n "compliance" collections_config.json
```

### Logs y Debugging
```python
# Configurar logging detallado
{
  "logging": {
    "level": "DEBUG",
    "file": "./logs/collections_debug.log",
    "console": true,
    "compliance_log": "./logs/compliance_audit.log"
  }
}
```

## 📞 Soporte

### Documentación Adicional
- [API Reference](./api-reference.md)
- [Compliance Guide](./compliance-guide.md)
- [CRM Integrations](./crm-integrations.md)
- [Performance Tuning](./performance-tuning.md)

### Contacto
- **Technical Support**: tech-support@collections-ai.com
- **Compliance Questions**: compliance@collections-ai.com
- **Sales & Licensing**: sales@collections-ai.com

### Community
- [GitHub Issues](https://github.com/reyer3/ai-meeting-assistant-lean/issues)
- [Discussions](https://github.com/reyer3/ai-meeting-assistant-lean/discussions)
- [Knowledge Base](https://docs.collections-ai.com)

---

*Para setup enterprise o implementaciones customizadas, contacta nuestro equipo de Professional Services.*
