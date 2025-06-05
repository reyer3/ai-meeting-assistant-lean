"""
Integración con CRMs de Cobranza

Conectores para los principales sistemas de gestión de cobranza
including TOTALS, GECCO, y otros CRMs populares.
"""

import json
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DebtorAccount:
    """Información de cuenta del deudor"""
    debtor_id: str
    account_number: str
    original_creditor: str
    current_balance: float
    original_balance: float
    days_past_due: int
    last_payment_date: Optional[datetime]
    last_payment_amount: float
    payment_history: List[Dict]
    contact_info: Dict[str, str]
    notes: List[str]
    status: str  # "active", "settled", "legal", "closed"
    priority: str  # "high", "medium", "low"


@dataclass
class CallOutcome:
    """Resultado de una llamada para actualizar en CRM"""
    call_id: str
    debtor_id: str
    agent_id: str
    timestamp: datetime
    duration_seconds: int
    outcome_code: str
    disposition: str
    amount_promised: float
    payment_date: Optional[datetime]
    next_action: str
    notes: str
    compliance_score: float
    ai_suggestions_used: int


class CRMIntegration(ABC):
    """Clase base abstracta para integraciones CRM"""
    
    @abstractmethod
    def get_account_info(self, debtor_id: str) -> DebtorAccount:
        """Obtiene información de la cuenta del deudor"""
        pass
    
    @abstractmethod
    def update_call_outcome(self, outcome: CallOutcome) -> bool:
        """Actualiza el resultado de la llamada en el CRM"""
        pass
    
    @abstractmethod
    def get_next_accounts_to_call(self, agent_id: str, limit: int = 10) -> List[DebtorAccount]:
        """Obtiene las próximas cuentas para llamar"""
        pass
    
    @abstractmethod
    def update_account_status(self, debtor_id: str, new_status: str, notes: str = "") -> bool:
        """Actualiza el estado de la cuenta"""
        pass


class TOTALSConnector(CRMIntegration):
    """Conector para TOTALS Collection Software"""
    
    def __init__(self, api_base_url: str, username: str, password: str, client_id: str):
        self.base_url = api_base_url.rstrip('/')
        self.username = username
        self.password = password
        self.client_id = client_id
        self.session = requests.Session()
        self.auth_token = None
        
        # Configurar headers por defecto
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Collections-Assistant/1.0'
        })
        
        self._authenticate()
    
    def _authenticate(self):
        """Autenticación con TOTALS API"""
        auth_data = {
            'username': self.username,
            'password': self.password,
            'client_id': self.client_id
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/auth/login",
                json=auth_data
            )
            response.raise_for_status()
            
            auth_result = response.json()
            self.auth_token = auth_result.get('access_token')
            
            self.session.headers.update({
                'Authorization': f'Bearer {self.auth_token}'
            })
            
            logger.info("Successfully authenticated with TOTALS")
            
        except requests.RequestException as e:
            logger.error(f"TOTALS authentication failed: {e}")
            raise
    
    def get_account_info(self, debtor_id: str) -> DebtorAccount:
        """Obtiene información de cuenta desde TOTALS"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/accounts/{debtor_id}",
                params={'include': 'payment_history,contact_info,notes'}
            )
            response.raise_for_status()
            
            account_data = response.json()
            
            return DebtorAccount(
                debtor_id=account_data['debtor_id'],
                account_number=account_data['account_number'],
                original_creditor=account_data['original_creditor'],
                current_balance=float(account_data['current_balance']),
                original_balance=float(account_data['original_balance']),
                days_past_due=account_data['days_past_due'],
                last_payment_date=datetime.fromisoformat(account_data['last_payment_date']) if account_data.get('last_payment_date') else None,
                last_payment_amount=float(account_data.get('last_payment_amount', 0)),
                payment_history=account_data.get('payment_history', []),
                contact_info=account_data.get('contact_info', {}),
                notes=account_data.get('notes', []),
                status=account_data['status'],
                priority=account_data.get('priority', 'medium')
            )
            
        except requests.RequestException as e:
            logger.error(f"Failed to get account info for {debtor_id}: {e}")
            raise
    
    def update_call_outcome(self, outcome: CallOutcome) -> bool:
        """Actualiza resultado de llamada en TOTALS"""
        call_data = {
            'call_id': outcome.call_id,
            'debtor_id': outcome.debtor_id,
            'agent_id': outcome.agent_id,
            'timestamp': outcome.timestamp.isoformat(),
            'duration_seconds': outcome.duration_seconds,
            'outcome_code': outcome.outcome_code,
            'disposition': outcome.disposition,
            'amount_promised': outcome.amount_promised,
            'payment_date': outcome.payment_date.isoformat() if outcome.payment_date else None,
            'next_action': outcome.next_action,
            'notes': outcome.notes,
            'compliance_score': outcome.compliance_score,
            'ai_suggestions_used': outcome.ai_suggestions_used,
            'source': 'ai_collections_assistant'
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/calls",
                json=call_data
            )
            response.raise_for_status()
            
            logger.info(f"Call outcome updated in TOTALS: {outcome.call_id}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to update call outcome: {e}")
            return False
    
    def get_next_accounts_to_call(self, agent_id: str, limit: int = 10) -> List[DebtorAccount]:
        """Obtiene cuentas prioritarias para llamar"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/agents/{agent_id}/queue",
                params={'limit': limit, 'status': 'active'}
            )
            response.raise_for_status()
            
            queue_data = response.json()
            accounts = []
            
            for account_data in queue_data.get('accounts', []):
                accounts.append(DebtorAccount(
                    debtor_id=account_data['debtor_id'],
                    account_number=account_data['account_number'],
                    original_creditor=account_data['original_creditor'],
                    current_balance=float(account_data['current_balance']),
                    original_balance=float(account_data['original_balance']),
                    days_past_due=account_data['days_past_due'],
                    last_payment_date=datetime.fromisoformat(account_data['last_payment_date']) if account_data.get('last_payment_date') else None,
                    last_payment_amount=float(account_data.get('last_payment_amount', 0)),
                    payment_history=[],  # No incluido en queue response
                    contact_info=account_data.get('contact_info', {}),
                    notes=[],  # No incluido en queue response
                    status=account_data['status'],
                    priority=account_data.get('priority', 'medium')
                ))
            
            return accounts
            
        except requests.RequestException as e:
            logger.error(f"Failed to get agent queue: {e}")
            return []
    
    def update_account_status(self, debtor_id: str, new_status: str, notes: str = "") -> bool:
        """Actualiza estado de cuenta"""
        status_data = {
            'status': new_status,
            'notes': notes,
            'updated_by': 'ai_collections_assistant',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            response = self.session.patch(
                f"{self.base_url}/api/accounts/{debtor_id}/status",
                json=status_data
            )
            response.raise_for_status()
            
            logger.info(f"Account status updated: {debtor_id} -> {new_status}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to update account status: {e}")
            return False


class GECCOConnector(CRMIntegration):
    """Conector para GECCO Collection Software"""
    
    def __init__(self, database_config: Dict[str, str]):
        self.db_config = database_config
        # En implementación real, usar SQLAlchemy o pyodbc
        self.connection = None
        self._connect_database()
    
    def _connect_database(self):
        """Conecta a la base de datos GECCO"""
        try:
            # Placeholder para conexión real a GECCO DB
            # import pyodbc
            # connection_string = f"DRIVER={self.db_config['driver']};SERVER={self.db_config['server']};DATABASE={self.db_config['database']};UID={self.db_config['username']};PWD={self.db_config['password']}"
            # self.connection = pyodbc.connect(connection_string)
            logger.info("Connected to GECCO database")
        except Exception as e:
            logger.error(f"Failed to connect to GECCO database: {e}")
            raise
    
    def get_account_info(self, debtor_id: str) -> DebtorAccount:
        """Obtiene información de cuenta desde GECCO DB"""
        # Implementación placeholder
        # En realidad ejecutaría queries SQL a las tablas de GECCO
        return DebtorAccount(
            debtor_id=debtor_id,
            account_number=f"GECCO_{debtor_id}",
            original_creditor="Test Creditor",
            current_balance=1250.00,
            original_balance=1500.00,
            days_past_due=45,
            last_payment_date=datetime(2024, 1, 15),
            last_payment_amount=50.00,
            payment_history=[],
            contact_info={"phone": "555-0123", "email": "test@example.com"},
            notes=["Previous contact attempted"],
            status="active",
            priority="medium"
        )
    
    def update_call_outcome(self, outcome: CallOutcome) -> bool:
        """Actualiza resultado en GECCO DB"""
        # Implementación placeholder
        # En realidad insertaría en tabla de call_log de GECCO
        logger.info(f"GECCO call outcome updated: {outcome.call_id}")
        return True
    
    def get_next_accounts_to_call(self, agent_id: str, limit: int = 10) -> List[DebtorAccount]:
        """Obtiene cuentas desde GECCO DB"""
        # Implementación placeholder
        return []
    
    def update_account_status(self, debtor_id: str, new_status: str, notes: str = "") -> bool:
        """Actualiza estado en GECCO DB"""
        # Implementación placeholder
        return True


class GenericRESTConnector(CRMIntegration):
    """Conector genérico para CRMs con API REST"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config['base_url'].rstrip('/')
        self.session = requests.Session()
        
        # Configurar autenticación
        auth_type = config.get('auth_type', 'bearer')
        if auth_type == 'bearer':
            self.session.headers.update({
                'Authorization': f"Bearer {config['token']}"
            })
        elif auth_type == 'api_key':
            self.session.headers.update({
                config.get('api_key_header', 'X-API-Key'): config['api_key']
            })
        
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Collections-Assistant/1.0'
        })
    
    def get_account_info(self, debtor_id: str) -> DebtorAccount:
        """Obtiene cuenta usando endpoints configurados"""
        endpoint = self.config['endpoints']['get_account'].format(debtor_id=debtor_id)
        
        try:
            response = self.session.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            
            data = response.json()
            
            # Mapear campos según configuración
            field_mapping = self.config.get('field_mapping', {})
            
            return DebtorAccount(
                debtor_id=data[field_mapping.get('debtor_id', 'debtor_id')],
                account_number=data[field_mapping.get('account_number', 'account_number')],
                original_creditor=data[field_mapping.get('original_creditor', 'original_creditor')],
                current_balance=float(data[field_mapping.get('current_balance', 'current_balance')]),
                original_balance=float(data[field_mapping.get('original_balance', 'original_balance')]),
                days_past_due=data[field_mapping.get('days_past_due', 'days_past_due')],
                last_payment_date=datetime.fromisoformat(data[field_mapping.get('last_payment_date', 'last_payment_date')]) if data.get(field_mapping.get('last_payment_date', 'last_payment_date')) else None,
                last_payment_amount=float(data.get(field_mapping.get('last_payment_amount', 'last_payment_amount'), 0)),
                payment_history=data.get(field_mapping.get('payment_history', 'payment_history'), []),
                contact_info=data.get(field_mapping.get('contact_info', 'contact_info'), {}),
                notes=data.get(field_mapping.get('notes', 'notes'), []),
                status=data[field_mapping.get('status', 'status')],
                priority=data.get(field_mapping.get('priority', 'priority'), 'medium')
            )
            
        except requests.RequestException as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    def update_call_outcome(self, outcome: CallOutcome) -> bool:
        """Actualiza resultado usando endpoint configurado"""
        endpoint = self.config['endpoints']['update_call']
        
        # Mapear datos según configuración
        field_mapping = self.config.get('field_mapping', {})
        
        mapped_data = {
            field_mapping.get('call_id', 'call_id'): outcome.call_id,
            field_mapping.get('debtor_id', 'debtor_id'): outcome.debtor_id,
            field_mapping.get('agent_id', 'agent_id'): outcome.agent_id,
            field_mapping.get('timestamp', 'timestamp'): outcome.timestamp.isoformat(),
            field_mapping.get('duration_seconds', 'duration_seconds'): outcome.duration_seconds,
            field_mapping.get('outcome_code', 'outcome_code'): outcome.outcome_code,
            field_mapping.get('amount_promised', 'amount_promised'): outcome.amount_promised,
            field_mapping.get('notes', 'notes'): outcome.notes
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                json=mapped_data
            )
            response.raise_for_status()
            
            logger.info(f"Call outcome updated via REST API: {outcome.call_id}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to update call outcome: {e}")
            return False
    
    def get_next_accounts_to_call(self, agent_id: str, limit: int = 10) -> List[DebtorAccount]:
        """Obtiene queue usando endpoint configurado"""
        endpoint = self.config['endpoints']['get_queue'].format(agent_id=agent_id)
        
        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                params={'limit': limit}
            )
            response.raise_for_status()
            
            data = response.json()
            accounts = []
            
            for account_data in data.get('accounts', []):
                # Usar mismo mapeo que get_account_info pero con datos limitados
                field_mapping = self.config.get('field_mapping', {})
                
                accounts.append(DebtorAccount(
                    debtor_id=account_data[field_mapping.get('debtor_id', 'debtor_id')],
                    account_number=account_data[field_mapping.get('account_number', 'account_number')],
                    original_creditor=account_data.get(field_mapping.get('original_creditor', 'original_creditor'), ''),
                    current_balance=float(account_data[field_mapping.get('current_balance', 'current_balance')]),
                    original_balance=float(account_data.get(field_mapping.get('original_balance', 'original_balance'), 0)),
                    days_past_due=account_data[field_mapping.get('days_past_due', 'days_past_due')],
                    last_payment_date=None,  # No incluido en queue
                    last_payment_amount=0,   # No incluido en queue
                    payment_history=[],      # No incluido en queue
                    contact_info=account_data.get(field_mapping.get('contact_info', 'contact_info'), {}),
                    notes=[],                # No incluido en queue
                    status=account_data[field_mapping.get('status', 'status')],
                    priority=account_data.get(field_mapping.get('priority', 'priority'), 'medium')
                ))
            
            return accounts
            
        except requests.RequestException as e:
            logger.error(f"Failed to get queue: {e}")
            return []
    
    def update_account_status(self, debtor_id: str, new_status: str, notes: str = "") -> bool:
        """Actualiza estado usando endpoint configurado"""
        endpoint = self.config['endpoints']['update_status'].format(debtor_id=debtor_id)
        
        field_mapping = self.config.get('field_mapping', {})
        
        status_data = {
            field_mapping.get('status', 'status'): new_status,
            field_mapping.get('notes', 'notes'): notes,
            field_mapping.get('timestamp', 'timestamp'): datetime.now().isoformat()
        }
        
        try:
            response = self.session.patch(
                f"{self.base_url}{endpoint}",
                json=status_data
            )
            response.raise_for_status()
            
            logger.info(f"Account status updated: {debtor_id} -> {new_status}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to update account status: {e}")
            return False


class CRMIntegrationManager:
    """Gestor de integraciones CRM"""
    
    def __init__(self, config_file: str = "crm_config.json"):
        self.connectors: Dict[str, CRMIntegration] = {}
        self.active_connector = None
        self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Carga configuración de CRMs"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            for crm_name, crm_config in config.get('crms', {}).items():
                if not crm_config.get('enabled', False):
                    continue
                
                crm_type = crm_config['type']
                
                if crm_type == 'totals':
                    connector = TOTALSConnector(
                        api_base_url=crm_config['api_base_url'],
                        username=crm_config['username'],
                        password=crm_config['password'],
                        client_id=crm_config['client_id']
                    )
                
                elif crm_type == 'gecco':
                    connector = GECCOConnector(
                        database_config=crm_config['database']
                    )
                
                elif crm_type == 'generic_rest':
                    connector = GenericRESTConnector(crm_config)
                
                else:
                    logger.warning(f"Unknown CRM type: {crm_type}")
                    continue
                
                self.connectors[crm_name] = connector
                
                if crm_config.get('primary', False):
                    self.active_connector = connector
                    logger.info(f"Set {crm_name} as primary CRM connector")
            
            if not self.active_connector and self.connectors:
                # Si no hay primary, usar el primero disponible
                self.active_connector = list(self.connectors.values())[0]
                logger.info("No primary CRM set, using first available connector")
        
        except Exception as e:
            logger.error(f"Failed to load CRM config: {e}")
    
    def get_account_info(self, debtor_id: str) -> Optional[DebtorAccount]:
        """Obtiene información de cuenta usando conector activo"""
        if not self.active_connector:
            logger.error("No active CRM connector")
            return None
        
        try:
            return self.active_connector.get_account_info(debtor_id)
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None
    
    def update_call_outcome(self, outcome: CallOutcome) -> bool:
        """Actualiza resultado usando conector activo"""
        if not self.active_connector:
            logger.error("No active CRM connector")
            return False
        
        return self.active_connector.update_call_outcome(outcome)
    
    def get_next_accounts_to_call(self, agent_id: str, limit: int = 10) -> List[DebtorAccount]:
        """Obtiene queue usando conector activo"""
        if not self.active_connector:
            logger.error("No active CRM connector")
            return []
        
        return self.active_connector.get_next_accounts_to_call(agent_id, limit)


if __name__ == "__main__":
    # Test de integraciones CRM
    
    # Ejemplo de configuración para GenericRESTConnector
    generic_config = {
        'base_url': 'https://api.example-crm.com',
        'auth_type': 'bearer',
        'token': 'your-api-token',
        'endpoints': {
            'get_account': '/accounts/{debtor_id}',
            'update_call': '/calls',
            'get_queue': '/agents/{agent_id}/queue',
            'update_status': '/accounts/{debtor_id}/status'
        },
        'field_mapping': {
            'debtor_id': 'customer_id',
            'current_balance': 'balance',
            'days_past_due': 'days_overdue'
        }
    }
    
    # Test GenericRESTConnector
    try:
        connector = GenericRESTConnector(generic_config)
        print("Generic REST connector initialized successfully")
    except Exception as e:
        print(f"Generic REST connector failed: {e}")
    
    # Test con datos de ejemplo
    test_outcome = CallOutcome(
        call_id="TEST_001",
        debtor_id="DBT_123",
        agent_id="AGT_001",
        timestamp=datetime.now(),
        duration_seconds=300,
        outcome_code="PTP",
        disposition="Promise to Pay",
        amount_promised=150.0,
        payment_date=datetime.now(),
        next_action="Follow up on payment date",
        notes="Customer agreed to payment plan",
        compliance_score=0.95,
        ai_suggestions_used=2
    )
    
    print(f"Test outcome created: {test_outcome.call_id}")
