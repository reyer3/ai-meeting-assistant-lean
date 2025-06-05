#!/usr/bin/env python3
"""
Tests unitarios para el sistema de cobranza

Cubre los componentes principales: RAG, compliance, dashboard,
performance tracking y integraciones.
"""

import unittest
import tempfile
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Imports del sistema
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.chroma_manager import ChromaManager
from rag.collections_query_engine import CollectionsQueryEngine, CollectionsContext, ObjectionType
from compliance.compliance_engine import ComplianceEngine, ViolationType, SeverityLevel
from dashboard.collections_dashboard import CollectionsDashboard, AgentMetrics
from metrics.performance_tracker import PerformanceTracker, SuggestionLog, CallMetrics
from integrations.crm_integration import TOTALSConnector, CallOutcome


class TestChromaManager(unittest.TestCase):
    """Tests para ChromaManager"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_manager = ChromaManager(
            data_dir=self.temp_dir,
            model_name="all-MiniLM-L6-v2"
        )
    
    def test_add_document(self):
        """Test agregar documento a ChromaDB"""
        content = "Test content for collections"
        metadata = {
            "category": "test_category",
            "effectiveness_score": 0.85
        }
        
        doc_id = self.chroma_manager.add_document(
            "aeiou_examples", content, metadata
        )
        
        self.assertIsNotNone(doc_id)
        self.assertTrue(doc_id.startswith("aeiou_examples_"))
    
    def test_query_similar(self):
        """Test consulta de documentos similares"""
        # Agregar documento de prueba
        test_content = "Cliente dice que no tiene dinero para pagar"
        test_metadata = {
            "category": "financial_hardship",
            "effectiveness_score": 0.90
        }
        
        self.chroma_manager.add_document(
            "aeiou_examples", test_content, test_metadata
        )
        
        # Consultar similar
        results = self.chroma_manager.query_similar(
            "aeiou_examples",
            "no tengo dinero",
            n_results=1
        )
        
        self.assertEqual(len(results["documents"]), 1)
        self.assertIn("dinero", results["documents"][0])
    
    def test_update_effectiveness_score(self):
        """Test actualizar score de efectividad"""
        # Agregar documento
        doc_id = self.chroma_manager.add_document(
            "aeiou_examples",
            "Test content",
            {"effectiveness_score": 0.5}
        )
        
        # Actualizar score
        self.chroma_manager.update_effectiveness_score(
            "aeiou_examples", doc_id, 0.8
        )
        
        # Verificar actualización
        results = self.chroma_manager.collections["aeiou_examples"].get(ids=[doc_id])
        updated_metadata = results["metadatas"][0]
        
        self.assertEqual(updated_metadata["effectiveness_score"], 0.8)
        self.assertEqual(updated_metadata["usage_count"], 1)


class TestCollectionsQueryEngine(unittest.TestCase):
    """Tests para CollectionsQueryEngine"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_manager = ChromaManager(data_dir=self.temp_dir)
        self.query_engine = CollectionsQueryEngine(self.chroma_manager)
    
    def test_analyze_collections_context(self):
        """Test análisis de contexto de cobranza"""
        context = CollectionsContext(
            current_speaker="OTHER",
            recent_transcript="Ya pagué esa deuda el mes pasado",
            debtor_id="DBT_001",
            account_balance=1000.0
        )
        
        objection_type, debtor_profile = self.query_engine.analyze_collections_context(context)
        
        self.assertEqual(objection_type, ObjectionType.ALREADY_PAID)
    
    def test_build_collections_query(self):
        """Test construcción de query especializada"""
        context = CollectionsContext(
            current_speaker="OTHER",
            recent_transcript="No tengo dinero",
            days_past_due=45,
            account_balance=500.0
        )
        
        query = self.query_engine.build_collections_query(context, ObjectionType.CANNOT_PAY)
        
        self.assertIn("cannot_pay", query)
        self.assertIn("small_balance", query)
    
    def test_evaluate_escalation_need(self):
        """Test evaluación de necesidad de escalación"""
        # Caso que requiere escalación
        context_escalation = CollectionsContext(
            current_speaker="OTHER",
            recent_transcript="Esa deuda no es mía, quiero validación por escrito"
        )
        
        needs_escalation = self.query_engine.evaluate_escalation_need(
            context_escalation, ObjectionType.NOT_MY_DEBT
        )
        
        self.assertTrue(needs_escalation)
        
        # Caso que no requiere escalación
        context_normal = CollectionsContext(
            current_speaker="OTHER",
            recent_transcript="Entiendo, ¿qué opciones tengo?"
        )
        
        no_escalation = self.query_engine.evaluate_escalation_need(
            context_normal, ObjectionType.PAYMENT_PLAN_REQUEST
        )
        
        self.assertFalse(no_escalation)


class TestComplianceEngine(unittest.TestCase):
    """Tests para ComplianceEngine"""
    
    def setUp(self):
        self.compliance_engine = ComplianceEngine()
    
    def test_analyze_transcript_violations(self):
        """Test detección de violaciones en transcripción"""
        # Texto con violaciones
        violation_text = "Te vamos a demandar si no pagas ahora mismo"
        
        alerts = self.compliance_engine.analyze_transcript_real_time(
            violation_text, "AGT_001", "CALL_001"
        )
        
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0].severity, SeverityLevel.HIGH)
        self.assertTrue(alerts[0].auto_block)
    
    def test_compliant_language(self):
        """Test texto que cumple compliance"""
        compliant_text = "Entiendo su situación, ¿podríamos explorar opciones de pago?"
        
        alerts = self.compliance_engine.analyze_transcript_real_time(
            compliant_text, "AGT_001", "CALL_001"
        )
        
        self.assertEqual(len(alerts), 0)
    
    def test_validate_debt_validation_request(self):
        """Test detección de solicitud de validación"""
        validation_text = "Quiero que me envíen la documentación por escrito"
        
        alert = self.compliance_engine.validate_debt_validation_request(validation_text)
        
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, SeverityLevel.CRITICAL)
        self.assertTrue(alert.auto_block)
    
    def test_compliance_score_calculation(self):
        """Test cálculo de compliance score"""
        # Transcript limpio
        clean_transcript = "Buenos días, le llamo sobre su cuenta. ¿Podemos hablar?"
        
        score_result = self.compliance_engine.get_compliance_score(
            clean_transcript, "AGT_001"
        )
        
        self.assertEqual(score_result["compliance_score"], 100)
        self.assertEqual(score_result["total_violations"], 0)
        self.assertTrue(score_result["pass_threshold"])
        
        # Transcript con violaciones
        violation_transcript = "Te vamos a demandar. Estás mintiendo sobre tu situación."
        
        violation_score = self.compliance_engine.get_compliance_score(
            violation_transcript, "AGT_001"
        )
        
        self.assertLess(violation_score["compliance_score"], 100)
        self.assertGreater(violation_score["total_violations"], 0)


class TestCollectionsDashboard(unittest.TestCase):
    """Tests para CollectionsDashboard"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dashboard = CollectionsDashboard(data_dir=self.temp_dir)
    
    def test_update_agent_metrics(self):
        """Test actualización de métricas de agente"""
        agent_id = "AGT_001"
        
        self.dashboard.update_agent_metrics(
            agent_id=agent_id,
            agent_name="Test Agent",
            calls_today=25,
            promises_obtained=8,
            total_amount_promised=1200.0
        )
        
        self.assertIn(agent_id, self.dashboard.agent_metrics)
        
        metrics = self.dashboard.agent_metrics[agent_id]
        self.assertEqual(metrics.calls_today, 25)
        self.assertEqual(metrics.promises_obtained, 8)
        self.assertEqual(metrics.total_amount_promised, 1200.0)
    
    def test_get_agent_performance_summary(self):
        """Test obtener resumen de performance"""
        agent_id = "AGT_001"
        
        # Actualizar métricas primero
        self.dashboard.update_agent_metrics(
            agent_id=agent_id,
            agent_name="Test Agent",
            calls_today=50,
            promises_obtained=12,
            compliance_score=0.95
        )
        
        summary = self.dashboard.get_agent_performance_summary(agent_id)
        
        self.assertIn("agent_info", summary)
        self.assertIn("performance", summary)
        self.assertEqual(summary["agent_info"]["id"], agent_id)
        self.assertEqual(summary["performance"]["calls_today"]["value"], 50)
    
    def test_generate_daily_report(self):
        """Test generación de reporte diario"""
        # Agregar algunos agentes
        test_agents = [
            {"agent_id": "AGT_001", "agent_name": "Agent 1", "calls_today": 45, "promises_obtained": 10, "total_amount_promised": 1500.0},
            {"agent_id": "AGT_002", "agent_name": "Agent 2", "calls_today": 38, "promises_obtained": 8, "total_amount_promised": 1200.0}
        ]
        
        for agent_data in test_agents:
            self.dashboard.update_agent_metrics(**agent_data)
        
        report = self.dashboard.generate_daily_report()
        
        self.assertIn("summary", report)
        self.assertIn("top_performers", report)
        self.assertEqual(report["summary"]["total_agents"], 2)
        self.assertEqual(report["summary"]["total_calls"], 83)
    
    def test_export_metrics(self):
        """Test exportación de métricas"""
        # Agregar métricas de prueba
        self.dashboard.update_agent_metrics(
            agent_id="AGT_001",
            agent_name="Test Agent",
            calls_today=30,
            promises_obtained=5
        )
        
        export_path = self.dashboard.export_metrics(format_type="json")
        
        self.assertTrue(Path(export_path).exists())
        
        # Verificar contenido
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIn("agent_metrics", exported_data)
        self.assertIn("AGT_001", exported_data["agent_metrics"])


class TestPerformanceTracker(unittest.TestCase):
    """Tests para PerformanceTracker"""
    
    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix=".db")
        self.tracker = PerformanceTracker(db_path=self.temp_db)
    
    def tearDown(self):
        if Path(self.temp_db).exists():
            Path(self.temp_db).unlink()
    
    def test_log_suggestion(self):
        """Test logging de sugerencias"""
        suggestion_id = self.tracker.log_suggestion(
            agent_id="AGT_001",
            suggestion_text="Test suggestion",
            confidence=0.85,
            context={"objection_type": "cannot_pay", "debtor_profile": "cooperative"}
        )
        
        self.assertIsNotNone(suggestion_id)
        self.assertTrue(suggestion_id.startswith("SUG_AGT_001_"))
        
        # Verificar en base de datos
        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM suggestions WHERE suggestion_id = ?", (suggestion_id,))
            count = cursor.fetchone()[0]
            
        self.assertEqual(count, 1)
    
    def test_log_suggestion_feedback(self):
        """Test feedback de sugerencias"""
        # Crear sugerencia
        suggestion_id = self.tracker.log_suggestion(
            agent_id="AGT_001",
            suggestion_text="Test suggestion",
            confidence=0.85,
            context={}
        )
        
        # Agregar feedback
        self.tracker.log_suggestion_feedback(
            suggestion_id=suggestion_id,
            was_used=True,
            effectiveness_rating=4,
            call_outcome="promise"
        )
        
        # Verificar en base de datos
        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT was_used, effectiveness_rating, call_outcome FROM suggestions WHERE suggestion_id = ?",
                (suggestion_id,)
            )
            result = cursor.fetchone()
            
        self.assertEqual(result[0], 1)  # was_used = True
        self.assertEqual(result[1], 4)  # effectiveness_rating
        self.assertEqual(result[2], "promise")  # call_outcome
    
    def test_log_call_metrics(self):
        """Test logging de métricas de llamada"""
        call_data = {
            "agent_id": "AGT_001",
            "debtor_id": "DBT_001",
            "duration_seconds": 300,
            "outcome": "promise",
            "amount_promised": 150.0,
            "suggestions_shown": 2,
            "suggestions_used": 1
        }
        
        call_id = self.tracker.log_call_metrics(call_data)
        
        self.assertIsNotNone(call_id)
        
        # Verificar en base de datos
        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT outcome, amount_promised FROM calls WHERE call_id = ?", (call_id,))
            result = cursor.fetchone()
            
        self.assertEqual(result[0], "promise")
        self.assertEqual(result[1], 150.0)
    
    def test_analyze_suggestion_effectiveness(self):
        """Test análisis de efectividad de sugerencias"""
        # Crear algunas sugerencias con feedback
        for i in range(5):
            suggestion_id = self.tracker.log_suggestion(
                agent_id="AGT_001",
                suggestion_text=f"Test suggestion {i}",
                confidence=0.8 + (i * 0.05),
                context={"objection_type": "cannot_pay", "debtor_profile": "cooperative"}
            )
            
            self.tracker.log_suggestion_feedback(
                suggestion_id=suggestion_id,
                was_used=i % 2 == 0,  # 3 usadas, 2 no usadas
                effectiveness_rating=3 + i % 3,  # Ratings: 3,4,5,3,4
                call_outcome="promise" if i % 2 == 0 else "no_contact"
            )
        
        analysis = self.tracker.analyze_suggestion_effectiveness(agent_id="AGT_001")
        
        self.assertIn("total_suggestions", analysis)
        self.assertIn("overall_usage_rate", analysis)
        self.assertIn("by_objection_type", analysis)
        
        self.assertEqual(analysis["total_suggestions"], 5)
        self.assertAlmostEqual(analysis["overall_usage_rate"], 0.6, places=1)  # 3/5


class TestCRMIntegration(unittest.TestCase):
    """Tests para integraciones CRM"""
    
    @patch('requests.Session')
    def test_totals_connector_authentication(self, mock_session_class):
        """Test autenticación con TOTALS"""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock respuesta de autenticación exitosa
        mock_response = Mock()
        mock_response.json.return_value = {"access_token": "test_token"}
        mock_response.raise_for_status.return_value = None
        mock_session.post.return_value = mock_response
        
        connector = TOTALSConnector(
            api_base_url="https://api.test.com",
            username="test_user",
            password="test_pass",
            client_id="test_client"
        )
        
        self.assertEqual(connector.auth_token, "test_token")
        mock_session.post.assert_called_once()
    
    def test_call_outcome_creation(self):
        """Test creación de CallOutcome"""
        outcome = CallOutcome(
            call_id="TEST_001",
            debtor_id="DBT_001",
            agent_id="AGT_001",
            timestamp=datetime.now(),
            duration_seconds=300,
            outcome_code="PTP",
            disposition="Promise to Pay",
            amount_promised=150.0,
            payment_date=datetime.now() + timedelta(days=7),
            next_action="Follow up",
            notes="Customer agreed to payment",
            compliance_score=0.95,
            ai_suggestions_used=2
        )
        
        self.assertEqual(outcome.call_id, "TEST_001")
        self.assertEqual(outcome.outcome_code, "PTP")
        self.assertEqual(outcome.amount_promised, 150.0)
        self.assertEqual(outcome.ai_suggestions_used, 2)


if __name__ == "__main__":
    # Configurar logging para tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Ejecutar tests
    unittest.main(verbosity=2)
