"""
Test Suite for Pharma Agentic AI Agents
Comprehensive tests for all worker agents and master orchestrator
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.clinical_trials_agent import get_clinical_trials_agent
from agents.exim_agent import get_exim_agent
from agents.internal_knowledge_agent import get_internal_knowledge_agent
from agents.iqvia_agent import get_iqvia_agent
from agents.master_agent import get_master_agent
from agents.patent_agent import get_patent_agent
from agents.report_generator_agent import get_report_generator_agent
from agents.web_intelligence_agent import get_web_intelligence_agent


class TestIQVIAAgent:
    """Tests for IQVIA Market Intelligence Agent"""

    def test_agent_initialization(self):
        """Test that IQVIA agent initializes correctly"""
        agent = get_iqvia_agent(verbose=False)
        assert agent is not None
        assert hasattr(agent, "analyze_market")

    def test_analyze_market_with_drug(self):
        """Test market analysis for a specific drug"""
        agent = get_iqvia_agent(verbose=False)
        result = agent.analyze_market(drugs=["Metformin"])

        assert isinstance(result, dict)
        assert "summary" in result
        assert "drug_analyses" in result

    def test_analyze_therapeutic_area(self):
        """Test market analysis by therapeutic area"""
        agent = get_iqvia_agent(verbose=False)
        result = agent.analyze_market(therapeutic_areas=["Diabetes"])

        assert isinstance(result, dict)
        assert "therapeutic_area_analyses" in result

    def test_compare_drugs(self):
        """Test drug comparison functionality"""
        agent = get_iqvia_agent(verbose=False)
        result = agent.compare_drugs(["Metformin", "Insulin Glargine"])

        assert isinstance(result, dict)
        assert "comparison_table" in result


class TestEXIMAgent:
    """Tests for EXIM Trade Intelligence Agent"""

    def test_agent_initialization(self):
        """Test that EXIM agent initializes correctly"""
        agent = get_exim_agent(verbose=False)
        assert agent is not None
        assert hasattr(agent, "analyze_trade")
        # NEW: Check it has the real fetcher
        assert hasattr(agent.data_fetcher, "get_comprehensive_trade_data")

    def test_analyze_trade_with_drug(self):
        """Test trade analysis for a specific drug"""
        agent = get_exim_agent(verbose=False)
        result = agent.analyze_trade(drugs=["Metformin"])

        assert isinstance(result, dict)
        assert "summary" in result
        assert "drug_analyses" in result

        # NEW: Check for data sources
        if result["drug_analyses"]:
            first_drug = result["drug_analyses"][0]
            assert "data_sources" in first_drug
            # Should have either real API sources or mock fallback
            assert len(first_drug["data_sources"]) > 0

    def test_data_sources_tracking(self):
        """Test that data sources are properly tracked"""
        agent = get_exim_agent(verbose=False)
        result = agent.analyze_trade(drugs=["Metformin"])

        # Should have data_sources_used at top level
        assert "data_sources_used" in result
        assert isinstance(result["data_sources_used"], list)

        # Sources should be from real APIs or mock
        valid_sources = [
            "UN Comtrade",
            "World Bank",
            "USITC DataWeb",
            "Mock Data (Fallback)",
        ]
        for source in result["data_sources_used"]:
            assert any(valid in source for valid in valid_sources)


class TestPatentAgent:
    """Tests for Patent Landscape Agent"""

    def test_agent_initialization(self):
        """Test that Patent agent initializes correctly"""
        agent = get_patent_agent(verbose=False)
        assert agent is not None
        assert hasattr(agent, "search_patents")

    def test_search_patents(self):
        """Test patent search functionality"""
        agent = get_patent_agent(verbose=False)
        result = agent.search_patents(drugs=["Ibuprofen"])

        assert isinstance(result, dict)
        assert "summary" in result
        assert "total_patents_found" in result
        assert "expiry_timeline" in result

    def test_fto_assessment(self):
        """Test Freedom to Operate assessment"""
        agent = get_patent_agent(verbose=False)
        result = agent.search_patents(drugs=["Metformin"])

        assert "fto_assessment" in result
        assert "risk_level" in result["fto_assessment"]


class TestClinicalTrialsAgent:
    """Tests for Clinical Trials Agent"""

    def test_agent_initialization(self):
        """Test that Clinical Trials agent initializes correctly"""
        agent = get_clinical_trials_agent(verbose=False)
        assert agent is not None
        assert hasattr(agent, "search_trials")

    def test_search_trials_by_drug(self):
        """Test trial search by drug name"""
        agent = get_clinical_trials_agent(verbose=False)
        result = agent.search_trials(drugs=["Metformin"], max_results=5)

        assert isinstance(result, dict)
        assert "summary" in result
        assert "total_trials_found" in result

    def test_search_trials_by_disease(self):
        """Test trial search by disease"""
        agent = get_clinical_trials_agent(verbose=False)
        result = agent.search_trials(diseases=["Diabetes"], max_results=5)

        assert isinstance(result, dict)
        assert "total_trials_found" in result

    def test_pipeline_analysis(self):
        """Test pipeline analysis"""
        agent = get_clinical_trials_agent(verbose=False)
        result = agent.analyze_pipeline("Metformin")

        assert isinstance(result, dict)
        assert "intervention" in result
        assert "development_stage" in result


class TestWebIntelligenceAgent:
    """Tests for Web Intelligence Agent"""

    def test_agent_initialization(self):
        """Test that Web Intelligence agent initializes correctly"""
        agent = get_web_intelligence_agent(verbose=False)
        assert agent is not None
        assert hasattr(agent, "search_literature")

    def test_search_literature(self):
        """Test literature search"""
        agent = get_web_intelligence_agent(verbose=False)
        result = agent.search_literature(keywords=["metformin"], max_results=3)

        assert isinstance(result, dict)
        assert "summary" in result
        assert "total_publications_found" in result

    def test_repurposing_evidence(self):
        """Test repurposing evidence search"""
        agent = get_web_intelligence_agent(verbose=False)
        result = agent.search_repurposing_evidence("Metformin", "Alzheimer")

        assert isinstance(result, dict)
        assert "evidence_strength" in result
        assert "repurposing_potential" in result


class TestInternalKnowledgeAgent:
    """Tests for Internal Knowledge Agent"""

    def test_agent_initialization(self):
        """Test that Internal Knowledge agent initializes correctly"""
        agent = get_internal_knowledge_agent(verbose=False)
        assert agent is not None
        assert hasattr(agent, "search_internal")

    def test_list_documents(self):
        """Test document listing"""
        agent = get_internal_knowledge_agent(verbose=False)
        result = agent.list_all_documents()

        assert isinstance(result, dict)
        assert "total_documents" in result

    def test_search_internal(self):
        """Test internal document search"""
        agent = get_internal_knowledge_agent(verbose=False)
        result = agent.search_internal(keywords=["strategy"])

        assert isinstance(result, dict)
        assert "summary" in result


class TestReportGeneratorAgent:
    """Tests for Report Generator Agent"""

    def test_agent_initialization(self):
        """Test that Report Generator agent initializes correctly"""
        agent = get_report_generator_agent(verbose=False)
        assert agent is not None
        assert hasattr(agent, "generate_report")

    def test_list_reports(self):
        """Test report listing"""
        agent = get_report_generator_agent(verbose=False)
        result = agent.list_reports()

        assert isinstance(result, dict)
        assert "total_reports" in result


class TestMasterAgent:
    """Tests for Master Agent Orchestrator"""

    def test_agent_initialization(self):
        """Test that Master Agent initializes correctly"""
        agent = get_master_agent(verbose=False)
        assert agent is not None
        assert len(agent.worker_agents) > 0

    def test_query_parsing(self):
        """Test query parsing functionality"""
        agent = get_master_agent(verbose=False)
        parsed = agent.parse_query("What are the market trends for Metformin?")

        assert isinstance(parsed, dict)
        assert "intent" in parsed
        assert "required_agents" in parsed
        assert "keywords" in parsed

    def test_agent_validation(self):
        """Test that agent validation adds missing agents"""
        agent = get_master_agent(verbose=False)
        parsed = agent.parse_query("Which drugs have completed Phase 3 trials?")

        # Should include clinical_trials agent
        assert "clinical_trials" in parsed["required_agents"]

    def test_task_decomposition(self):
        """Test task decomposition"""
        agent = get_master_agent(verbose=False)
        parsed = agent.parse_query("Analyze Metformin market and patents")
        tasks = agent.decompose_tasks(parsed)

        assert isinstance(tasks, list)
        assert len(tasks) > 0

    def test_full_query_processing(self):
        """Test end-to-end query processing"""
        agent = get_master_agent(verbose=False)
        result = agent.process_query("What is Metformin?")

        assert isinstance(result, dict)
        assert "query" in result
        assert "response" in result
        assert "processing_time" in result

    def test_query_history(self):
        """Test query history tracking"""
        agent = get_master_agent(verbose=False)
        agent.process_query("Test query 1")
        agent.process_query("Test query 2")

        history = agent.get_query_history()
        assert len(history) >= 2

    def test_statistics(self):
        """Test statistics generation"""
        agent = get_master_agent(verbose=False)
        agent.process_query("Test query")

        stats = agent.get_statistics()
        assert isinstance(stats, dict)
        assert "total_queries" in stats
        assert stats["total_queries"] >= 1


# Integration Tests
class TestIntegration:
    """Integration tests for multi-agent workflows"""

    def test_market_and_trials_query(self):
        """Test query that requires multiple agents"""
        agent = get_master_agent(verbose=False)
        result = agent.process_query(
            "Which oncology drugs have Phase 3 trials and good market potential?"
        )

        # Should use both clinical_trials and iqvia agents
        agents_used = [r["agent"] for r in result["agent_responses"]]
        assert "clinical_trials" in agents_used or "iqvia" in agents_used

    def test_repurposing_analysis(self):
        """Test repurposing opportunity analysis"""
        agent = get_master_agent(verbose=False)
        result = agent.process_query(
            "What are the repurposing opportunities for Metformin?"
        )

        assert result["response"] is not None
        assert len(result["response"]) > 100  # Should have substantial response


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
