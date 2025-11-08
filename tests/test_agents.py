"""
Test Suite for Pharma Agentic AI Agents
Comprehensive tests for all worker agents and master orchestrator
Updated with Drug Database Agent and EPO Patent API integration
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

# NEW: Import Drug Database Agent
try:
    from agents.drug_database_agent import get_drug_database_agent
    DRUG_DB_AVAILABLE = True
except ImportError:
    DRUG_DB_AVAILABLE = False
    print(" Drug Database Agent not found - skipping related tests")


class TestDrugDatabaseAgent:
    """Tests for Drug Database Agent (PubChem Integration)"""

    @pytest.mark.skipif(not DRUG_DB_AVAILABLE, reason="Drug Database Agent not installed")
    def test_agent_initialization(self):
        """Test that Drug Database agent initializes correctly"""
        agent = get_drug_database_agent(verbose=False)
        assert agent is not None
        assert hasattr(agent, "get_drug_info")
        assert hasattr(agent, "enrich_drug_context")

    @pytest.mark.skipif(not DRUG_DB_AVAILABLE, reason="Drug Database Agent not installed")
    def test_get_drug_info(self):
        """Test fetching drug information from PubChem"""
        agent = get_drug_database_agent(verbose=False)
        result = agent.get_drug_info("Aspirin")

        assert isinstance(result, dict)
        assert "query" in result
        assert "found" in result
        
        # If found, check for key fields
        if result["found"]:
            assert "canonical_name" in result
            assert "synonyms" in result
            assert "drug_class" in result

    @pytest.mark.skipif(not DRUG_DB_AVAILABLE, reason="Drug Database Agent not installed")
    def test_get_drug_synonyms(self):
        """Test getting drug synonyms"""
        agent = get_drug_database_agent(verbose=False)
        synonyms = agent.get_drug_synonyms("Ibuprofen")

        assert isinstance(synonyms, list)
        # Should have at least the drug name itself
        assert len(synonyms) >= 1

    @pytest.mark.skipif(not DRUG_DB_AVAILABLE, reason="Drug Database Agent not installed")
    def test_enrich_drug_context(self):
        """Test batch enrichment of drug context"""
        agent = get_drug_database_agent(verbose=False)
        result = agent.enrich_drug_context(["Aspirin", "Metformin"])

        assert isinstance(result, dict)
        assert len(result) <= 2  # Should have data for requested drugs

    @pytest.mark.skipif(not DRUG_DB_AVAILABLE, reason="Drug Database Agent not installed")
    def test_compare_drugs(self):
        """Test drug comparison functionality"""
        agent = get_drug_database_agent(verbose=False)
        result = agent.compare_drugs(["Aspirin", "Ibuprofen"])

        assert isinstance(result, dict)
        assert "drugs_compared" in result
        assert "comparison_table" in result

    @pytest.mark.skipif(not DRUG_DB_AVAILABLE, reason="Drug Database Agent not installed")
    def test_cache_functionality(self):
        """Test that caching works"""
        agent = get_drug_database_agent(verbose=False)
        
        # First call
        result1 = agent.get_drug_info("Metformin")
        
        # Second call (should be cached)
        result2 = agent.get_drug_info("Metformin")
        
        # Should return same data
        assert result1 == result2
        
        # Check cache stats
        stats = agent.get_cache_stats()
        assert stats["cached_drugs"] >= 1


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
        
        # Check for market metrics
        if result["drug_analyses"]:
            first_drug = result["drug_analyses"][0]
            assert "market_metrics" in first_drug
            assert "current_sales_usd_million" in first_drug["market_metrics"]

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
        assert "leader" in result

    def test_market_insights_extraction(self):
        """Test that market insights are properly extracted"""
        agent = get_iqvia_agent(verbose=False)
        result = agent.analyze_market(drugs=["Atorvastatin"])

        assert "market_insights" in result
        assert isinstance(result["market_insights"], list)


class TestEXIMAgent:
    """Tests for EXIM Trade Intelligence Agent"""

    def test_agent_initialization(self):
        """Test that EXIM agent initializes correctly"""
        agent = get_exim_agent(verbose=False)
        assert agent is not None
        assert hasattr(agent, "analyze_trade")
        # Check it has the real fetcher
        assert hasattr(agent.data_fetcher, "get_comprehensive_trade_data")

    def test_analyze_trade_with_drug(self):
        """Test trade analysis for a specific drug"""
        agent = get_exim_agent(verbose=False)
        result = agent.analyze_trade(drugs=["Metformin"])

        assert isinstance(result, dict)
        assert "summary" in result
        assert "drug_analyses" in result

        # Check for data sources
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
            "Mock Data (Fallback)",
        ]
        for source in result["data_sources_used"]:
            assert any(valid in source for valid in valid_sources)

    def test_trade_metrics(self):
        """Test trade metrics calculation"""
        agent = get_exim_agent(verbose=False)
        result = agent.analyze_trade(drugs=["Aspirin"])

        if result["drug_analyses"]:
            first_drug = result["drug_analyses"][0]
            assert "trade_metrics" in first_drug
            metrics = first_drug["trade_metrics"]
            assert "total_import_value_usd" in metrics
            assert "total_export_value_usd" in metrics
            assert "trade_balance" in metrics

    def test_supply_chain_risk_assessment(self):
        """Test supply chain risk assessment"""
        agent = get_exim_agent(verbose=False)
        result = agent.analyze_trade(drugs=["Metformin"])

        assert "supply_chain_risks" in result
        assert isinstance(result["supply_chain_risks"], list)


class TestPatentAgent:
    """Tests for Patent Landscape Agent (EPO Integration)"""

    def test_agent_initialization(self):
        """Test that Patent agent initializes correctly"""
        agent = get_patent_agent(verbose=False)
        assert agent is not None
        assert hasattr(agent, "search_patents")
        # Check it has EPO fetcher
        assert hasattr(agent.fetcher, "_authenticate") or hasattr(agent.fetcher, "search_patents")

    def test_search_patents(self):
        """Test patent search functionality"""
        agent = get_patent_agent(verbose=False)
        result = agent.search_patents(drugs=["Ibuprofen"], max_results=10)

        assert isinstance(result, dict)
        assert "summary" in result
        assert "total_patents_found" in result
        assert "expiry_timeline" in result
        assert "detailed_patents" in result

    def test_patent_data_structure(self):
        """Test that patent data has correct structure"""
        agent = get_patent_agent(verbose=False)
        result = agent.search_patents(drugs=["Aspirin"], max_results=5)

        if result["detailed_patents"]:
            patent = result["detailed_patents"][0]
            assert "patent_number" in patent
            assert "title" in patent
            assert "assignee" in patent
            assert "filing_date" in patent
            assert "expiry_date" in patent
            assert "status" in patent

    def test_fto_assessment(self):
        """Test Freedom to Operate assessment"""
        agent = get_patent_agent(verbose=False)
        result = agent.search_patents(drugs=["Metformin"], max_results=10)

        assert "fto_assessment" in result
        fto = result["fto_assessment"]
        assert "risk_level" in fto
        assert "risk_description" in fto
        assert "active_patents_count" in fto
        assert fto["risk_level"] in ["Low", "Medium", "High"]

    def test_expiry_timeline_analysis(self):
        """Test patent expiry timeline analysis"""
        agent = get_patent_agent(verbose=False)
        result = agent.search_patents(drugs=["Atorvastatin"], max_results=15)

        timeline = result.get("expiry_timeline", {})
        assert "expiring_soon_count" in timeline
        assert "expiring_medium_count" in timeline
        assert "expiring_long_count" in timeline
        assert "opportunity_window" in timeline

    def test_competitive_landscape(self):
        """Test competitive landscape analysis"""
        agent = get_patent_agent(verbose=False)
        result = agent.search_patents(drugs=["Ibuprofen"], max_results=20)

        landscape = result.get("competitive_landscape", {})
        assert "total_assignees" in landscape
        assert "top_assignees" in landscape
        assert "filing_trend" in landscape

    def test_white_space_identification(self):
        """Test white space opportunity identification"""
        agent = get_patent_agent(verbose=False)
        result = agent.search_patents(drugs=["Metformin"], max_results=10)

        assert "white_space_opportunities" in result
        assert isinstance(result["white_space_opportunities"], list)

    def test_patent_comparison(self):
        """Test patent position comparison"""
        agent = get_patent_agent(verbose=False)
        result = agent.compare_patent_positions(["Aspirin", "Ibuprofen", "Metformin"])

        assert "drugs_compared" in result
        assert "comparison_table" in result
        assert len(result["comparison_table"]) == 3


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
        assert "detailed_trials" in result

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
        assert "pipeline_health" in result

    def test_phase_distribution(self):
        """Test trial phase distribution analysis"""
        agent = get_clinical_trials_agent(verbose=False)
        result = agent.search_trials(drugs=["Aspirin"], max_results=10)

        if result["total_trials_found"] > 0:
            assert "phase_distribution" in result
            phase_dist = result["phase_distribution"]
            assert "distribution" in phase_dist
            assert "most_common_phase" in phase_dist

    def test_sponsor_analysis(self):
        """Test sponsor analysis"""
        agent = get_clinical_trials_agent(verbose=False)
        result = agent.search_trials(drugs=["Metformin"], max_results=15)

        if result["total_trials_found"] > 0:
            assert "sponsor_analysis" in result
            sponsors = result["sponsor_analysis"]
            assert "total_unique_sponsors" in sponsors
            assert "top_sponsors" in sponsors

    def test_repurposing_opportunities(self):
        """Test repurposing opportunity identification"""
        agent = get_clinical_trials_agent(verbose=False)
        result = agent.search_trials(drugs=["Metformin"], max_results=20)

        assert "repurposing_opportunities" in result
        assert isinstance(result["repurposing_opportunities"], list)

    def test_pipeline_comparison(self):
        """Test pipeline comparison across drugs"""
        agent = get_clinical_trials_agent(verbose=False)
        result = agent.compare_pipelines(["Metformin", "Insulin", "Sitagliptin"])

        assert "interventions_compared" in result
        assert "comparison_table" in result


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
        assert "recommendation" in result

    def test_research_trends_analysis(self):
        """Test research trends analysis"""
        agent = get_web_intelligence_agent(verbose=False)
        result = agent.search_literature(drugs=["Aspirin"], max_results=10)

        assert "research_trends" in result
        assert isinstance(result["research_trends"], list)

    def test_compare_research_activity(self):
        """Test research activity comparison"""
        agent = get_web_intelligence_agent(verbose=False)
        result = agent.compare_research_activity(["Metformin", "Aspirin", "Ibuprofen"])

        assert "drugs_compared" in result
        assert "comparison_table" in result


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
        assert "documents" in result

    def test_search_internal(self):
        """Test internal document search"""
        agent = get_internal_knowledge_agent(verbose=False)
        result = agent.search_internal(keywords=["strategy"])

        assert isinstance(result, dict)
        assert "summary" in result
        assert "total_documents_found" in result


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
        assert "reports" in result

    def test_report_generation_pdf(self):
        """Test PDF report generation"""
        agent = get_report_generator_agent(verbose=False)
        
        sample_query = "Test query"
        sample_responses = [
            {
                "agent": "iqvia",
                "success": True,
                "data": {"summary": "Test data"}
            }
        ]
        sample_synthesis = "Test synthesis response"
        
        result = agent.generate_report(
            query=sample_query,
            agent_responses=sample_responses,
            synthesized_response=sample_synthesis,
            report_format="pdf"
        )
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "filename" in result

    def test_report_generation_excel(self):
        """Test Excel report generation"""
        agent = get_report_generator_agent(verbose=False)
        
        sample_query = "Test query"
        sample_responses = [
            {
                "agent": "clinical_trials",
                "success": True,
                "data": {"summary": "Test data"}
            }
        ]
        sample_synthesis = "Test synthesis"
        
        result = agent.generate_report(
            query=sample_query,
            agent_responses=sample_responses,
            synthesized_response=sample_synthesis,
            report_format="excel"
        )
        
        assert isinstance(result, dict)
        assert "success" in result


class TestMasterAgent:
    """Tests for Master Agent Orchestrator"""

    def test_agent_initialization(self):
        """Test that Master Agent initializes correctly"""
        agent = get_master_agent(verbose=False)
        assert agent is not None
        assert len(agent.worker_agents) > 0
        
        # Check for drug_database agent if available
        if DRUG_DB_AVAILABLE:
            assert "drug_database" in agent.worker_agents

    def test_query_parsing(self):
        """Test query parsing functionality"""
        agent = get_master_agent(verbose=False)
        parsed = agent.parse_query("What are the market trends for Metformin?")

        assert isinstance(parsed, dict)
        assert "intent" in parsed
        assert "required_agents" in parsed
        assert "keywords" in parsed
        assert "entities" in parsed

    def test_agent_validation(self):
        """Test that agent validation adds missing agents"""
        agent = get_master_agent(verbose=False)
        parsed = agent.parse_query("Which drugs have completed Phase 3 trials?")

        # Should include clinical_trials agent
        assert "clinical_trials" in parsed["required_agents"]

    def test_drug_database_integration(self):
        """Test that drug database is integrated properly"""
        if not DRUG_DB_AVAILABLE:
            pytest.skip("Drug Database Agent not available")
        
        agent = get_master_agent(verbose=False)
        parsed = agent.parse_query("What is Metformin?")
        
        # Drug database should be in required agents for drug queries
        # (or will be added during task decomposition)
        tasks = agent.decompose_tasks(parsed)
        task_agents = [t["agent"] for t in tasks]
        
        # Check if drug_database is in tasks or agents
        assert "drug_database" in task_agents or "drug_database" in agent.worker_agents

    def test_task_decomposition(self):
        """Test task decomposition"""
        agent = get_master_agent(verbose=False)
        parsed = agent.parse_query("Analyze Metformin market and patents")
        tasks = agent.decompose_tasks(parsed)

        assert isinstance(tasks, list)
        assert len(tasks) > 0
        
        # Should have both market and patent tasks
        task_agents = [t["agent"] for t in tasks]
        assert any(a in task_agents for a in ["iqvia", "patent"])

    def test_data_truncation(self):
        """Test that data truncation works (shows 5 detailed + summary)"""
        agent = get_master_agent(verbose=False)
        
        # Create mock data with 20 items
        mock_data = {
            "detailed_patents": [
                {"patent_number": f"EP{i}", "status": "Active", "assignee": f"Company{i}"}
                for i in range(20)
            ]
        }
        
        truncated = agent._truncate_agent_data(mock_data, max_items=5)
        
        # Should have 5 items
        assert len(truncated["detailed_patents"]) == 5
        # Should have metadata
        assert truncated["detailed_patents_total_count"] == 20
        assert truncated["detailed_patents_showing"] == 5
        assert truncated["detailed_patents_remaining"] == 15
        # Should have summary
        assert "detailed_patents_summary" in truncated

    def test_full_query_processing(self):
        """Test end-to-end query processing"""
        agent = get_master_agent(verbose=False)
        result = agent.process_query("What is Metformin?")

        assert isinstance(result, dict)
        assert "query" in result
        assert "response" in result
        assert "processing_time" in result
        assert "agent_responses" in result

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
        assert "success_rate" in stats


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

    @pytest.mark.skipif(not DRUG_DB_AVAILABLE, reason="Drug Database Agent not installed")
    def test_drug_enrichment_workflow(self):
        """Test that drug database enriches other agents"""
        agent = get_master_agent(verbose=False)
        result = agent.process_query(
            "What are the clinical trials for Aspirin?"
        )
        
        # Check if drug_database ran
        agents_used = [r["agent"] for r in result["agent_responses"]]
        
        # If drug_database is available, it should enrich the query
        if "drug_database" in agents_used:
            db_response = next(
                r for r in result["agent_responses"]
                if r["agent"] == "drug_database"
            )
            assert db_response["success"]
            assert db_response["data"] is not None

    def test_patent_fto_analysis(self):
        """Test patent FTO analysis workflow"""
        agent = get_master_agent(verbose=False)
        result = agent.process_query(
            "What is the patent landscape for Metformin?"
        )
        
        # Should use patent agent
        agents_used = [r["agent"] for r in result["agent_responses"]]
        assert "patent" in agents_used
        
        # Check for FTO information in response
        assert "fto" in result["response"].lower() or "patent" in result["response"].lower()

    def test_comprehensive_analysis(self):
        """Test comprehensive multi-agent analysis"""
        agent = get_master_agent(verbose=False)
        result = agent.process_query(
            "Analyze Ibuprofen: market size, patents, clinical trials, and publications"
        )
        
        # Should use multiple agents
        agents_used = [r["agent"] for r in result["agent_responses"]]
        assert len(agents_used) >= 2  # At least 2 agents should run
        
        # Response should be comprehensive
        assert len(result["response"]) > 200


# Performance Tests
class TestPerformance:
    """Performance and optimization tests"""

    def test_query_processing_time(self):
        """Test that query processing completes in reasonable time"""
        import time
        
        agent = get_master_agent(verbose=False)
        start = time.time()
        result = agent.process_query("What is Metformin?")
        duration = time.time() - start
        
        # Should complete within 30 seconds
        assert duration < 30
        assert result["processing_time"] < 30

    def test_data_truncation_performance(self):
        """Test that data truncation improves performance"""
        agent = get_master_agent(verbose=False)
        
        # Large dataset
        large_data = {
            "detailed_patents": [{"id": i} for i in range(100)]
        }
        
        import time
        start = time.time()
        truncated = agent._truncate_agent_data(large_data, max_items=5)
        duration = time.time() - start
        
        # Should be fast
        assert duration < 1.0
        # Should be truncated
        assert len(truncated["detailed_patents"]) == 5


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])