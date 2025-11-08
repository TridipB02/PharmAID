# settings.py
"""
Configuration settings for Pharma Agentic AI System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
INTERNAL_DOCS_DIR = DATA_DIR / "internal_docs"

# Ensure directories exist
REPORTS_DIR.mkdir(exist_ok=True)
INTERNAL_DOCS_DIR.mkdir(exist_ok=True)

# Ollama Configuration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "model": "llama3.1:8b",
    "temperature": 0.5,
    "max_tokens": 3000,
    "num_predict": 3000,
}

# Data file paths
DATA_FILES = {
    "drugs_database": DATA_DIR / "drugs_database.json",
    "mock_iqvia": DATA_DIR / "mock_iqvia.json",
}

EPO_CONFIG = {
    "consumer_key": os.getenv("EPO_CONSUMER_KEY", ""),
    "consumer_secret": os.getenv("EPO_CONSUMER_SECRET", ""),
    "base_url": "https://ops.epo.org/3.2/rest-services",
     "auth_url": "https://ops.epo.org/3.2/auth/accesstoken",
}

# API Endpoints (Free sources)
API_ENDPOINTS = {
    # Clinical Trials
    "clinical_trials": "https://clinicaltrials.gov/api/v2/studies",
    # PubMed for scientific literature
    "pubmed_search": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
    "pubmed_fetch": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
    # FDA OpenFDA
    "openfda_drugs": "https://api.fda.gov/drug/event.json",
    "openfda_labels": "https://api.fda.gov/drug/label.json",
    # Trade Data APIs
    "un_comtrade": "https://comtradeapi.un.org/public/v1/getDA",
    "world_bank_trade": "https://api.worldbank.org/v2",
}

# Agent Configuration
AGENT_CONFIG = {
    "master_agent": {
        "name": "Master Orchestrator",
        "role": "Portfolio Planning Coordinator",
        "goal": "Orchestrate research tasks and synthesize insights",
        "verbose": True,
    },
    "iqvia_agent": {
        "name": "IQVIA Market Analyst",
        "role": "Market Intelligence Specialist",
        "goal": "Analyze market trends, sales data, and competition",
        "verbose": True,
    },
    "exim_agent": {
        "name": "Trade Intelligence Analyst",
        "role": "Import-Export Data Specialist",
        "goal": "Analyze global trade patterns and supply chain dynamics",
        "verbose": True,
    },
    "patent_agent": {
        "name": "Patent Landscape Analyst",
        "role": "Intellectual Property Specialist",
        "goal": "Analyze patent filings, expiry timelines, and FTO risks",
        "verbose": True,
    },
    "clinical_trials_agent": {
        "name": "Clinical Pipeline Analyst",
        "role": "Clinical Research Specialist",
        "goal": "Track ongoing trials, sponsors, and development pipeline",
        "verbose": True,
    },
    "internal_knowledge_agent": {
        "name": "Internal Knowledge Manager",
        "role": "Document Intelligence Specialist",
        "goal": "Extract insights from internal documents and reports",
        "verbose": True,
    },
    "web_intelligence_agent": {
        "name": "Web Intelligence Analyst",
        "role": "Real-time Information Specialist",
        "goal": "Gather current guidelines, publications, and market signals",
        "verbose": True,
    },
    "report_generator_agent": {
        "name": "Report Synthesis Specialist",
        "role": "Documentation and Reporting Expert",
        "goal": "Generate comprehensive PDF reports with visualizations",
        "verbose": True,
    },
}

# Report Generation Settings
REPORT_CONFIG = {
    "format": "pdf",
    "include_charts": True,
    "include_tables": True,
    "watermark": "PharmAID - Confidential",
    "max_pages": 50,
}

# Streamlit UI Configuration
UI_CONFIG = {
    "page_title": "PharmAID - Portfolio Planning Assistant",
    "page_icon": "",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": BASE_DIR / "app.log",
}

# Synthetic Test Queries (10 strategic questions)
TEST_QUERIES = [
    "Which respiratory diseases show low competition but high patient burden in India?",
    "Find molecules with expiring patents in the cardiovascular space that could be repurposed for diabetes",
    "What are the ongoing clinical trials for metformin in non-diabetes indications?",
    "Analyze the market potential for repurposing ibuprofen in novel dosage forms",
    "Which oncology drugs have completed Phase 3 trials but are not yet launched in emerging markets?",
    "Identify generic antibiotics with alternative indication opportunities based on recent scientific publications",
    "What are the export-import trends for SGLT2 inhibitors and their market dynamics?",
    "Find CNS drugs with expiring patents that show potential for pediatric formulations",
    "Analyze the competitive landscape for proton pump inhibitors with novel delivery mechanisms",
    "Which immunology drugs have successful off-label use documented in clinical literature?",
]

# Cache settings
CACHE_SETTINGS = {
    "enable_cache": True,
    "cache_ttl_hours": 24,
    "cache_dir": BASE_DIR / ".cache",
}

# Rate limiting for API calls
RATE_LIMIT = {
    "clinical_trials_per_minute": 10,
    "pubmed_per_second": 3,
    "uspto_per_minute": 5,
}

# Export settings
__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "REPORTS_DIR",
    "OLLAMA_CONFIG",
    "API_ENDPOINTS",
    "AGENT_CONFIG",
    "REPORT_CONFIG",
    "UI_CONFIG",
    "TEST_QUERIES",
    "EPO_CONFIG"
]
