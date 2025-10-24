"""
Master Agent - Conversation Orchestrator
Coordinates all worker agents and synthesizes responses

This is the central orchestrator that:
1. Parses user queries to understand intent
2. Decomposes queries into tasks for worker agents
3. Coordinates worker agent execution
4. Synthesizes responses into coherent summaries
5. Manages query history and context
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import re
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from config.settings import OLLAMA_CONFIG, AGENT_CONFIG

# Import worker agents (will be imported as they're created)
try:
    from agents.iqvia_agent import IQVIAAgent
except ImportError:
    IQVIAAgent = None

try:
    from agents.exim_agent import EXIMAgent
except ImportError:
    EXIMAgent = None

try:
    from agents.patent_agent import PatentAgent
except ImportError:
    PatentAgent = None

try:
    from agents.clinical_trials_agent import ClinicalTrialsAgent
except ImportError:
    ClinicalTrialsAgent = None

try:
    from agents.web_intelligence_agent import WebIntelligenceAgent
except ImportError:
    WebIntelligenceAgent = None

try:
    from agents.internal_knowledge_agent import InternalKnowledgeAgent
except ImportError:
    InternalKnowledgeAgent = None


class MasterAgentOrchestrator:
    """
    Master Agent that orchestrates all worker agents
    Implements intelligent query parsing, task decomposition, and response synthesis
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the Master Agent
        
        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.llm = self._initialize_llm()
        self.query_history = []
        self.worker_agents = self._initialize_worker_agents()
        
        if self.verbose:
            print("✓ Master Agent initialized successfully")
            print(f"✓ Available worker agents: {list(self.worker_agents.keys())}")
        
    def _initialize_llm(self) -> ChatOllama:
        """Initialize Ollama LLM"""
        try:
            llm = ChatOllama(
                model=OLLAMA_CONFIG['model'],
                base_url=OLLAMA_CONFIG['base_url'],
                temperature=OLLAMA_CONFIG['temperature'],
            )
            if self.verbose:
                print(f"✓ LLM initialized: {OLLAMA_CONFIG['model']}")
            return llm
        except Exception as e:
            print(f"✗ Error initializing LLM: {e}")
            raise
    
    def _initialize_worker_agents(self) -> Dict[str, Any]:
        """
        Initialize all available worker agents
        
        Returns:
            Dictionary of agent name to agent instance
        """
        agents = {}
        
        # Initialize IQVIA Agent
        if IQVIAAgent:
            try:
                agents['iqvia'] = IQVIAAgent()
                if self.verbose:
                    print("✓ IQVIA Agent loaded")
            except Exception as e:
                print(f"✗ Failed to load IQVIA Agent: {e}")
        
        # Initialize EXIM Agent
        if EXIMAgent:
            try:
                agents['exim'] = EXIMAgent()
                if self.verbose:
                    print("✓ EXIM Agent loaded")
            except Exception as e:
                print(f"✗ Failed to load EXIM Agent: {e}")
        
        # Initialize Patent Agent
        if PatentAgent:
            try:
                agents['patent'] = PatentAgent()
                if self.verbose:
                    print("✓ Patent Agent loaded")
            except Exception as e:
                print(f"✗ Failed to load Patent Agent: {e}")
        
        # Initialize Clinical Trials Agent
        if ClinicalTrialsAgent:
            try:
                agents['clinical_trials'] = ClinicalTrialsAgent()
                if self.verbose:
                    print("✓ Clinical Trials Agent loaded")
            except Exception as e:
                print(f"✗ Failed to load Clinical Trials Agent: {e}")
        
        # Initialize Web Intelligence Agent
        if WebIntelligenceAgent:
            try:
                agents['web_intelligence'] = WebIntelligenceAgent()
                if self.verbose:
                    print("✓ Web Intelligence Agent loaded")
            except Exception as e:
                print(f"✗ Failed to load Web Intelligence Agent: {e}")
        
        # Initialize Internal Knowledge Agent
        if InternalKnowledgeAgent:
            try:
                agents['internal_knowledge'] = InternalKnowledgeAgent()
                if self.verbose:
                    print("✓ Internal Knowledge Agent loaded")
            except Exception as e:
                print(f"✗ Failed to load Internal Knowledge Agent: {e}")
        
        return agents
    
    def parse_query(self, user_query: str) -> Dict[str, Any]:
        """
        Parse user query using LLM to understand intent and extract entities
        
        Args:
            user_query: Natural language query from user
        
        Returns:
            Dictionary with parsed information
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"QUERY PARSING")
            print(f"{'='*60}")
        
        # Enhanced parsing prompt
        parsing_prompt = f"""You are a pharmaceutical intelligence assistant. Analyze this query and extract structured information.

Query: "{user_query}"

IMPORTANT RULES FOR AGENT SELECTION:
- If query mentions "trial", "clinical", "phase", "study", or "pipeline" → MUST include "clinical_trials"
- If query mentions "market", "sales", "revenue", "launch", or "competition" → MUST include "iqvia"
- If query mentions "patent", "ip", or "expiry" → MUST include "patent"
- If query mentions "export", "import", or "trade" → MUST include "exim"
- If query mentions "publication", "research", or "literature" → MUST include "web_intelligence"
- Multiple agents can be selected if query touches multiple topics

Return a JSON object with these fields:

1. "intent": Choose the PRIMARY intent from:
   - "market_analysis", "patent_search", "clinical_trials", "drug_information",
   - "repurposing_opportunity", "competitive_analysis", "trade_analysis", 
   - "literature_search", "regulatory_inquiry", "general_inquiry"

2. "entities": {{
   "drugs": [list of drug names],
   "diseases": [list of diseases/conditions],
   "therapeutic_areas": [list of therapeutic areas like "Oncology", "Diabetes"],
   "countries": [list of countries],
   "companies": [list of pharmaceutical companies]
}}

3. "keywords": [5-10 most important keywords]

4. "required_agents": [list of agents from: "iqvia", "exim", "patent", "clinical_trials", "web_intelligence", "internal_knowledge"]

5. "time_scope": "current", "recent", "historical", "all_time", or "specific"

6. "priority": "high", "medium", or "low"

7. "expected_output": "summary", "detailed_report", "data_table", "comparison", or "visualization"

Return ONLY the JSON object, no other text.

Example:
{{
  "intent": "clinical_trials",
  "entities": {{"drugs": ["Metformin"], "diseases": ["Cancer"], "therapeutic_areas": ["Oncology"], "countries": [], "companies": []}},
  "keywords": ["metformin", "phase 3", "trials", "oncology", "completed"],
  "required_agents": ["clinical_trials", "iqvia"],
  "time_scope": "recent",
  "priority": "medium",
  "expected_output": "summary"
}}"""
        
        try:
            response = self.llm.invoke(parsing_prompt)
            content = response.content
            
            # Extract JSON from response
            parsed = self._extract_json_from_llm_response(content)
            
            if parsed:
                # CRITICAL: Validate and fix agent selection using keyword-based backup
                parsed = self._validate_and_fix_agents(user_query, parsed)
                
                if self.verbose:
                    print(f"✓ Query parsed successfully")
                    print(f"  Intent: {parsed.get('intent', 'unknown')}")
                    print(f"  Entities: {parsed.get('entities', {})}")
                    print(f"  Required Agents: {parsed.get('required_agents', [])}")
                return parsed
            else:
                # Fallback to keyword-based parsing
                if self.verbose:
                    print("⚠ LLM parsing failed, using fallback method")
                return self._fallback_parsing(user_query)
        
        except Exception as e:
            if self.verbose:
                print(f"✗ Error in LLM parsing: {e}")
            return self._fallback_parsing(user_query)
    
    def _validate_and_fix_agents(self, query: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate LLM's agent selection and add missing agents based on keywords
        
        Args:
            query: Original query string
            parsed: Parsed dictionary from LLM
        
        Returns:
            Fixed parsed dictionary with correct agents
        """
        query_lower = query.lower()
        required_agents = set(parsed.get('required_agents', []))
        
        # Check for clinical trials keywords
        if any(word in query_lower for word in ["trial", "clinical", "phase 1", "phase 2", "phase 3", "phase 4", "phase", "study", "pipeline"]):
            if "clinical_trials" not in required_agents:
                required_agents.add("clinical_trials")
                if self.verbose:
                    print("  ✓ Added 'clinical_trials' agent (detected from keywords)")
        
        # Check for market keywords
        if any(word in query_lower for word in ["market", "sales", "revenue", "launch", "competition", "emerging", "cagr"]):
            if "iqvia" not in required_agents:
                required_agents.add("iqvia")
                if self.verbose:
                    print("  ✓ Added 'iqvia' agent (detected from keywords)")
        
        # Check for patent keywords
        if any(word in query_lower for word in ["patent", "ip", "intellectual property", "expir", "fto"]):
            if "patent" not in required_agents:
                required_agents.add("patent")
                if self.verbose:
                    print("  ✓ Added 'patent' agent (detected from keywords)")
        
        # Check for trade keywords
        if any(word in query_lower for word in ["export", "import", "trade", "sourcing", "supply"]):
            if "exim" not in required_agents:
                required_agents.add("exim")
                if self.verbose:
                    print("  ✓ Added 'exim' agent (detected from keywords)")
        
        # Check for literature keywords
        if any(word in query_lower for word in ["publication", "research", "literature", "pubmed", "paper", "guideline"]):
            if "web_intelligence" not in required_agents:
                required_agents.add("web_intelligence")
                if self.verbose:
                    print("  ✓ Added 'web_intelligence' agent (detected from keywords)")
        
        # Update parsed dictionary
        parsed['required_agents'] = list(required_agents)
        return parsed
    
    def _extract_json_from_llm_response(self, content: str) -> Optional[Dict]:
        """
        Extract JSON from LLM response that might have extra text
        
        Args:
            content: LLM response content
        
        Returns:
            Parsed JSON dict or None
        """
        try:
            # Method 1: Try direct JSON parse
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        try:
            # Method 2: Find JSON object in text
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        try:
            # Method 3: Find JSON in code blocks
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except:
            pass
        
        return None
    
    def _fallback_parsing(self, user_query: str) -> Dict[str, Any]:
        """
        Fallback parsing using keyword matching and heuristics
        
        Args:
            user_query: User's query string
        
        Returns:
            Parsed query dictionary
        """
        query_lower = user_query.lower()
        
        # Determine intent
        intent = "general_inquiry"
        if any(word in query_lower for word in ["trial", "clinical", "phase", "study", "pipeline"]):
            intent = "clinical_trials"
        elif any(word in query_lower for word in ["market", "sales", "revenue", "competition", "share"]):
            intent = "market_analysis"
        elif any(word in query_lower for word in ["patent", "ip", "intellectual property", "expir", "fto"]):
            intent = "patent_search"
        elif any(word in query_lower for word in ["repurpose", "repurposing", "alternative", "new indication"]):
            intent = "repurposing_opportunity"
        elif any(word in query_lower for word in ["export", "import", "trade", "sourcing"]):
            intent = "trade_analysis"
        elif any(word in query_lower for word in ["publication", "research", "literature", "paper"]):
            intent = "literature_search"
        
        # Determine required agents - COMPREHENSIVE CHECK
        required_agents = []
        
        # Clinical trials
        if any(word in query_lower for word in ["trial", "clinical", "phase", "study", "pipeline"]):
            required_agents.append("clinical_trials")
        
        # Market analysis
        if any(word in query_lower for word in ["market", "sales", "revenue", "prescription", "cagr", "launch", "emerging", "competition"]):
            required_agents.append("iqvia")
        
        # Trade analysis
        if any(word in query_lower for word in ["export", "import", "trade", "sourcing", "supply"]):
            required_agents.append("exim")
        
        # Patent analysis
        if any(word in query_lower for word in ["patent", "ip", "expir", "intellectual property"]):
            required_agents.append("patent")
        
        # Literature search
        if any(word in query_lower for word in ["publication", "research", "guideline", "literature", "pubmed"]):
            required_agents.append("web_intelligence")
        
        # Internal knowledge
        if any(word in query_lower for word in ["internal", "company", "our"]):
            required_agents.append("internal_knowledge")
        
        # If no specific agents, use general set
        if not required_agents:
            required_agents = ["iqvia", "clinical_trials", "web_intelligence"]
        
        # Extract entities
        entities = self._extract_entities_heuristic(user_query)
        
        # Time scope
        time_scope = "all_time"
        if any(word in query_lower for word in ["recent", "current", "latest", "now", "today"]):
            time_scope = "recent"
        elif any(word in query_lower for word in ["last year", "past year", "2024", "2023"]):
            time_scope = "recent"
        elif any(word in query_lower for word in ["historical", "past", "trend"]):
            time_scope = "historical"
        
        return {
            "intent": intent,
            "entities": entities,
            "keywords": self._extract_keywords(user_query),
            "required_agents": required_agents,
            "time_scope": time_scope,
            "priority": "medium",
            "expected_output": "summary"
        }
    
    def _extract_entities_heuristic(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities using heuristics
        
        Args:
            query: User query
        
        Returns:
            Dictionary of entity types to lists of entities
        """
        # Common drug names (from our database)
        common_drugs = [
            "atorvastatin", "metoprolol", "losartan", "amlodipine", "clopidogrel",
            "metformin", "glimepiride", "sitagliptin", "insulin", "empagliflozin",
            "salbutamol", "budesonide", "montelukast", "tiotropium", "formoterol",
            "sertraline", "escitalopram", "risperidone", "donepezil", "gabapentin",
            "imatinib", "tamoxifen", "capecitabine", "paclitaxel", "carboplatin",
            "amoxicillin", "azithromycin", "ciprofloxacin", "oseltamivir", "acyclovir",
            "omeprazole", "pantoprazole", "ondansetron", "loperamide", "mesalamine",
            "ibuprofen", "diclofenac", "celecoxib", "tramadol", "acetaminophen",
            "adalimumab", "methotrexate", "hydroxychloroquine", "azathioprine", "prednisone",
            "levothyroxine", "alendronate", "testosterone", "estradiol", "cabergoline"
        ]
        
        query_lower = query.lower()
        
        # Find drugs
        drugs = [drug for drug in common_drugs if drug in query_lower]
        
        # Find therapeutic areas
        therapeutic_areas_keywords = {
            "Cardiovascular": ["heart", "cardiac", "hypertension", "blood pressure", "cholesterol", "cardiovascular"],
            "Diabetes": ["diabetes", "diabetic", "insulin", "glucose", "blood sugar"],
            "Respiratory": ["asthma", "copd", "respiratory", "lung", "breathing"],
            "CNS/Mental Health": ["depression", "anxiety", "mental health", "alzheimer", "epilepsy", "cns"],
            "Oncology": ["cancer", "oncology", "tumor", "chemotherapy", "oncological"],
            "Infectious Disease": ["infection", "antibiotic", "antiviral", "bacterial", "infectious"],
            "Gastroenterology": ["stomach", "gastric", "digestive", "gerd", "gastro"],
            "Pain/Inflammation": ["pain", "analgesic", "inflammation", "arthritis"],
            "Immunology": ["immune", "autoimmune", "rheumatoid", "lupus", "immunology"],
            "Endocrine": ["thyroid", "hormone", "osteoporosis", "endocrine"]
        }
        
        therapeutic_areas = []
        for area, keywords in therapeutic_areas_keywords.items():
            if any(kw in query_lower for kw in keywords):
                therapeutic_areas.append(area)
        
        # Find countries
        countries = []
        country_keywords = ["india", "usa", "us", "america", "china", "germany", "uk", "britain", 
                           "france", "japan", "brazil", "canada", "mexico", "emerging"]
        for country in country_keywords:
            if country in query_lower:
                countries.append(country.capitalize() if country != "usa" and country != "us" else "USA")
        
        return {
            "drugs": drugs,
            "diseases": [],  # Would need medical NER for this
            "therapeutic_areas": therapeutic_areas,
            "countries": list(set(countries)),
            "companies": []
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'which'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [
            word for word in words 
            if word not in stopwords and len(word) > 3
        ]
        
        return keywords[:10]  # Top 10 keywords
    
    def decompose_tasks(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompose query into specific tasks for worker agents
        
        Args:
            parsed_query: Parsed query information
        
        Returns:
            List of task dictionaries with agent assignments
        """
        tasks = []
        required_agents = parsed_query.get('required_agents', [])
        entities = parsed_query.get('entities', {})
        intent = parsed_query.get('intent', 'general_inquiry')
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TASK DECOMPOSITION")
            print(f"{'='*60}")
        
        # IQVIA Agent tasks
        if 'iqvia' in required_agents and 'iqvia' in self.worker_agents:
            task = {
                "agent": "iqvia",
                "description": "Retrieve market data, sales trends, and competitive analysis",
                "action": "analyze_market",
                "params": {
                    "drugs": entities.get('drugs', []),
                    "therapeutic_areas": entities.get('therapeutic_areas', []),
                    "time_scope": parsed_query.get('time_scope', 'recent')
                },
                "priority": 1  # High priority for market data
            }
            tasks.append(task)
            if self.verbose:
                print(f"✓ Task created: IQVIA Market Analysis")
        
        # EXIM Agent tasks
        if 'exim' in required_agents and 'exim' in self.worker_agents:
            task = {
                "agent": "exim",
                "description": "Analyze import/export trends and supply chain data",
                "action": "analyze_trade",
                "params": {
                    "drugs": entities.get('drugs', []),
                    "countries": entities.get('countries', [])
                },
                "priority": 2
            }
            tasks.append(task)
            if self.verbose:
                print(f"✓ Task created: EXIM Trade Analysis")
        
        # Patent Agent tasks
        if 'patent' in required_agents and 'patent' in self.worker_agents:
            task = {
                "agent": "patent",
                "description": "Search patent landscape and identify IP opportunities",
                "action": "search_patents",
                "params": {
                    "drugs": entities.get('drugs', []),
                    "keywords": parsed_query.get('keywords', [])
                },
                "priority": 1
            }
            tasks.append(task)
            if self.verbose:
                print(f"✓ Task created: Patent Landscape Analysis")
        
        # Clinical Trials Agent tasks
        if 'clinical_trials' in required_agents and 'clinical_trials' in self.worker_agents:
            task = {
                "agent": "clinical_trials",
                "description": "Search ongoing clinical trials and pipeline data",
                "action": "search_trials",
                "params": {
                    "drugs": entities.get('drugs', []),
                    "diseases": entities.get('diseases', []),
                    "therapeutic_areas": entities.get('therapeutic_areas', [])
                },
                "priority": 1
            }
            tasks.append(task)
            if self.verbose:
                print(f"✓ Task created: Clinical Trials Search")
        
        # Web Intelligence Agent tasks
        if 'web_intelligence' in required_agents and 'web_intelligence' in self.worker_agents:
            task = {
                "agent": "web_intelligence",
                "description": "Search scientific literature and guidelines",
                "action": "search_literature",
                "params": {
                    "keywords": parsed_query.get('keywords', []),
                    "drugs": entities.get('drugs', [])
                },
                "priority": 3
            }
            tasks.append(task)
            if self.verbose:
                print(f"✓ Task created: Literature Search")
        
        # Internal Knowledge Agent tasks
        if 'internal_knowledge' in required_agents and 'internal_knowledge' in self.worker_agents:
            task = {
                "agent": "internal_knowledge",
                "description": "Search internal documents and company knowledge",
                "action": "search_internal",
                "params": {
                    "keywords": parsed_query.get('keywords', [])
                },
                "priority": 2
            }
            tasks.append(task)
            if self.verbose:
                print(f"✓ Task created: Internal Knowledge Search")
        
        # Sort tasks by priority
        tasks.sort(key=lambda x: x.get('priority', 99))
        
        if self.verbose:
            print(f"\nTotal tasks: {len(tasks)}")
        
        return tasks
    
    def execute_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute tasks by calling worker agents
        
        Args:
            tasks: List of task dictionaries
        
        Returns:
            List of agent responses
        """
        responses = []
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TASK EXECUTION")
            print(f"{'='*60}")
        
        for i, task in enumerate(tasks, 1):
            agent_name = task['agent']
            action = task['action']
            params = task['params']
            
            if self.verbose:
                print(f"\n[{i}/{len(tasks)}] Executing: {agent_name} - {action}")
            
            # Get the worker agent
            agent = self.worker_agents.get(agent_name)
            
            if not agent:
                if self.verbose:
                    print(f"  ✗ Agent '{agent_name}' not available")
                responses.append({
                    "agent": agent_name,
                    "success": False,
                    "error": f"Agent '{agent_name}' not initialized",
                    "data": None
                })
                continue
            
            try:
                # Call the agent's action method
                if hasattr(agent, action):
                    result = getattr(agent, action)(**params)
                    responses.append({
                        "agent": agent_name,
                        "action": action,
                        "success": True,
                        "data": result,
                        "task_description": task.get('description', '')
                    })
                    if self.verbose:
                        print(f"  ✓ Success")
                else:
                    if self.verbose:
                        print(f"  ✗ Method '{action}' not found")
                    responses.append({
                        "agent": agent_name,
                        "success": False,
                        "error": f"Method '{action}' not found on agent",
                        "data": None
                    })
            
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Error: {str(e)}")
                responses.append({
                    "agent": agent_name,
                    "success": False,
                    "error": str(e),
                    "data": None
                })
        
        return responses
    
    def synthesize_responses(
        self, 
        agent_responses: List[Dict[str, Any]], 
        original_query: str,
        parsed_query: Dict[str, Any]
    ) -> str:
        """
        Synthesize responses from multiple agents into coherent summary
        
        Args:
            agent_responses: List of responses from worker agents
            original_query: Original user query
            parsed_query: Parsed query information
        
        Returns:
            Synthesized summary text
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RESPONSE SYNTHESIS")
            print(f"{'='*60}")
        
        if not agent_responses:
            return "No data available from agents. Please check agent initialization."
        
        # Filter successful responses
        successful_responses = [r for r in agent_responses if r.get('success', False)]
        
        if not successful_responses:
            error_summary = "\n".join([
                f"- {r['agent']}: {r.get('error', 'Unknown error')}" 
                for r in agent_responses if not r.get('success', False)
            ])
            return f"All agents failed to retrieve data:\n{error_summary}"
        
        # Compile data from successful agents
        compiled_data = []
        for response in successful_responses:
            agent_name = response['agent'].upper()
            data = response.get('data', {})
            task_desc = response.get('task_description', '')
            
            formatted_data = self._format_agent_data(agent_name, data)
            compiled_data.append(f"\n### {agent_name} AGENT\n{task_desc}\n{formatted_data}")
        
        # Create synthesis prompt
        synthesis_prompt = f"""You are a senior pharmaceutical portfolio planning analyst. Synthesize the following data into a comprehensive, actionable report.

ORIGINAL QUERY: "{original_query}"

QUERY INTENT: {parsed_query.get('intent', 'general_inquiry')}

DATA FROM AGENTS:
{''.join(compiled_data)}

INSTRUCTIONS:
1. Create a well-structured report with clear sections
2. Start with an Executive Summary (2-3 sentences)
3. Present key findings organized by topic
4. Include specific data points and metrics
5. Identify opportunities and risks
6. Provide actionable recommendations
7. Always cite the source agent for each claim (e.g., "According to IQVIA data...")
8. Use bullet points for clarity where appropriate
9. If data is missing or incomplete, acknowledge it
10. Be concise but thorough

FORMAT:
## Executive Summary
[Brief overview]

## Key Findings
[Organized findings with data]

## Opportunities
[Potential opportunities identified]

## Risks & Challenges
[Potential issues]

## Recommendations
[Actionable next steps]

## Data Sources
[List agents that provided data]

Write the report now:"""
        
        try:
            response = self.llm.invoke(synthesis_prompt)
            synthesized = response.content
            
            if self.verbose:
                print(f"✓ Response synthesized successfully")
            
            return synthesized
        
        except Exception as e:
            if self.verbose:
                print(f"✗ Error in synthesis: {e}")
            
            # Fallback: Return raw compiled data
            return f"# Analysis Results\n\n{chr(10).join(compiled_data)}"
    
    def _format_agent_data(self, agent_name: str, data: Any) -> str:
        """
        Format agent data for display
        
        Args:
            agent_name: Name of the agent
            data: Data returned by agent
        
        Returns:
            Formatted string
        """
        if data is None:
            return "No data available."
        
        if isinstance(data, dict):
            # Format dict as readable text
            formatted = []
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    formatted.append(f"**{key}:**\n{json.dumps(value, indent=2)}")
                else:
                    formatted.append(f"**{key}:** {value}")
            return "\n".join(formatted)
        
        elif isinstance(data, list):
            if not data:
                return "No results found."
            # Format list items
            return "\n".join([f"• {item}" for item in data[:10]])  # Limit to 10 items
        
        else:
            return str(data)
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Main orchestration method - processes query end-to-end
        
        Args:
            user_query: Natural language query from user
        
        Returns:
            Dictionary with complete result
        """
        import time
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"PROCESSING QUERY: {user_query}")
            print(f"{'='*70}")
        
        # Step 1: Parse query
        parsed_query = self.parse_query(user_query)
        
        # Step 2: Decompose into tasks
        tasks = self.decompose_tasks(parsed_query)
        
        # Step 3: Execute tasks
        agent_responses = self.execute_tasks(tasks)
        
        # Step 4: Synthesize responses
        final_response = self.synthesize_responses(
            agent_responses, 
            user_query, 
            parsed_query
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create result object
        result = {
            "query": user_query,
            "parsed_query": parsed_query,
            "tasks": tasks,
            "agent_responses": agent_responses,
            "response": final_response,
            "timestamp": self._get_timestamp(),
            "processing_time": round(processing_time, 2)
        }
        
        # Store in history
        self.query_history.append(result)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"QUERY COMPLETED in {processing_time:.2f}s")
            print(f"{'='*70}\n")
        
        return result
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in standard format"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get query processing history"""
        return self.query_history
    
    def clear_history(self):
        """Clear query history"""
        self.query_history = []
        if self.verbose:
            print("✓ Query history cleared")
    
    def get_latest_query(self) -> Optional[Dict[str, Any]]:
        """Get the most recent query result"""
        return self.query_history[-1] if self.query_history else None
    
    def export_history(self, filepath: str):
        """Export query history to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.query_history, f, indent=2, ensure_ascii=False)
            if self.verbose:
                print(f"✓ History exported to {filepath}")
        except Exception as e:
            print(f"✗ Error exporting history: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about query processing"""
        if not self.query_history:
            return {
                "total_queries": 0,
                "average_processing_time": 0,
                "most_common_intent": None,
                "most_used_agents": {}
            }
        
        # Calculate statistics
        total_queries = len(self.query_history)
        avg_time = sum(q.get('processing_time', 0) for q in self.query_history) / total_queries
        
        # Intent distribution
        intents = [q['parsed_query']['intent'] for q in self.query_history]
        most_common_intent = max(set(intents), key=intents.count) if intents else None
        
        # Agent usage
        agent_usage = {}
        for query in self.query_history:
            for response in query.get('agent_responses', []):
                agent_name = response.get('agent', 'unknown')
                agent_usage[agent_name] = agent_usage.get(agent_name, 0) + 1
        
        return {
            "total_queries": total_queries,
            "average_processing_time": round(avg_time, 2),
            "most_common_intent": most_common_intent,
            "most_used_agents": agent_usage,
            "success_rate": self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate percentage of successful agent calls"""
        if not self.query_history:
            return 0.0
        
        total_calls = 0
        successful_calls = 0
        
        for query in self.query_history:
            for response in query.get('agent_responses', []):
                total_calls += 1
                if response.get('success', False):
                    successful_calls += 1
        
        return round((successful_calls / total_calls * 100), 2) if total_calls > 0 else 0.0
    
    def ask(self, query: str) -> str:
        """
        Convenience method to ask a question and get just the response text
        
        Args:
            query: User's question
        
        Returns:
            Response text
        """
        result = self.process_query(query)
        return result['response']
    
    def __str__(self) -> str:
        """String representation of Master Agent"""
        return (
            f"MasterAgentOrchestrator(\n"
            f"  Available Agents: {list(self.worker_agents.keys())}\n"
            f"  Queries Processed: {len(self.query_history)}\n"
            f"  LLM Model: {OLLAMA_CONFIG['model']}\n"
            f")"
        )


# Convenience function
def get_master_agent(verbose: bool = True) -> MasterAgentOrchestrator:
    """Get instance of Master Agent Orchestrator"""
    return MasterAgentOrchestrator(verbose=verbose)


# Test the Master Agent
if __name__ == "__main__":
    print("="*70)
    print("MASTER AGENT - TEST SUITE")
    print("="*70)
    
    # Initialize Master Agent
    master = get_master_agent(verbose=True)
    
    print("\n" + "="*70)
    print("TEST 1: Query Parsing")
    print("="*70)
    
    test_query = "What are the market trends for Metformin in India?"
    parsed = master.parse_query(test_query)
    print("\nParsed Query:")
    print(json.dumps(parsed, indent=2))
    
    print("\n" + "="*70)
    print("TEST 2: Agent Validation Test")
    print("="*70)
    
    # Test the query that was failing
    test_query_2 = "Which oncology drugs have completed Phase 3 trials but are not yet launched in emerging markets?"
    parsed_2 = master.parse_query(test_query_2)
    print("\nParsed Query:")
    print(json.dumps(parsed_2, indent=2))
    print(f"\nRequired Agents: {parsed_2['required_agents']}")
    print("✓ Should include both 'clinical_trials' AND 'iqvia'")
    
    print("\n" + "="*70)
    print("TEST 3: Full Query Processing")
    print("="*70)
    
    test_queries = [
        "What are the market trends for Metformin?",
        "Which oncology drugs have completed Phase 3 trials?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i} ---")
        result = master.process_query(query)
        
        print(f"\nQuery: {result['query']}")
        print(f"Intent: {result['parsed_query']['intent']}")
        print(f"Required Agents: {result['parsed_query']['required_agents']}")
        print(f"Processing Time: {result['processing_time']}s")
    
    print("\n" + "="*70)
    print("MASTER AGENT TEST COMPLETED")
    print("="*70)
    
    print("\n✓ All tests completed successfully!")
    print("\nKey Improvement: Agent validation now ensures correct agents are called")
    print("even if LLM makes mistakes in agent selection!")