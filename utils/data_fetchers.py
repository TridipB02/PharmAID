#data_fetchers.py
"""
Data Fetchers for Pharma Agentic AI System
Handles loading mock data and fetching from real APIs
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime

# Import settings
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DATA_FILES, API_ENDPOINTS, RATE_LIMIT


class DataFetcher:
    """Base class for data fetching operations"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PharmaAgenticAI/1.0'
        })
    
    def _rate_limit(self, delay: float = 0.5):
        """Simple rate limiting"""
        time.sleep(delay)


class MockDataFetcher(DataFetcher):
    """Fetches data from mock JSON files"""
    
    def __init__(self):
        super().__init__()
        self.drugs_db = self._load_drugs_database()
        self.iqvia_data = self._load_iqvia_data()
        self.exim_data = self._load_exim_data()
    
    def _load_json_file(self, file_path: Path) -> Dict:
        """Load JSON file safely"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Returning empty dict.")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: {file_path} is not valid JSON. Returning empty dict.")
            return {}
    
    def _load_drugs_database(self) -> Dict:
        """Load drugs database"""
        return self._load_json_file(DATA_FILES['drugs_database'])
    
    def _load_iqvia_data(self) -> Dict:
        """Load IQVIA mock data"""
        return self._load_json_file(DATA_FILES['mock_iqvia'])
    
    def _load_exim_data(self) -> Dict:
        """Load EXIM mock data"""
        return self._load_json_file(DATA_FILES['mock_exim'])
    
    def get_drug_info(self, drug_name: str) -> Optional[Dict]:
        """Get drug information by name"""
        if 'drugs' not in self.drugs_db:
            return None
        
        drug_name_lower = drug_name.lower()
        for drug in self.drugs_db['drugs']:
            if (drug['name'].lower() == drug_name_lower or 
                drug['generic_name'].lower() == drug_name_lower):
                return drug
        return None
    
    def get_drugs_by_therapeutic_area(self, therapeutic_area: str) -> List[Dict]:
        """Get all drugs in a therapeutic area"""
        if 'drugs' not in self.drugs_db:
            return []
        
        area_lower = therapeutic_area.lower()
        return [
            drug for drug in self.drugs_db['drugs']
            if drug['therapeutic_area'].lower() == area_lower
        ]
    
    def get_iqvia_market_data(self, drug_name: str) -> Optional[Dict]:
        """Get IQVIA market data for a drug"""
        if 'market_data' not in self.iqvia_data:
            return None
        
        drug_name_lower = drug_name.lower()
        for entry in self.iqvia_data['market_data']:
            if entry['drug_name'].lower() == drug_name_lower:
                return entry
        return None
    
    def get_exim_trade_data(self, drug_name: str) -> Optional[Dict]:
        """Get EXIM trade data for a drug"""
        if 'trade_data' not in self.exim_data:
            return None
        
        drug_name_lower = drug_name.lower()
        for entry in self.exim_data['trade_data']:
            if entry['drug_name'].lower() == drug_name_lower:
                return entry
        return None
    
    def calculate_cagr(self, drug_name: str) -> Optional[float]:
        """Calculate CAGR from IQVIA data"""
        market_data = self.get_iqvia_market_data(drug_name)
        if not market_data or 'data_points' not in market_data:
            return None
        
        data_points = sorted(market_data['data_points'], key=lambda x: x['year'])
        if len(data_points) < 2:
            return None
        
        start_value = data_points[0]['sales_usd_million']
        end_value = data_points[-1]['sales_usd_million']
        years = data_points[-1]['year'] - data_points[0]['year']
        
        if start_value <= 0 or years <= 0:
            return None
        
        cagr = (pow(end_value / start_value, 1 / years) - 1) * 100
        return round(cagr, 2)


class ClinicalTrialsFetcher(DataFetcher):
    """Fetches data from ClinicalTrials.gov API"""
    
    def __init__(self):
        super().__init__()
        self.base_url = API_ENDPOINTS['clinical_trials']
    
    def search_trials(self, condition: str = None, intervention: str = None, 
                     max_results: int = 10) -> List[Dict]:
        """
        Search clinical trials
        
        Args:
            condition: Disease or condition (e.g., "Diabetes")
            intervention: Drug/intervention name (e.g., "Metformin")
            max_results: Maximum number of results to return
        
        Returns:
            List of trial dictionaries
        """
        try:
            params = {
                'format': 'json',
                'pageSize': max_results
            }
            
            query_parts = []
            if condition:
                query_parts.append(f'AREA[ConditionSearch]{condition}')
            if intervention:
                query_parts.append(f'AREA[InterventionSearch]{intervention}')
            
            if query_parts:
                params['query.cond'] = condition if condition else ''
                params['query.intr'] = intervention if intervention else ''
            
            self._rate_limit(0.1)  # Rate limiting
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse the response
            trials = []
            if 'studies' in data:
                for study in data['studies'][:max_results]:
                    trial_info = self._parse_trial(study)
                    trials.append(trial_info)
            
            return trials
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching clinical trials: {e}")
            return []
    
    def _parse_trial(self, study: Dict) -> Dict:
        """Parse trial data from API response"""
        protocol_section = study.get('protocolSection', {})
        identification = protocol_section.get('identificationModule', {})
        status = protocol_section.get('statusModule', {})
        design = protocol_section.get('designModule', {})
        
        return {
            'nct_id': identification.get('nctId', 'N/A'),
            'title': identification.get('officialTitle', identification.get('briefTitle', 'N/A')),
            'status': status.get('overallStatus', 'Unknown'),
            'phase': design.get('phases', ['N/A'])[0] if design.get('phases') else 'N/A',
            'start_date': status.get('startDateStruct', {}).get('date', 'N/A'),
            'completion_date': status.get('completionDateStruct', {}).get('date', 'N/A'),
            'sponsor': protocol_section.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name', 'N/A'),
            'conditions': protocol_section.get('conditionsModule', {}).get('conditions', []),
            'interventions': [i.get('name', '') for i in protocol_section.get('armsInterventionsModule', {}).get('interventions', [])]
        }


class PubMedFetcher(DataFetcher):
    """Fetches data from PubMed API"""
    
    def __init__(self):
        super().__init__()
        self.search_url = API_ENDPOINTS['pubmed_search']
        self.fetch_url = API_ENDPOINTS['pubmed_fetch']
    
    def search_publications(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search PubMed for publications
        
        Args:
            query: Search query (e.g., "Metformin AND cancer")
            max_results: Maximum number of results
        
        Returns:
            List of publication dictionaries
        """
        try:
            # Step 1: Search for PMIDs
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            self._rate_limit(0.34)  # 3 requests per second max
            search_response = self.session.get(self.search_url, params=search_params, timeout=10)
            search_response.raise_for_status()
            
            search_data = search_response.json()
            pmids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                return []
            
            # Step 2: Fetch details for PMIDs
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }
            
            self._rate_limit(0.34)
            fetch_response = self.session.get(self.fetch_url, params=fetch_params, timeout=10)
            fetch_response.raise_for_status()
            
            # Parse XML (simplified - you might want to use xml.etree)
            # For now, return basic info
            publications = []
            for pmid in pmids:
                publications.append({
                    'pmid': pmid,
                    'title': f'Publication {pmid}',
                    'abstract': 'Abstract not parsed (XML parsing needed)',
                    'authors': [],
                    'journal': 'Journal name',
                    'year': 'Year',
                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
                })
            
            return publications
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching PubMed data: {e}")
            return []


class PatentFetcher(DataFetcher):
    """Fetches patent data (simplified version)"""
    
    def __init__(self):
        super().__init__()
        # Note: USPTO API requires registration, using simplified approach
    
    def search_patents(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for patents (mock implementation for now)
        
        Args:
            query: Patent search query (molecule name, indication)
            max_results: Maximum results
        
        Returns:
            List of patent dictionaries
        """
        # This is a simplified mock implementation
        # In production, you'd use USPTO API or Google Patents
        
        print(f"Searching patents for: {query}")
        
        # Return mock patent data
        mock_patents = [
            {
                'patent_number': 'US1234567B2',
                'title': f'Pharmaceutical composition of {query}',
                'filing_date': '2015-01-15',
                'grant_date': '2018-03-20',
                'expiry_date': '2035-01-15',
                'assignee': 'Pharma Company Inc.',
                'status': 'Active',
                'claims_count': 20
            },
            {
                'patent_number': 'US2345678B2',
                'title': f'Method of using {query} for treatment',
                'filing_date': '2017-06-10',
                'grant_date': '2020-08-15',
                'expiry_date': '2037-06-10',
                'assignee': 'Research Institute',
                'status': 'Active',
                'claims_count': 15
            }
        ]
        
        return mock_patents[:max_results]


class OpenFDAFetcher(DataFetcher):
    """Fetches data from OpenFDA API"""
    
    def __init__(self):
        super().__init__()
        self.labels_url = API_ENDPOINTS['openfda_labels']
    
    def get_drug_label(self, drug_name: str) -> Optional[Dict]:
        """
        Get FDA drug label information
        
        Args:
            drug_name: Name of the drug
        
        Returns:
            Drug label information
        """
        try:
            params = {
                'search': f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
                'limit': 1
            }
            
            self._rate_limit(0.25)  # 4 per second max without key
            response = self.session.get(self.labels_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                return {
                    'brand_name': result.get('openfda', {}).get('brand_name', ['N/A'])[0],
                    'generic_name': result.get('openfda', {}).get('generic_name', ['N/A'])[0],
                    'manufacturer': result.get('openfda', {}).get('manufacturer_name', ['N/A'])[0],
                    'indications': result.get('indications_and_usage', ['N/A'])[0] if result.get('indications_and_usage') else 'N/A',
                    'warnings': result.get('warnings', ['N/A'])[0] if result.get('warnings') else 'N/A',
                    'route': result.get('openfda', {}).get('route', []),
                }
            
            return None
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching OpenFDA data: {e}")
            return None


# Convenience functions
def get_mock_data_fetcher() -> MockDataFetcher:
    """Get instance of MockDataFetcher"""
    return MockDataFetcher()

def get_clinical_trials_fetcher() -> ClinicalTrialsFetcher:
    """Get instance of ClinicalTrialsFetcher"""
    return ClinicalTrialsFetcher()

def get_pubmed_fetcher() -> PubMedFetcher:
    """Get instance of PubMedFetcher"""
    return PubMedFetcher()

def get_patent_fetcher() -> PatentFetcher:
    """Get instance of PatentFetcher"""
    return PatentFetcher()

def get_openfda_fetcher() -> OpenFDAFetcher:
    """Get instance of OpenFDAFetcher"""
    return OpenFDAFetcher()


# Test functions
if __name__ == "__main__":
    print("Testing Data Fetchers...")
    
    # Test Mock Data
    print("\n1. Testing Mock Data Fetcher:")
    mock_fetcher = get_mock_data_fetcher()
    drug_info = mock_fetcher.get_drug_info("Metformin")
    print(f"Drug Info: {drug_info}")
    
    market_data = mock_fetcher.get_iqvia_market_data("Metformin")
    print(f"Market Data: {market_data}")
    
    cagr = mock_fetcher.calculate_cagr("Metformin")
    print(f"CAGR: {cagr}%")
    
    # Test Clinical Trials
    print("\n2. Testing Clinical Trials API:")
    ct_fetcher = get_clinical_trials_fetcher()
    trials = ct_fetcher.search_trials(condition="Diabetes", intervention="Metformin", max_results=3)
    print(f"Found {len(trials)} trials")
    if trials:
        print(f"First trial: {trials[0]['title']}")
    
    # Test PubMed
    print("\n3. Testing PubMed API:")
    pubmed_fetcher = get_pubmed_fetcher()
    publications = pubmed_fetcher.search_publications("Metformin cancer", max_results=3)
    print(f"Found {len(publications)} publications")
    
    print("\nAll tests completed!")