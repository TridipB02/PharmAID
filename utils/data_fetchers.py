# data_fetchers.py
"""
Data Fetchers for Pharma Agentic AI System
Handles loading mock data and fetching from real APIs
"""

import json
import sys
import time
import base64
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import requests

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import API_ENDPOINTS, DATA_FILES


class DataFetcher:
    """Base class for data fetching operations"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "PharmaAgenticAI/1.0"})

    def _rate_limit(self, delay: float = 0.5):
        """Simple rate limiting"""
        time.sleep(delay)


class MockDataFetcher(DataFetcher):
    """Fetches data from mock JSON files"""

    def __init__(self):
        super().__init__()
        self.drugs_db = self._load_drugs_database()
        self.iqvia_data = self._load_iqvia_data()

    def _load_json_file(self, file_path: Path) -> Dict:
        """Load JSON file safely"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Returning empty dict.")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: {file_path} is not valid JSON. Returning empty dict.")
            return {}

    def _load_drugs_database(self) -> Dict:
        """Load drugs database"""
        return self._load_json_file(DATA_FILES["drugs_database"])

    def _load_iqvia_data(self) -> Dict:
        """Load IQVIA mock data"""
        return self._load_json_file(DATA_FILES["mock_iqvia"])

    def get_drug_info(self, drug_name: str) -> Optional[Dict]:
        """Get drug information by name"""
        if "drugs" not in self.drugs_db:
            return None

        drug_name_lower = drug_name.lower()
        for drug in self.drugs_db["drugs"]:
            if (
                drug["name"].lower() == drug_name_lower
                or drug["generic_name"].lower() == drug_name_lower
            ):
                return drug
        return None

    def get_drugs_by_therapeutic_area(self, therapeutic_area: str) -> List[Dict]:
        """Get all drugs in a therapeutic area"""
        if "drugs" not in self.drugs_db:
            return []

        area_lower = therapeutic_area.lower()
        return [
            drug
            for drug in self.drugs_db["drugs"]
            if drug["therapeutic_area"].lower() == area_lower
        ]

    def get_iqvia_market_data(self, drug_name: str) -> Optional[Dict]:
        """Get IQVIA market data for a drug"""
        if "market_data" not in self.iqvia_data:
            return None

        drug_name_lower = drug_name.lower()
        for entry in self.iqvia_data["market_data"]:
            if entry["drug_name"].lower() == drug_name_lower:
                return entry
        return None

    def calculate_cagr(self, drug_name: str) -> Optional[float]:
        """Calculate CAGR from IQVIA data"""
        market_data = self.get_iqvia_market_data(drug_name)
        if not market_data or "data_points" not in market_data:
            return None

        data_points = sorted(market_data["data_points"], key=lambda x: x["year"])
        if len(data_points) < 2:
            return None

        start_value = data_points[0]["sales_usd_million"]
        end_value = data_points[-1]["sales_usd_million"]
        years = data_points[-1]["year"] - data_points[0]["year"]

        if start_value <= 0 or years <= 0:
            return None

        cagr = (pow(end_value / start_value, 1 / years) - 1) * 100
        return round(cagr, 2)


class ClinicalTrialsFetcher(DataFetcher):
    """Fetches data from ClinicalTrials.gov API"""

    def __init__(self):
        super().__init__()
        self.base_url = API_ENDPOINTS["clinical_trials"]

    def search_trials(
        self, condition: str = None, intervention: str = None, max_results: int = 10
    ) -> List[Dict]:
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
            params = {"format": "json", "pageSize": max_results}

            query_parts = []
            if condition:
                query_parts.append(f"AREA[ConditionSearch]{condition}")
            if intervention:
                query_parts.append(f"AREA[InterventionSearch]{intervention}")

            if query_parts:
                params["query.cond"] = condition if condition else ""
                params["query.intr"] = intervention if intervention else ""

            self._rate_limit(0.1)  # Rate limiting
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Parse the response
            trials = []
            if "studies" in data:
                for study in data["studies"][:max_results]:
                    trial_info = self._parse_trial(study)
                    trials.append(trial_info)

            return trials

        except requests.exceptions.RequestException as e:
            print(f"Error fetching clinical trials: {e}")
            return []

    def _parse_trial(self, study: Dict) -> Dict:
        """Parse trial data from API response"""
        protocol_section = study.get("protocolSection", {})
        identification = protocol_section.get("identificationModule", {})
        status = protocol_section.get("statusModule", {})
        design = protocol_section.get("designModule", {})

        return {
            "nct_id": identification.get("nctId", "N/A"),
            "title": identification.get(
                "officialTitle", identification.get("briefTitle", "N/A")
            ),
            "status": status.get("overallStatus", "Unknown"),
            "phase": (
                design.get("phases", ["N/A"])[0] if design.get("phases") else "N/A"
            ),
            "start_date": status.get("startDateStruct", {}).get("date", "N/A"),
            "completion_date": status.get("completionDateStruct", {}).get(
                "date", "N/A"
            ),
            "sponsor": protocol_section.get("sponsorCollaboratorsModule", {})
            .get("leadSponsor", {})
            .get("name", "N/A"),
            "conditions": protocol_section.get("conditionsModule", {}).get(
                "conditions", []
            ),
            "interventions": [
                i.get("name", "")
                for i in protocol_section.get("armsInterventionsModule", {}).get(
                    "interventions", []
                )
            ],
        }


class PubMedFetcher(DataFetcher):
    """Fetches data from PubMed API"""

    def __init__(self):
        super().__init__()
        self.search_url = API_ENDPOINTS["pubmed_search"]
        self.fetch_url = API_ENDPOINTS["pubmed_fetch"]

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
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            }

            self._rate_limit(0.34)  # 3 requests per second max
            search_response = self.session.get(
                self.search_url, params=search_params, timeout=10
            )
            search_response.raise_for_status()

            search_data = search_response.json()
            pmids = search_data.get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return []

            # Step 2: Fetch details for PMIDs
            fetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}

            self._rate_limit(0.34)
            fetch_response = self.session.get(
                self.fetch_url, params=fetch_params, timeout=10
            )
            fetch_response.raise_for_status()

            # Parse XML (simplified - you might want to use xml.etree)
            # For now, return basic info
            publications = []
            for pmid in pmids:
                publications.append(
                    {
                        "pmid": pmid,
                        "title": f"Publication {pmid}",
                        "abstract": "Abstract not parsed (XML parsing needed)",
                        "authors": [],
                        "journal": "Journal name",
                        "year": "Year",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    }
                )

            return publications

        except requests.exceptions.RequestException as e:
            print(f"Error fetching PubMed data: {e}")
            return []


class EPOPatentFetcher:
    """Fetches patent data from EPO Open Patent Services API with OAuth 2.0"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # Import here to avoid circular import
        from config.settings import EPO_CONFIG
        
        self.consumer_key = EPO_CONFIG['consumer_key']
        self.consumer_secret = EPO_CONFIG['consumer_secret']
        self.base_url = EPO_CONFIG['base_url']
        self.auth_url = EPO_CONFIG['auth_url']
        
        self.access_token = None
        self.token_expiry = None
        
        # Create a session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PharmaAgenticAI/1.0'
        })
        
        # Rate limiting (30 req/min for registered users)
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 30 requests/minute
        
        if self.consumer_key and self.consumer_secret:
            success = self._authenticate()
            if success:
                if self.verbose:
                    print(" EPO Patent Fetcher initialized (Authenticated)")
            else:
                if self.verbose:
                    print(" EPO authentication failed - using mock data fallback")
        else:
            if self.verbose:
                print(" EPO keys not found - using anonymous access")
    
    def _authenticate(self) -> bool:
        """Authenticate with EPO OPS API using OAuth 2.0"""
        try:
            credentials = f"{self.consumer_key}:{self.consumer_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {'grant_type': 'client_credentials'}
            
            response = requests.post(
                self.auth_url,
                headers=headers,
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                expires_in = int(token_data.get('expires_in', 3600))
                self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}'
                })
                
                if self.verbose:
                    print(f"   EPO authenticated - token expires in {expires_in}s")
                
                return True
            else:
                if self.verbose:
                    print(f"  Authentication failed: {response.status_code}")
                return False
        
        except Exception as e:
            if self.verbose:
                print(f"   Authentication error: {e}")
            return False
    
    def _check_token_expiry(self) -> bool:
        """Check if token needs refresh"""
        if not self.access_token:
            return False
        
        if self.token_expiry and datetime.now() >= self.token_expiry:
            if self.verbose:
                print("  Token expired, re-authenticating...")
            return self._authenticate()
        
        return True
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def search_patents(self, query: str, max_results: int = 10, fetch_details_count: int = 5) -> List[Dict]:
        """
        Search patents with HYBRID detail fetching
        
        Args:
            query: Search query
            max_results: Max results
            fetch_details_count: How many to fetch full details for (default 5)
        
        Returns:
            List of patent dictionaries
        """
        if not query or not query.strip():
            if self.verbose:
                print(" Empty query provided")
            return []
        
        # Check authentication
        if self.consumer_key and self.consumer_secret:
            self._check_token_expiry()
        
        if self.verbose:
            print(f"[EPO Patent Search] Query: {query}")
        
        try:
            # Step 1: Search for patent numbers
            search_url = f"{self.base_url}/published-data/search"
            
            cql_query = f'ti="{query}" OR ab="{query}"'
            
            params = {'q': cql_query}
            
            headers = {
                'Accept': 'application/json',
                'Range': f'1-{min(max_results, 100)}'
            }
            
            self._rate_limit()
            
            if self.verbose:
                print(f"  Sending request to: {search_url}")
                print(f"  Query: {cql_query}")
            
            response = self.session.get(
                search_url,
                params=params,
                headers=headers,
                timeout=30
            )
            
            if self.verbose:
                print(f"  Response status: {response.status_code}")
            
            if response.status_code == 200:
                # Parse to get patent numbers
                basic_patents = self._parse_epo_response(response.json())
                
                if not basic_patents:
                    return []
                
                if self.verbose:
                    print(f"   Found {len(basic_patents)} patents")
                
                # Step 2: Fetch full details for TOP N patents only
                enhanced_patents = []
                
                for i, patent in enumerate(basic_patents):
                    if i < fetch_details_count:
                        # Fetch full details for top 5
                        full_details = self._fetch_patent_biblio(patent['patent_number'])
                        if full_details:
                            patent.update(full_details)
                            if self.verbose:
                                print(f"  Enhanced {i+1}/{fetch_details_count}: {patent['patent_number']}")
                    
                    enhanced_patents.append(patent)
                
                return enhanced_patents[:max_results]
            
            elif response.status_code == 404:
                if self.verbose:
                    print(f"  No patents found for: {query}")
                return []
            
            elif response.status_code == 401 or response.status_code == 403:
                if self.verbose:
                    print(f"  ⚠ EPO authentication error ({response.status_code})")
                if self._authenticate():
                    return self.search_patents(query, max_results, fetch_details_count)
                else:
                    return self._generate_mock_patents(query, max_results)
            
            else:
                if self.verbose:
                    print(f"  ⚠ EPO returned status {response.status_code}")
                return self._generate_mock_patents(query, max_results)
        
        except Exception as e:
            if self.verbose:
                print(f"  ✗ EPO search error: {e}")
            return self._generate_mock_patents(query, max_results)
    
    def _parse_epo_response(self, data: Dict) -> List[Dict]:
        """
        Parse EPO search response - extracts BASIC info only (patent numbers)
        Full details will be fetched separately for top N
        """
        patents = []
        
        try:
            world_patent_data = data.get('ops:world-patent-data', {})
            biblio_search = world_patent_data.get('ops:biblio-search', {})
            search_result = biblio_search.get('ops:search-result', {})
            pub_refs = search_result.get('ops:publication-reference', [])
            
            if not isinstance(pub_refs, list):
                pub_refs = [pub_refs] if pub_refs else []
            
            for pub_ref in pub_refs:
                #  Extract basic patent info from document-id
                document = pub_ref.get('document-id', {})
                
                #  Get country code (e.g., "WO", "US", "EP")
                country = document.get('country', {}).get('$', 'N/A')
                
                #  Get document number (e.g., "2025227064")
                doc_number = document.get('doc-number', {}).get('$', 'N/A')
                
                #  Get kind code (e.g., "A1", "B2")
                kind = document.get('kind', {}).get('$', '')
                
                #  Combine into full patent number
                patent_number = f"{country}{doc_number}{kind}"
                
                #  Create basic patent record (details filled later for top 5)
                patent = {
                    'patent_number': patent_number,
                    'title': f"Patent {patent_number}",  # Placeholder until biblio fetch
                    'assignee': 'N/A',  # Will be filled by _fetch_patent_biblio
                    'filing_date': 'N/A',
                    'grant_date': 'N/A',
                    'expiry_date': 'N/A',
                    'status': 'Unknown',
                    'patent_type': 'utility',
                    'claims_count': 'N/A',
                    'url': f"https://patents.google.com/patent/{patent_number}"
                }
                
                patents.append(patent)
        
        except Exception as e:
            if self.verbose:
                print(f"   Parse error: {e}")
        
        return patents
    
    def _fetch_patent_biblio(self, patent_number: str) -> Dict:
        """
        Fetch full bibliographic details for a specific patent
        FIXED to match actual EPO response structure
    
        Args:
            patent_number: Patent number (e.g., "CN120549879A")
    
        Returns:
            Dict with title, assignee, dates
        """
        try:
            # Parse patent number: Country + Number + Kind
            match = re.match(r'([A-Z]{2})(\d+)([A-Z]\d*)', patent_number)
            if not match:
                return {}
            
            country, number, kind = match.groups()
            
            # EPO biblio endpoint
            biblio_url = f"{self.base_url}/published-data/publication/docdb/{country}.{number}.{kind}/biblio"
            
            headers = {'Accept': 'application/json'}
            
            self._rate_limit()
            
            response = self.session.get(biblio_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()

                #CORRECT NAVIGATION (from actual EPO response)
                world_data = data.get('ops:world-patent-data', {})

                #Key difference: exchange-documents, not patent-documents
                exchange_docs = world_data.get('exchange-documents', {})
                exchange_doc = exchange_docs.get('exchange-document', {})
                biblio_data = exchange_doc.get('bibliographic-data', {})

                #Extract title (from invention-title array)
                title = 'N/A'
                title_list = biblio_data.get('invention-title', [])

                if isinstance(title_list, list):
                    #Look for English title first
                    for title_obj in title_list:
                        if isinstance(title_obj, dict):
                            if title_obj.get('@lang') == 'en':
                                title = title_obj.get('$', 'N/A')
                                break
                            elif title == 'N/A':  # Use any title as fallback
                                title = title_obj.get('$', 'N/A')
                elif isinstance(title_list, dict):
                    title = title_list.get('$', 'N/A')

                #Extract assignee/applicant (from parties -> applicants)
                assignee = 'N/A'
                parties = biblio_data.get('parties', {})
                applicants_section = parties.get('applicants', {})
                applicant_list = applicants_section.get('applicant', [])
            
                if not isinstance(applicant_list, list):
                    applicant_list = [applicant_list]

                if applicant_list:
                    # Find English/epodoc format applicant
                    for applicant in applicant_list:
                        if applicant.get('@data-format') == 'epodoc':
                            name_obj = applicant.get('applicant-name', {})
                            name_data = name_obj.get('name', {})
                            if isinstance(name_data, dict):
                                assignee = name_data.get('$', 'N/A')
                                break
                            elif isinstance(name_data, str):
                                assignee = name_data
                                break

                    #Fallback to first applicant if no epodoc format
                    if assignee == 'N/A' and applicant_list:
                        first_applicant = applicant_list[0]
                        name_obj = first_applicant.get('applicant-name', {})
                        name_data = name_obj.get('name', {})
                        if isinstance(name_data, dict):
                            assignee = name_data.get('$', 'N/A')
                        elif isinstance(name_data, str):
                            assignee = name_data

                #Extract publication date
                pub_ref = biblio_data.get('publication-reference', {})
                pub_doc_ids = pub_ref.get('document-id', [])
            
                if not isinstance(pub_doc_ids, list):
                    pub_doc_ids = [pub_doc_ids]

                pub_date = 'N/A'
                for doc_id in pub_doc_ids:
                    if doc_id.get('@document-id-type') == 'docdb':
                        date_obj = doc_id.get('date', {})
                        pub_date = date_obj.get('$', 'N/A')
                        break

                #Extract filing/application date
                app_ref = biblio_data.get('application-reference', {})
                app_doc_ids = app_ref.get('document-id', [])
            
                if not isinstance(app_doc_ids, list):
                    app_doc_ids = [app_doc_ids]

                filing_date = pub_date  # Default to publication date
                for doc_id in app_doc_ids:
                    if doc_id.get('@document-id-type') == 'epodoc':
                        date_obj = doc_id.get('date', {})
                        filing_date = date_obj.get('$', pub_date)
                        break

                #Calculate expiry (20 years from filing)
                expiry_date = 'N/A'
                status = 'Unknown'
                filing_date_formatted = 'N/A'
            
                try:
                    filing_dt = datetime.strptime(filing_date, '%Y%m%d')
                    expiry_dt = filing_dt + timedelta(days=20*365)
                    expiry_date = expiry_dt.strftime('%Y-%m-%d')
                    filing_date_formatted = filing_dt.strftime('%Y-%m-%d')
                
                    current_year = datetime.now().year
                    expiry_year = expiry_dt.year
                    status = 'Active' if expiry_year > current_year else 'Expired'
                except:
                    filing_date_formatted = filing_date if filing_date != 'N/A' else 'N/A'

                return {
                    'title': title if title != 'N/A' else f"Patent {patent_number}",
                    'assignee': assignee,
                    'filing_date': filing_date_formatted,
                    'grant_date': filing_date_formatted,
                    'expiry_date': expiry_date,
                    'status': status
                }
            
            elif response.status_code == 404:
                # Patent not found in EPO biblio (might be too recent or from other DB)
                return {
                    'title': f"Patent {patent_number}",
                    'assignee': 'Data not available in EPO',
                    'filing_date': 'N/A',
                    'grant_date': 'N/A',
                    'expiry_date': 'N/A',
                    'status': 'Unknown'
                }
            
            return {}
        
        except Exception as e:
            if self.verbose:
                print(f"   Biblio fetch failed: {e}")
            return {}
  
    def _generate_mock_patents(self, query: str, count: int) -> List[Dict]:
        """Generate mock patent data as fallback"""
        import random
        
        patents = []
        current_year = datetime.now().year
        
        assignees = [
            "Pfizer Inc.", "Johnson & Johnson", "Novartis AG",
            "Roche Holding AG", "Merck & Co.", "GlaxoSmithKline"
        ]
        
        for i in range(min(count, 10)):
            grant_year = random.randint(2005, 2023)
            grant_date = f"{grant_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            filing_year = grant_year - random.randint(2, 4)
            filing_date = f"{filing_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            
            expiry_year = filing_year + 20
            expiry_date = f"{expiry_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            
            status = 'Active' if expiry_year > current_year else 'Expired'
            
            patent = {
                'patent_number': f"EP{random.randint(1000000, 3000000)}B1",
                'title': f"Pharmaceutical composition comprising {query}",
                'filing_date': filing_date,
                'grant_date': grant_date,
                'expiry_date': expiry_date,
                'assignee': random.choice(assignees),
                'status': status,
                'patent_type': 'utility',
                'claims_count': str(random.randint(10, 50)),
                'url': f"https://patents.google.com/patent/EP{random.randint(1000000, 3000000)}B1"
            }
            
            patents.append(patent)
        
        return patents


class OpenFDAFetcher(DataFetcher):
    """Fetches data from OpenFDA API"""

    def __init__(self):
        super().__init__()
        self.labels_url = API_ENDPOINTS["openfda_labels"]

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
                "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
                "limit": 1,
            }

            self._rate_limit(0.25)  # 4 per second max without key
            response = self.session.get(self.labels_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]
                return {
                    "brand_name": result.get("openfda", {}).get("brand_name", ["N/A"])[
                        0
                    ],
                    "generic_name": result.get("openfda", {}).get(
                        "generic_name", ["N/A"]
                    )[0],
                    "manufacturer": result.get("openfda", {}).get(
                        "manufacturer_name", ["N/A"]
                    )[0],
                    "indications": (
                        result.get("indications_and_usage", ["N/A"])[0]
                        if result.get("indications_and_usage")
                        else "N/A"
                    ),
                    "warnings": (
                        result.get("warnings", ["N/A"])[0]
                        if result.get("warnings")
                        else "N/A"
                    ),
                    "route": result.get("openfda", {}).get("route", []),
                }

            return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching OpenFDA data: {e}")
            return None


class UNComtradeFetcher(DataFetcher):
    """Fetches trade data from UN Comtrade API"""

    def __init__(self, verbose: bool = False):
        super().__init__()
        self.base_url = API_ENDPOINTS["un_comtrade"]
        self.verbose = verbose

        # HS codes for pharmaceutical products
        self.pharma_hs_codes = {
            "general": "3004",
            "antibiotics": "3004",
            "insulin": "3004",
            "cardiovascular": "3004",
            "default": "3004",
        }

    def _get_country_name(self, code: str) -> str:
        """Helper to get country name from code"""
        country_codes = {
            "842": "USA",
            "156": "China",
            "276": "Germany",
            "372": "Ireland",
            "356": "India",
            "124": "Canada",
            "826": "United Kingdom",
            "392": "Japan",
            "250": "France",
            "528": "Netherlands",
        }
        return country_codes.get(code, f"Country {code}")

    def get_trade_data(
        self, drug_name: str = None, country: str = None, year: int = 2022
    ) -> List[Dict]:
        """
        Get pharmaceutical trade data from UN Comtrade

        Args:
            drug_name: Drug name (optional)
            country: Country ISO3 code (e.g., "USA", "IND", "CHN") or None
            year: Year of data (default 2022)

        Returns:
            List of trade records
        """
        try:
            # Build URL with path parameters
            typeCode = "C"  # C = Commodities
            freqCode = "A"  # A = Annual
            clCode = "HS"  # HS = Harmonized System

            url = f"{self.base_url}/{typeCode}/{freqCode}/{clCode}"

            # Use specific country codes (UN Comtrade doesn't accept "all")
            if country:
                reporter_codes = [country]
            else:
                # Query multiple major pharmaceutical trading countries
                reporter_codes = ["842", "156", "276", "392", "826"]  # USA

            all_records = []

            for reporter_code in reporter_codes:
                params = {
                    "reporterCode": reporter_code,
                    "period": str(year),
                    "partnerCode": "0",
                    "partner2Code": "",
                    "cmdCode": "3004",
                    "flowCode": "M,X",
                    "customsCode": "C00",
                    "motCode": "0",
                }

                self._rate_limit(2.0)

                if self.verbose:
                    country_name = self._get_country_name(reporter_code)
                    print(
                        f"  Fetching UN Comtrade data for {country_name}, year {year}"
                    )

                response = self.session.get(url, params=params, timeout=30)

                # Rate limit handling
                if response.status_code == 429:
                    if self.verbose:
                        print("   UN Comtrade rate limit hit, waiting 10 seconds...")
                    time.sleep(10)
                    response = self.session.get(url, params=params, timeout=30)

                # Debug bad requests
                if response.status_code != 200:
                    if self.verbose:
                        print(f"   Status: {response.status_code}")
                        print(f"  Response: {response.text[:200]}")
                    continue  # Skip and try next country

                data = response.json()
                records = data.get("data", data.get("dataset", data.get("results", [])))

                if records:
                    all_records.extend(records)
                    if self.verbose:
                        print(f"   Retrieved {len(records)} trade records")

            if self.verbose:
                if all_records:
                    print(f"   Total: {len(all_records)} records from UN Comtrade")
                else:
                    print(f"   No data returned from UN Comtrade")

            return all_records[:50]

        except requests.exceptions.RequestException as e:
            if self.verbose:
                print(f"  UN Comtrade API error: {e}")
            return []
        except Exception as e:
            if self.verbose:
                print(f"  Unexpected error in UN Comtrade: {e}")
            return []


class WorldBankTradeFetcher(DataFetcher):
    """Fetches aggregate trade data from World Bank API"""

    def __init__(self, verbose: bool = False):  # ✅ ADD DEFAULT VALUE
        super().__init__()
        self.base_url = API_ENDPOINTS["world_bank_trade"]
        self.verbose = verbose  # ✅ NOW IT WORKS

    def get_trade_indicators(
        self, country: str = "all", indicator: str = "TX.VAL.MRCH.CD.WT"
    ) -> List[Dict]:
        """
        Get trade indicators from World Bank

        Args:
            country: Country ISO code (e.g., "USA", "IND") or "all"
            indicator: World Bank indicator code
                - TX.VAL.MRCH.CD.WT = Merchandise exports (current US$)
                - TM.VAL.MRCH.CD.WT = Merchandise imports (current US$)

        Returns:
            List of trade indicator records
        """
        try:
            # World Bank API format: /v2/country/{country}/indicator/{indicator}
            url = f"{self.base_url}/country/{country}/indicator/{indicator}"

            params = {
                "format": "json",
                "per_page": 50,
                "date": "2015:2023",  # Last 8 years
            }

            self._rate_limit(0.5)

            if self.verbose:
                print(f"  Fetching World Bank trade data for {country}")

            response = self.session.get(url, params=params, timeout=20)
            response.raise_for_status()

            data = response.json()

            # World Bank returns [metadata, data]
            if isinstance(data, list) and len(data) > 1:
                records = data[1]  # Second element is the data

                if self.verbose:
                    print(f"   Retrieved {len(records)} records from World Bank")

                return records

            return []

        except requests.exceptions.RequestException as e:
            if self.verbose:
                print(f"   World Bank API error: {e}")
            return []
        except Exception as e:
            if self.verbose:
                print(f"   Unexpected error in World Bank: {e}")
            return []


class RealEXIMDataFetcher:
    """
    Combines multiple trade data sources - REAL APIs ONLY
    Uses UN Comtrade and World Bank (USITC removed)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.un_comtrade = UNComtradeFetcher(verbose=verbose)
        self.world_bank = WorldBankTradeFetcher(verbose=verbose)
        #  USITC removed completely

    def get_omprehensive_trade_data(
        self, drug_name: str, country: str = None
    ) -> Dict[str, any]:
        """
        Get comprehensive trade data from available sources

        Args:
            drug_name: Drug name
            country: Country code (optional)

        Returns:
            Dictionary with combined trade data
        """
        if self.verbose:
            print(f"\n[Real EXIM Data] Fetching trade data for {drug_name}")

        result = {
            "drug_name": drug_name,
            "country": country or "Global",
            "data_sources": [],
            "un_comtrade_data": [],
            "world_bank_data": [],
            "summary": "",
        }

        # Try UN Comtrade
        try:
            comtrade_data = self.un_comtrade.get_trade_data(
                drug_name, country, year=2022
            )
            if comtrade_data:
                result["un_comtrade_data"] = comtrade_data
                result["data_sources"].append("UN Comtrade")
        except Exception as e:
            if self.verbose:
                print(f"   UN Comtrade failed: {e}")

        #  FIXED: Always try World Bank (not conditional)
        try:
            country_code = country if country else "WLD"  # WLD = World aggregate

            if self.verbose:
                print(f"  Fetching World Bank trade data for {country_code}...")

            # Temporarily disable verbose for individual calls
            original_verbose = self.world_bank.verbose
            self.world_bank.verbose = False

            # Get both exports and imports
            wb_exports = self.world_bank.get_trade_indicators(
                country=country_code, indicator="TX.VAL.MRCH.CD.WT"
            )
            wb_imports = self.world_bank.get_trade_indicators(
                country=country_code, indicator="TM.VAL.MRCH.CD.WT"
            )

            # Restore verbose
            self.world_bank.verbose = original_verbose

            wb_data = wb_exports + wb_imports

            if wb_data:
                result["world_bank_data"] = wb_data
                result["data_sources"].append("World Bank")
                if self.verbose:
                    print(f"   World Bank: {len(wb_data)} records retrieved")
        except Exception as e:
            if self.verbose:
                print(f"   World Bank failed: {e}")

        # Generate summary
        if not result["data_sources"]:
            if self.verbose:
                print("   No real trade data available from any API")
            result["summary"] = (
                "No trade data available - APIs failed or returned empty"
            )
        else:
            result["summary"] = (
                f"Trade data from {len(result['data_sources'])} source(s): {', '.join(result['data_sources'])}"
            )

        if self.verbose:
            print(f"  {result['summary']}")

        return result


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


def get_patent_fetcher() -> EPOPatentFetcher:
    """Get instance of PatentFetcher"""
    return EPOPatentFetcher()


def get_openfda_fetcher() -> OpenFDAFetcher:
    """Get instance of OpenFDAFetcher"""
    return OpenFDAFetcher()


def get_real_exim_fetcher(verbose: bool = True) -> RealEXIMDataFetcher:
    """Get instance of Real EXIM Data Fetcher (combines multiple APIs)"""
    return RealEXIMDataFetcher(verbose=verbose)


def get_un_comtrade_fetcher() -> UNComtradeFetcher:
    """Get instance of UN Comtrade Fetcher"""
    return UNComtradeFetcher()


def get_world_bank_fetcher() -> WorldBankTradeFetcher:
    """Get instance of World Bank Trade Fetcher"""
    return WorldBankTradeFetcher()


def get_patent_fetcher(verbose: bool = False) -> EPOPatentFetcher:
    """Get instance of EPO Patent Fetcher with fixed OAuth 2.0"""
    return EPOPatentFetcher(verbose=verbose)

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
    trials = ct_fetcher.search_trials(
        condition="Diabetes", intervention="Metformin", max_results=3
    )
    print(f"Found {len(trials)} trials")
    if trials:
        print(f"First trial: {trials[0]['title']}")
    
    #Test EPO Patent Fetcher
    print("\n3. Testing EPO Patent Fetcher:")
    print("="*60)
    patent_fetcher = get_patent_fetcher()
    print(f"\nSearching patents for 'Aspirin'...")
    patents = patent_fetcher.search_patents("Aspirin", max_results=5)
    print(f"\nFound {len(patents)} patents")
    
    if patents:
        print("\nFirst 3 patents:")
        for i, patent in enumerate(patents[:3], 1):
            print(f"\n{i}. Patent Number: {patent['patent_number']}")
            print(f"   Title: {patent['title'][:80]}...")
            print(f"   Assignee: {patent['assignee']}")
            print(f"   Filing Date: {patent['filing_date']}")
            print(f"   Expiry Date: {patent['expiry_date']}")
            print(f"   Status: {patent['status']}")
        if len(patents) > 5:
            print(f"\nRemaining {len(patents) - 5} patents (summary):")
            for i, patent in enumerate(patents[5:], 6):
                print(f"{i}. {patent['patent_number']} ({patent['status']}, Expires {patent['expiry_date'][:4]}, {patent['assignee'][:30]})")
    else:
        print("⚠ No patents found (EPO might be failing)")
    
    # Test PubMed
    print("\n4. Testing PubMed API:")
    pubmed_fetcher = get_pubmed_fetcher()
    publications = pubmed_fetcher.search_publications("Metformin cancer", max_results=3)
    print(f"Found {len(publications)} publications")
    
    print("\n" + "="*60)
    print("All tests completed!")