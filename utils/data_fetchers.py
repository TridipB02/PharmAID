# data_fetchers.py
"""
Data Fetchers for Pharma Agentic AI System
Handles loading mock data and fetching from real APIs
"""

import json

# Import settings
import sys
import time
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


class PatentFetcher(DataFetcher):
    """Fetches patent data using Google Patents Public Search"""

    def __init__(self, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.max_retries = 2
        self.timeout = 15

        # Patent term rules
        self.patent_terms = {"utility": 20, "design": 15, "plant": 20, "reissue": 20}

    def search_patents(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search patents using a simple heuristic approach
        Returns mock-like structured data for demonstration

        Args:
            query: Patent search query (drug name)
            max_results: Maximum results (limited to 10)

        Returns:
            List of patent dictionaries
        """
        if not query or not query.strip():
            if self.verbose:
                print("❌ Empty query provided")
            return []

        if self.verbose:
            print(f"⚠ Using mock patent data (Patent APIs require authentication)")

        # Generate structured mock patents based on query
        patents = self._generate_mock_patents(query, max_results)

        if self.verbose:
            print(f"✓ Generated {len(patents)} mock patents for demonstration")

        return patents

    def _generate_mock_patents(self, query: str, count: int) -> List[Dict]:
        """Generate structured mock patent data"""
        import random
        from datetime import datetime

        patents = []
        current_year = datetime.now().year

        # Common pharmaceutical companies
        assignees = [
            "Pfizer Inc.",
            "Johnson & Johnson",
            "Novartis AG",
            "Roche Holding AG",
            "Merck & Co.",
            "GlaxoSmithKline",
            "Sanofi",
            "AstraZeneca",
            "Bristol-Myers Squibb",
            "Eli Lilly",
        ]

        # Patent title templates
        title_templates = [
            f"Pharmaceutical composition comprising {query}",
            f"Methods of treating diseases using {query}",
            f"Novel formulation of {query} for enhanced delivery",
            f"{query} derivatives and therapeutic applications",
            f"Sustained release composition containing {query}",
            f"Combination therapy comprising {query}",
            f"Process for preparing {query} pharmaceutical composition",
            f"{query}-based drug delivery system",
            f"Oral dosage form of {query}",
            f"Injectable formulation containing {query}",
        ]

        for i in range(min(count, 10)):
            # Generate patent details
            grant_year = random.randint(2005, 2023)
            grant_date = (
                f"{grant_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            )
            filing_year = grant_year - random.randint(2, 4)
            filing_date = (
                f"{filing_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            )

            # Calculate expiry (20 years from grant)
            expiry_year = grant_year + 20
            expiry_date = (
                f"{expiry_year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            )

            # Determine status
            status = "Active" if expiry_year > current_year else "Expired"

            patent = {
                "patent_number": f"US{random.randint(7000000, 11000000)}B2",
                "title": random.choice(title_templates),
                "filing_date": filing_date,
                "grant_date": grant_date,
                "expiry_date": expiry_date,
                "assignee": random.choice(assignees),
                "status": status,
                "patent_type": "utility",
                "claims_count": str(random.randint(10, 50)),
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
                        print("  ⚠ UN Comtrade rate limit hit, waiting 10 seconds...")
                    time.sleep(10)
                    response = self.session.get(url, params=params, timeout=30)

                # Debug bad requests
                if response.status_code != 200:
                    if self.verbose:
                        print(f"  ⚠ Status: {response.status_code}")
                        print(f"  Response: {response.text[:200]}")
                    continue  # Skip and try next country

                data = response.json()
                records = data.get("data", data.get("dataset", data.get("results", [])))

                if records:
                    all_records.extend(records)
                    if self.verbose:
                        print(f"  ✓ Retrieved {len(records)} trade records")

            if self.verbose:
                if all_records:
                    print(f"  ✓ Total: {len(all_records)} records from UN Comtrade")
                else:
                    print(f"  ⚠ No data returned from UN Comtrade")

            return all_records[:50]

        except requests.exceptions.RequestException as e:
            if self.verbose:
                print(f"  ✗ UN Comtrade API error: {e}")
            return []
        except Exception as e:
            if self.verbose:
                print(f"  ✗ Unexpected error in UN Comtrade: {e}")
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
                    print(f"  ✓ Retrieved {len(records)} records from World Bank")

                return records

            return []

        except requests.exceptions.RequestException as e:
            if self.verbose:
                print(f"  ✗ World Bank API error: {e}")
            return []
        except Exception as e:
            if self.verbose:
                print(f"  ✗ Unexpected error in World Bank: {e}")
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
        # ✅ USITC removed completely

    def get_comprehensive_trade_data(
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
                print(f"  ⚠ UN Comtrade failed: {e}")

        # ✅ FIXED: Always try World Bank (not conditional)
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
                    print(f"  ✓ World Bank: {len(wb_data)} records retrieved")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ World Bank failed: {e}")

        # Generate summary
        if not result["data_sources"]:
            if self.verbose:
                print("  ⚠ No real trade data available from any API")
            result["summary"] = (
                "No trade data available - APIs failed or returned empty"
            )
        else:
            result["summary"] = (
                f"Trade data from {len(result['data_sources'])} source(s): {', '.join(result['data_sources'])}"
            )

        if self.verbose:
            print(f"  ✓ {result['summary']}")

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


def get_patent_fetcher() -> PatentFetcher:
    """Get instance of PatentFetcher"""
    return PatentFetcher()


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

    # Test PubMed
    print("\n3. Testing PubMed API:")
    pubmed_fetcher = get_pubmed_fetcher()
    publications = pubmed_fetcher.search_publications("Metformin cancer", max_results=3)
    print(f"Found {len(publications)} publications")

    print("\nAll tests completed!")
