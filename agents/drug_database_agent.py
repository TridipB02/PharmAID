"""
Drug Database Agent - Pharmaceutical Knowledge Nucleus
Integrates PubChem REST API to provide canonical drug information
Serves as the central knowledge source for all other agents
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import requests


class DrugDatabaseAgent:
    """
    Drug Database Intelligence Agent

    Responsibilities:
    - Fetch canonical drug information from PubChem
    - Maintain drug synonyms and brand names
    - Provide mechanism of action and classification
    - Support drug name resolution and standardization
    - Cache frequently accessed drug data
    - Enrich other agents with drug context
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize Drug Database Agent

        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose

        # PubChem API endpoints
        self.pubchem_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.compound_url = f"{self.pubchem_base}/compound"
        self.substance_url = f"{self.pubchem_base}/substance"

        # Cache for drug data (reduce API calls)
        self.drug_cache = {}

        # Session for persistent connections
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "PharmaAgenticAI/1.0 (Educational/Research)"}
        )

        # Rate limiting (PubChem allows ~5 requests/second)
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests

        if self.verbose:
            print("✓ Drug Database Agent initialized")
            print("  Data source: PubChem REST API (NCBI)")

    def _rate_limit(self):
        """Enforce rate limiting for API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

    def get_drug_info(self, drug_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive drug information from PubChem

        Args:
            drug_name: Drug name (generic or brand)
            use_cache: Whether to use cached data

        Returns:
            Dictionary with complete drug information
        """
        if self.verbose:
            print(f"\n[Drug Database Agent] Fetching data for: {drug_name}")

        # Check cache first
        cache_key = drug_name.lower()
        if use_cache and cache_key in self.drug_cache:
            if self.verbose:
                print(f"  ✓ Retrieved from cache")
            return self.drug_cache[cache_key]

        result = {
            "query": drug_name,
            "found": False,
            "pubchem_cid": None,
            "canonical_name": None,
            "iupac_name": None,
            "molecular_formula": None,
            "molecular_weight": None,
            "synonyms": [],
            "brand_names": [],
            "drug_class": None,
            "mechanism_of_action": None,
            "atc_codes": [],
            "therapeutic_area": None,
            "indications": [],
            "pharmacology": {},
            "manufacturers": [],
            "status": None,
            "chemical_structure": {},
            "data_sources": ["PubChem"],
            "error": None,
        }

        try:
            # Step 1: Search for compound by name
            search_result = self._search_compound_by_name(drug_name)

            if not search_result:
                result["error"] = "Compound not found in PubChem"
                if self.verbose:
                    print(f"  ✗ Not found in PubChem")
                return result

            cid = search_result["cid"]
            result["pubchem_cid"] = cid
            result["found"] = True

            # Step 2: Get detailed compound properties
            compound_data = self._get_compound_properties(cid)

            if compound_data:
                result.update(compound_data)

            # Step 3: Get drug/pharmacology information
            drug_data = self._get_drug_information(cid)

            if drug_data:
                result.update(drug_data)

            # Step 4: Enrich with synonyms and brand names
            synonyms_data = self._get_compound_synonyms(cid)

            if synonyms_data:
                result["synonyms"] = synonyms_data.get("synonyms", [])
                result["brand_names"] = self._extract_brand_names(
                    synonyms_data.get("synonyms", [])
                )

            # Cache the result
            self.drug_cache[cache_key] = result

            if self.verbose:
                print(f"  ✓ Retrieved data for CID: {cid}")
                print(f"    Canonical name: {result.get('canonical_name', 'N/A')}")
                print(f"    Synonyms found: {len(result['synonyms'])}")

        except Exception as e:
            result["error"] = str(e)
            if self.verbose:
                print(f"  ✗ Error: {e}")

        return result

    def _search_compound_by_name(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Search PubChem for compound by name

        Args:
            drug_name: Drug name to search

        Returns:
            Dictionary with CID or None
        """
        try:
            self._rate_limit()

            # Search endpoint
            url = f"{self.compound_url}/name/{drug_name}/cids/JSON"

            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                cids = data.get("IdentifierList", {}).get("CID", [])

                if cids:
                    return {"cid": cids[0]}  # Return first CID

            return None

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Search error: {e}")
            return None

    def _get_compound_properties(self, cid: int) -> Dict[str, Any]:
        """
        Get compound properties from PubChem

        Args:
            cid: PubChem Compound ID

        Returns:
            Dictionary with properties
        """
        try:
            self._rate_limit()

            # Properties to fetch
            properties = [
                "IUPACName",
                "MolecularFormula",
                "MolecularWeight",
                "CanonicalSMILES",
                "InChI",
                "InChIKey",
            ]

            properties_str = ",".join(properties)
            url = f"{self.compound_url}/cid/{cid}/property/{properties_str}/JSON"

            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                props = data.get("PropertyTable", {}).get("Properties", [])

                if props:
                    prop_data = props[0]

                    return {
                        "iupac_name": prop_data.get("IUPACName"),
                        "molecular_formula": prop_data.get("MolecularFormula"),
                        "molecular_weight": prop_data.get("MolecularWeight"),
                        "chemical_structure": {
                            "smiles": prop_data.get("CanonicalSMILES"),
                            "inchi": prop_data.get("InChI"),
                            "inchi_key": prop_data.get("InChIKey"),
                        },
                    }

            return {}

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Properties error: {e}")
            return {}

    def _get_compound_synonyms(self, cid: int) -> Dict[str, Any]:
        """
        Get compound synonyms (includes brand names)

        Args:
            cid: PubChem Compound ID

        Returns:
            Dictionary with synonyms
        """
        try:
            self._rate_limit()

            url = f"{self.compound_url}/cid/{cid}/synonyms/JSON"

            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                info = data.get("InformationList", {}).get("Information", [])

                if info:
                    synonyms = info[0].get("Synonym", [])

                    # Limit to first 100 synonyms (can be thousands)
                    return {"synonyms": synonyms[:100]}

            return {}

        except Exception as e:
            if self.verbose:
                print(f"  Synonyms error: {e}")
            return {}

    def _get_drug_information(self, cid: int) -> Dict[str, Any]:
        """
        Get drug/pharmacology information
        Uses PubChem's Classification and Description endpoints

        Args:
            cid: PubChem Compound ID

        Returns:
            Dictionary with drug information
        """
        drug_info = {}

        try:
            # Get classifications
            self._rate_limit()
            url = f"{self.compound_url}/cid/{cid}/classification/JSON"

            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                hierarchies = data.get("Hierarchies", {}).get("Hierarchy", [])

                # Extract ATC codes and therapeutic areas
                for hierarchy in hierarchies:
                    source = hierarchy.get("SourceName", "")

                    if "ATC" in source:
                        # ATC Classification
                        nodes = hierarchy.get("Node", [])
                        atc_codes = self._extract_atc_codes(nodes)
                        drug_info["atc_codes"] = atc_codes

                    elif "Therapeutic" in source or "Drug" in source:
                        # Therapeutic classification
                        nodes = hierarchy.get("Node", [])
                        drug_info["drug_class"] = self._extract_drug_class(nodes)

            # Get description (mechanism of action, indications)
            self._rate_limit()
            desc_url = f"{self.compound_url}/cid/{cid}/description/JSON"

            desc_response = self.session.get(desc_url, timeout=10)

            if desc_response.status_code == 200:
                desc_data = desc_response.json()
                info = desc_data.get("InformationList", {}).get("Information", [])

                if info:
                    description = info[0]

                    drug_info["canonical_name"] = description.get("Title")

                    # Extract mechanism and indications from description
                    desc_text = description.get("Description", "")
                    drug_info["mechanism_of_action"] = self._extract_mechanism(
                        desc_text
                    )
                    drug_info["indications"] = self._extract_indications(desc_text)
                    drug_info["pharmacology"] = {
                        "description": (
                            desc_text[:500] + "..."
                            if len(desc_text) > 500
                            else desc_text
                        )
                    }

        except Exception as e:
            if self.verbose:
                print(f" Drug info error: {e}")

        return drug_info

    def _extract_atc_codes(self, nodes: List[Dict]) -> List[str]:
        """Extract ATC codes from classification nodes"""
        atc_codes = []

        for node in nodes:
            node_id = node.get("Information", {}).get("Identifier", "")
            if node_id and len(node_id) <= 7:  # ATC codes are max 7 chars
                atc_codes.append(node_id)

        return atc_codes

    def _extract_drug_class(self, nodes: List[Dict]) -> Optional[str]:
        """Extract drug class from classification nodes"""
        for node in nodes:
            name = node.get("Information", {}).get("Name", "")
            if name and "agent" in name.lower() or "inhibitor" in name.lower():
                return name

        # Return first meaningful classification
        if nodes:
            return nodes[0].get("Information", {}).get("Name")

        return None

    def _extract_mechanism(self, description: str) -> Optional[str]:
        """Extract mechanism of action from description"""
        # Look for mechanism keywords
        moa_keywords = [
            "mechanism of action",
            "works by",
            "inhibits",
            "activates",
            "blocks",
            "binds to",
        ]

        description_lower = description.lower()

        for keyword in moa_keywords:
            if keyword in description_lower:
                # Extract sentence containing keyword
                sentences = description.split(".")
                for sentence in sentences:
                    if keyword in sentence.lower():
                        return sentence.strip()

        return None

    def _extract_indications(self, description: str) -> List[str]:
        """Extract indications from description"""
        indications = []

        # Look for indication keywords
        indication_keywords = [
            "used to treat",
            "indicated for",
            "treatment of",
            "used for",
        ]

        description_lower = description.lower()

        for keyword in indication_keywords:
            if keyword in description_lower:
                # Extract sentence
                sentences = description.split(".")
                for sentence in sentences:
                    if keyword in sentence.lower():
                        # Extract conditions mentioned
                        conditions = self._extract_conditions(sentence)
                        indications.extend(conditions)

        return list(set(indications))  # Remove duplicates

    def _extract_conditions(self, text: str) -> List[str]:
        """Extract medical conditions from text"""
        # Common medical conditions (basic extraction)
        conditions = [
            "diabetes",
            "hypertension",
            "cancer",
            "asthma",
            "depression",
            "pain",
            "infection",
            "inflammation",
            "obesity",
            "alzheimer",
            "parkinson",
            "arthritis",
            "cholesterol",
            "heart disease",
        ]

        found_conditions = []
        text_lower = text.lower()

        for condition in conditions:
            if condition in text_lower:
                found_conditions.append(condition.title())

        return found_conditions

    def _extract_brand_names(self, synonyms: List[str]) -> List[str]:
        """
        Extract likely brand names from synonyms
        Brand names are usually capitalized and shorter
        """
        brand_names = []

        for synonym in synonyms[:50]:  # Check first 50 synonyms
            # Heuristics for brand names:
            # 1. Capitalized first letter
            # 2. Not all uppercase (chemical names)
            # 3. Shorter than 20 characters
            # 4. Doesn't contain numbers at start

            if (
                synonym
                and len(synonym) < 20
                and synonym[0].isupper()
                and not synonym.isupper()
                and not synonym[0].isdigit()
            ):

                brand_names.append(synonym)

        return brand_names[:20]  # Limit to 20 brand names

    def resolve_drug_name(self, query: str, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Resolve ambiguous drug name to canonical form
        Handles typos, brand names, generic names

        Args:
            query: Drug name query (possibly misspelled)
            threshold: Similarity threshold (0-1)

        Returns:
            Dictionary with resolution results
        """
        if self.verbose:
            print(f"\n[Drug Database Agent] Resolving drug name: '{query}'")

        # First try exact match
        drug_info = self.get_drug_info(query)

        if drug_info["found"]:
            return {
                "query": query,
                "resolved": True,
                "canonical_name": drug_info["canonical_name"],
                "confidence": 1.0,
                "suggestions": [],
                "drug_info": drug_info,
            }

        # If not found, try fuzzy matching with common drugs
        # (In production, you'd use a fuzzy matching library)

        return {
            "query": query,
            "resolved": False,
            "canonical_name": None,
            "confidence": 0.0,
            "suggestions": ["Check spelling", "Try generic name", "Try brand name"],
            "drug_info": None,
        }

    def get_drug_synonyms(self, drug_name: str) -> List[str]:
        """
        Get all known synonyms for a drug
        Useful for clinical trials and patent searches

        Args:
            drug_name: Drug name

        Returns:
            List of synonyms
        """
        drug_info = self.get_drug_info(drug_name)

        if drug_info["found"]:
            synonyms = set(drug_info["synonyms"])
            synonyms.add(drug_info["canonical_name"])
            synonyms.update(drug_info["brand_names"])

            # Remove None values
            synonyms.discard(None)

            return sorted(list(synonyms))

        return []

    def enrich_drug_context(self, drug_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Batch enrich multiple drugs with context
        Used by other agents to get drug metadata

        Args:
            drug_names: List of drug names

        Returns:
            Dictionary mapping drug names to their info
        """
        if self.verbose:
            print(
                f"\n[Drug Database Agent] Enriching context for {len(drug_names)} drugs"
            )

        enriched_data = {}

        for drug_name in drug_names:
            drug_info = self.get_drug_info(drug_name)

            if drug_info["found"]:
                enriched_data[drug_name] = {
                    "canonical_name": drug_info["canonical_name"],
                    "drug_class": drug_info["drug_class"],
                    "therapeutic_area": drug_info["therapeutic_area"],
                    "mechanism": drug_info["mechanism_of_action"],
                    "synonyms": drug_info["synonyms"][:10],  # Top 10
                    "brand_names": drug_info["brand_names"][:5],  # Top 5
                }
            else:
                enriched_data[drug_name] = None

        return enriched_data

    def compare_drugs(self, drug_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple drugs side-by-side

        Args:
            drug_names: List of drugs to compare

        Returns:
            Comparison dictionary
        """
        if self.verbose:
            print(f"\n[Drug Database Agent] Comparing {len(drug_names)} drugs")

        comparisons = []

        for drug_name in drug_names:
            drug_info = self.get_drug_info(drug_name)

            if drug_info["found"]:
                comparisons.append(
                    {
                        "drug": drug_name,
                        "canonical_name": drug_info["canonical_name"],
                        "drug_class": drug_info["drug_class"],
                        "molecular_weight": drug_info["molecular_weight"],
                        "atc_codes": drug_info["atc_codes"],
                        "indications": drug_info["indications"],
                    }
                )

        return {
            "drugs_compared": len(comparisons),
            "comparison_table": comparisons,
            "summary": f"Compared {len(comparisons)} drugs from PubChem database",
        }

    def search_by_therapeutic_area(self, therapeutic_area: str) -> Dict[str, Any]:
        """
        Search drugs by therapeutic area
        Note: PubChem doesn't have direct therapeutic area search
        This is a placeholder for future enhancement

        Args:
            therapeutic_area: Therapeutic area name

        Returns:
            Search results
        """
        return {
            "therapeutic_area": therapeutic_area,
            "note": "Direct therapeutic area search requires additional data source",
            "recommendation": "Use IQVIA Agent or internal database for therapy area filtering",
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached drug data"""
        return {
            "cached_drugs": len(self.drug_cache),
            "cache_keys": list(self.drug_cache.keys()),
        }

    def clear_cache(self):
        """Clear drug data cache"""
        self.drug_cache.clear()
        if self.verbose:
            print("✓ Drug cache cleared")


# Convenience function
def get_drug_database_agent(verbose: bool = True) -> DrugDatabaseAgent:
    """
    Get instance of Drug Database Agent

    Args:
        verbose: Whether to enable verbose logging

    Returns:
        Initialized DrugDatabaseAgent instance
    """
    return DrugDatabaseAgent(verbose=verbose)


# Test the agent
if __name__ == "__main__":
    print("=" * 70)
    print("DRUG DATABASE AGENT - TEST SUITE")
    print("=" * 70)

    # Initialize agent
    agent = get_drug_database_agent(verbose=True)

    # Test 1: Get drug info - Metformin
    print("\n" + "=" * 70)
    print("TEST 1: Get Drug Info - Metformin")
    print("=" * 70)
    metformin_info = agent.get_drug_info("Metformin")
    print(json.dumps(metformin_info, indent=2))

    # Test 2: Get drug synonyms
    print("\n" + "=" * 70)
    print("TEST 2: Get Drug Synonyms - Ibuprofen")
    print("=" * 70)
    synonyms = agent.get_drug_synonyms("Ibuprofen")
    print(f"Found {len(synonyms)} synonyms:")
    print(synonyms[:10])  # Show first 10

    # Test 3: Enrich multiple drugs
    print("\n" + "=" * 70)
    print("TEST 3: Enrich Drug Context")
    print("=" * 70)
    enriched = agent.enrich_drug_context(["Aspirin", "Atorvastatin", "Lisinopril"])
    print(json.dumps(enriched, indent=2))

    # Test 4: Compare drugs
    print("\n" + "=" * 70)
    print("TEST 4: Compare Drugs")
    print("=" * 70)
    comparison = agent.compare_drugs(["Metformin", "Insulin", "Glipizide"])
    print(json.dumps(comparison, indent=2))

    # Test 5: Cache stats
    print("\n" + "=" * 70)
    print("TEST 5: Cache Statistics")
    print("=" * 70)
    cache_stats = agent.get_cache_stats()
    print(json.dumps(cache_stats, indent=2))

    print("\n All Drug Database Agent tests completed!")
