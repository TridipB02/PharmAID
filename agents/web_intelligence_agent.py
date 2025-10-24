"""
Web Intelligence Agent - Real-time Information Specialist
Gathers current guidelines, publications, and market signals from web sources
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_fetchers import PubMedFetcher, OpenFDAFetcher


class WebIntelligenceAgent:
    """
    Web Intelligence Agent
    
    Responsibilities:
    - Search scientific literature (PubMed)
    - Retrieve FDA drug information
    - Gather treatment guidelines
    - Monitor recent publications
    - Identify research trends
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize Web Intelligence Agent
        
        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.pubmed_fetcher = PubMedFetcher()
        self.fda_fetcher = OpenFDAFetcher()
        
        if self.verbose:
            print("✓ Web Intelligence Agent initialized")
    
    def search_literature(
        self,
        keywords: List[str] = None,
        drugs: List[str] = None,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Comprehensive literature search
        
        Args:
            keywords: Search keywords
            drugs: Drug names to search for
            max_results: Maximum number of publications to retrieve
        
        Returns:
            Dictionary with literature search results
        """
        if self.verbose:
            print(f"\n[Web Intelligence Agent] Searching literature...")
            print(f"  Keywords: {keywords}")
            print(f"  Drugs: {drugs}")
        
        results = {
            "summary": "",
            "total_publications_found": 0,
            "publications_by_topic": {},
            "fda_information": {},
            "research_trends": [],
            "key_findings": [],
            "detailed_publications": []
        }
        
        all_publications = []
        
        # Search by keywords
        if keywords:
            for keyword in keywords:
                publications = self.pubmed_fetcher.search_publications(
                    keyword, 
                    max_results=max_results
                )
                if publications:
                    results["publications_by_topic"][keyword] = {
                        "count": len(publications),
                        "publications": publications
                    }
                    all_publications.extend(publications)
                    
                    if self.verbose:
                        print(f"  Found {len(publications)} publications for: {keyword}")
        
        # Search by drugs
        if drugs:
            for drug in drugs:
                # PubMed search
                publications = self.pubmed_fetcher.search_publications(
                    drug, 
                    max_results=max_results
                )
                if publications:
                    all_publications.extend(publications)
                    if self.verbose:
                        print(f"  Found {len(publications)} publications for drug: {drug}")
                
                # FDA information
                fda_info = self.fda_fetcher.get_drug_label(drug)
                if fda_info:
                    results["fda_information"][drug] = fda_info
                    if self.verbose:
                        print(f"  Retrieved FDA information for: {drug}")
        
        # Remove duplicate publications (by PMID)
        unique_pubs = {pub['pmid']: pub for pub in all_publications}.values()
        results["detailed_publications"] = list(unique_pubs)
        results["total_publications_found"] = len(results["detailed_publications"])
        
        # Analyze publications
        if results["detailed_publications"]:
            results["research_trends"] = self._analyze_research_trends(results)
            results["key_findings"] = self._extract_key_findings(results)
        
        # Generate summary
        results["summary"] = self._generate_summary(results)
        
        if self.verbose:
            print(f"✓ Search complete - {results['total_publications_found']} publications found")
        
        return results
    
    def _analyze_research_trends(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze research trends from publications
        
        Args:
            results: Search results
        
        Returns:
            List of research trend insights
        """
        trends = []
        
        # Analyze topics
        if results['publications_by_topic']:
            topic_counts = {
                topic: data['count'] 
                for topic, data in results['publications_by_topic'].items()
            }
            
            # Identify hot topics
            sorted_topics = sorted(
                topic_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            if sorted_topics:
                top_topic = sorted_topics[0]
                trends.append({
                    "type": "Hot Research Topic",
                    "topic": top_topic[0],
                    "publication_count": top_topic[1],
                    "note": f"Highest research activity in this area"
                })
        
        # Analyze FDA data
        if results['fda_information']:
            trends.append({
                "type": "FDA Data Available",
                "drugs_with_fda_info": len(results['fda_information']),
                "note": "Official FDA drug information retrieved"
            })
        
        return trends
    
    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """
        Extract key findings from literature search
        
        Args:
            results: Search results
        
        Returns:
            List of key finding strings
        """
        findings = []
        
        total_pubs = results['total_publications_found']
        if total_pubs > 0:
            findings.append(f"Found {total_pubs} relevant scientific publications")
        
        # FDA findings
        if results['fda_information']:
            fda_count = len(results['fda_information'])
            findings.append(f"Retrieved FDA labels for {fda_count} drug(s)")
            
            # Check for warnings
            for drug, info in results['fda_information'].items():
                if info.get('warnings') and info['warnings'] != 'N/A':
                    findings.append(f"{drug}: FDA warnings available - review recommended")
        
        # Publication volume
        if results['publications_by_topic']:
            most_researched = max(
                results['publications_by_topic'].items(),
                key=lambda x: x[1]['count']
            )
            findings.append(
                f"Most researched topic: '{most_researched[0]}' ({most_researched[1]['count']} publications)"
            )
        
        return findings
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate executive summary
        
        Args:
            results: Analysis results
        
        Returns:
            Summary string
        """
        total_pubs = results['total_publications_found']
        
        if total_pubs == 0 and not results['fda_information']:
            return "No literature or FDA information found for the search criteria."
        
        summary_parts = []
        
        if total_pubs > 0:
            summary_parts.append(f"Found {total_pubs} relevant publication(s).")
        
        if results['fda_information']:
            fda_count = len(results['fda_information'])
            summary_parts.append(f"Retrieved FDA information for {fda_count} drug(s).")
        
        if results['research_trends']:
            summary_parts.append(f"{len(results['research_trends'])} research trend(s) identified.")
        
        return " ".join(summary_parts)
    
    def get_drug_guidelines(self, drug_name: str, indication: str = None) -> Dict[str, Any]:
        """
        Get treatment guidelines for a drug
        
        Args:
            drug_name: Name of the drug
            indication: Specific indication (optional)
        
        Returns:
            Guidelines information
        """
        if self.verbose:
            print(f"\n[Web Intelligence Agent] Searching guidelines for: {drug_name}")
        
        # Search PubMed for guidelines
        query = f"{drug_name} guidelines"
        if indication:
            query += f" {indication}"
        
        publications = self.pubmed_fetcher.search_publications(query, max_results=10)
        
        # Get FDA info
        fda_info = self.fda_fetcher.get_drug_label(drug_name)
        
        return {
            "drug": drug_name,
            "indication": indication,
            "guideline_publications": publications,
            "fda_indications": fda_info.get('indications') if fda_info else None,
            "guideline_count": len(publications),
            "fda_data_available": fda_info is not None
        }
    
    def search_repurposing_evidence(
        self, 
        drug_name: str, 
        new_indication: str
    ) -> Dict[str, Any]:
        """
        Search for evidence of drug repurposing
        
        Args:
            drug_name: Name of the drug
            new_indication: Potential new indication
        
        Returns:
            Repurposing evidence dictionary
        """
        if self.verbose:
            print(f"\n[Web Intelligence Agent] Searching repurposing evidence...")
            print(f"  Drug: {drug_name}")
            print(f"  New indication: {new_indication}")
        
        # Search for drug + new indication
        query = f"{drug_name} {new_indication}"
        publications = self.pubmed_fetcher.search_publications(query, max_results=20)
        
        # Get original FDA indication
        fda_info = self.fda_fetcher.get_drug_label(drug_name)
        original_indication = fda_info.get('indications') if fda_info else "Unknown"
        
        # Assess evidence strength
        evidence_strength = "Strong" if len(publications) >= 10 else "Moderate" if len(publications) >= 5 else "Limited"
        
        return {
            "drug": drug_name,
            "original_indication": original_indication,
            "proposed_indication": new_indication,
            "publications_found": len(publications),
            "evidence_strength": evidence_strength,
            "publications": publications,
            "repurposing_potential": self._assess_repurposing_potential(
                len(publications), 
                new_indication
            ),
            "recommendation": self._generate_repurposing_recommendation(
                evidence_strength, 
                len(publications)
            )
        }
    
    def _assess_repurposing_potential(
        self, 
        publication_count: int, 
        indication: str
    ) -> str:
        """
        Assess repurposing potential
        
        Args:
            publication_count: Number of publications found
            indication: Target indication
        
        Returns:
            Potential assessment
        """
        if publication_count >= 15:
            return "High - Strong research support"
        elif publication_count >= 10:
            return "Moderate-High - Good research base"
        elif publication_count >= 5:
            return "Moderate - Some evidence available"
        else:
            return "Low - Limited research evidence"
    
    def _generate_repurposing_recommendation(
        self, 
        evidence_strength: str, 
        pub_count: int
    ) -> str:
        """
        Generate recommendation for repurposing
        
        Args:
            evidence_strength: Strength of evidence
            pub_count: Number of publications
        
        Returns:
            Recommendation string
        """
        if evidence_strength == "Strong":
            return f"Proceed with clinical evaluation - {pub_count} publications support this use"
        elif evidence_strength == "Moderate":
            return f"Conduct feasibility study - {pub_count} publications provide preliminary support"
        else:
            return f"High risk - only {pub_count} publications found, requires extensive research"
    
    def compare_research_activity(self, drugs: List[str]) -> Dict[str, Any]:
        """
        Compare research activity across multiple drugs
        
        Args:
            drugs: List of drug names
        
        Returns:
            Comparative research analysis
        """
        if self.verbose:
            print(f"\n[Web Intelligence Agent] Comparing research activity: {drugs}")
        
        comparisons = []
        
        for drug in drugs:
            publications = self.pubmed_fetcher.search_publications(drug, max_results=50)
            fda_info = self.fda_fetcher.get_drug_label(drug)
            
            comparisons.append({
                "drug": drug,
                "publication_count": len(publications),
                "fda_approved": fda_info is not None,
                "research_activity": "High" if len(publications) >= 30 else "Moderate" if len(publications) >= 15 else "Low"
            })
        
        # Sort by publication count
        comparisons.sort(key=lambda x: x['publication_count'], reverse=True)
        
        return {
            "drugs_compared": len(comparisons),
            "comparison_table": comparisons,
            "most_researched": comparisons[0]['drug'] if comparisons else None,
            "total_publications": sum(c['publication_count'] for c in comparisons),
            "summary": self._generate_comparison_summary(comparisons)
        }
    
    def _generate_comparison_summary(self, comparisons: List[Dict]) -> str:
        """Generate summary of research comparison"""
        if not comparisons:
            return "No drugs to compare"
        
        total_pubs = sum(c['publication_count'] for c in comparisons)
        leader = comparisons[0]['drug']
        
        return f"Compared {len(comparisons)} drugs with {total_pubs} total publications. " \
               f"Most researched: {leader} ({comparisons[0]['publication_count']} publications)."


# Convenience function
def get_web_intelligence_agent(verbose: bool = True) -> WebIntelligenceAgent:
    """
    Get instance of Web Intelligence Agent
    
    Args:
        verbose: Whether to enable verbose logging
    
    Returns:
        Initialized WebIntelligenceAgent instance
    """
    return WebIntelligenceAgent(verbose=verbose)


# Test the agent
if __name__ == "__main__":
    print("="*70)
    print("WEB INTELLIGENCE AGENT - TEST SUITE")
    print("="*70)
    
    # Initialize agent
    agent = get_web_intelligence_agent(verbose=True)
    
    # Test 1: Literature search
    print("\n" + "="*70)
    print("TEST 1: Literature Search - Metformin Cancer")
    print("="*70)
    result = agent.search_literature(keywords=["metformin cancer"], max_results=5)
    print(f"\nSummary: {result['summary']}")
    print(f"Total publications: {result['total_publications_found']}")
    print(f"Key findings: {result['key_findings']}")
    
    # Test 2: Drug guidelines
    print("\n" + "="*70)
    print("TEST 2: Drug Guidelines - Metformin")
    print("="*70)
    guidelines = agent.get_drug_guidelines("Metformin", "Type 2 Diabetes")
    print(json.dumps(guidelines, indent=2))
    
    # Test 3: Repurposing evidence
    print("\n" + "="*70)
    print("TEST 3: Repurposing Evidence - Metformin for Alzheimer's")
    print("="*70)
    repurposing = agent.search_repurposing_evidence("Metformin", "Alzheimer")
    print(json.dumps(repurposing, indent=2))
    
    # Test 4: Compare research activity
    print("\n" + "="*70)
    print("TEST 4: Research Activity Comparison")
    print("="*70)
    comparison = agent.compare_research_activity(["Metformin", "Ibuprofen", "Atorvastatin"])
    print(json.dumps(comparison, indent=2))
    
    print("\n✓ All Web Intelligence Agent tests completed!")