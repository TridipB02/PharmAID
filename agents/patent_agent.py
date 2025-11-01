"""
Patent Landscape Agent - Intellectual Property Specialist
Analyzes patent filings, expiry timelines, and FTO (Freedom to Operate) risks
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_fetchers import EPOPatentFetcher


class PatentAgent:
    """
    Patent Landscape Intelligence Agent

    Responsibilities:
    - Search patents by molecule, indication, or technology
    - Track patent expiry timelines and exclusivity windows
    - Assess Freedom to Operate (FTO) risks
    - Identify white space opportunities
    - Analyze competitive patent filings
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize Patent Agent

        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.fetcher = EPOPatentFetcher(verbose=verbose)

        if self.verbose:
            print("✓ Patent Agent initialized")

    def _filter_keywords(self, keywords: List[str]) -> List[str]:
        """
        Filter out generic/useless keywords that pollute patent searches
        Args:
            keywords: List of raw keywords from query parser
        Returns:
             Filtered list of meaningful keywords
        """

        skip_keywords = [
            # Generic words
            'patents', 'patent', 'recent', 'expiring', 'space', 'area', 'field',
            'search', 'find', 'get', 'show', 'list', 'top', 'best',
        
            # Action words
            'repurposed', 'novel', 'new', 'innovative', 'advanced',
            'companies', 'jointly', 'filed', 'have', 'that', 'could',
            'would', 'should', 'might', 'for', 'with', 'and', 'or',
        
            # Question words
            'what', 'which', 'who', 'where', 'when', 'how', 'why',
        
            # Meta terms
            'related', 'referencing', 'about', 'regarding', 'concerning'
        ]

        good_keywords = []
        for keyword in keywords:
            if keyword.lower() in skip_keywords:
                continue
            if len(keyword) < 4:
                continue

            words = keyword.lower().split()
            if len(words) > 1 and all(word in skip_keywords for word in words):
                continue

            words = keyword.lower().split()
            if len(words) > 1 and all(word in skip_keywords for word in words):
                continue

            good_keywords.append(keyword)

        if not good_keywords and keywords:
            good_keywords = keywords[:2]
        
        return good_keywords


    def search_patents(
        self, drugs: List[str] = None, keywords: List[str] = None, max_results: int = 30
    ) -> Dict[str, Any]:
        """
        Comprehensive patent search and landscape analysis
        """
        if self.verbose:
            print(f"\n[Patent Agent] Searching patents...")
            print(f"  Drugs: {drugs}")
            print(f"  Keywords: {keywords}")

        results = {
            "summary": "",
            "total_patents_found": 0,
            "patents_by_drug": {},
            "expiry_timeline": {},
            "fto_assessment": {},
            "competitive_landscape": {},
            "white_space_opportunities": [],
            "detailed_patents": [],
        }

        all_patents = []

        # Search by drugs
        if drugs:
            for drug in drugs:
                patents = self.fetcher.search_patents(drug, max_results=max_results)
                if patents:
                    results["patents_by_drug"][drug] = {
                        "count": len(patents),
                        "patents": patents,
                    }
                    all_patents.extend(patents)

                    if self.verbose:
                        print(f"  Found {len(patents)} patents for drug: {drug}")

        # Search by keywords
        if keywords:
            filtered_keywords = self._filter_keywords(keywords)

            if self.verbose:
                print(f"  Keywords (filtered): {filtered_keywords}")

            for keyword in filtered_keywords:
                patents = self.fetcher.search_patents(keyword, max_results=max_results)
                if patents:
                    all_patents.extend(patents)

                    if self.verbose:
                        print(f"  Found {len(patents)} patents for keyword: {keyword}")
                
        # Remove duplicates based on patent number
        unique_patents = {p["patent_number"]: p for p in all_patents}.values()
        results["detailed_patents"] = list(unique_patents)
        results["total_patents_found"] = len(results["detailed_patents"])

        # Analyze patents
        if results["detailed_patents"]:
            results["expiry_timeline"] = self._analyze_expiry_timeline(
                results["detailed_patents"]
            )
            results["fto_assessment"] = self._assess_fto(
                results["detailed_patents"], drugs
            )
            results["competitive_landscape"] = self._analyze_competitive_landscape(
                results["detailed_patents"]
            )
            results["white_space_opportunities"] = self._identify_white_space(results)

        # Generate summary
        results["summary"] = self._generate_summary(results)

        if self.verbose:
            print(
                f"✓ Search complete - {results['total_patents_found']} unique patents found"
            )

        return results

    def _analyze_expiry_timeline(self, patents: List[Dict]) -> Dict[str, Any]:
        """
        Analyze patent expiry timeline

        Args:
            patents: List of patent dictionaries

        Returns:
            Expiry timeline analysis
        """
        current_year = datetime.now().year

        # Categorize patents by expiry period
        expiring_soon = []  # Within 2 years
        expiring_medium = []  # 2-5 years
        expiring_long = []  # 5+ years
        already_expired = []

        for patent in patents:
            try:
                expiry_year = int(patent["expiry_date"].split("-")[0])
                years_until_expiry = expiry_year - current_year

                patent_summary = {
                    "patent_number": patent["patent_number"],
                    "title": patent["title"],
                    "expiry_date": patent["expiry_date"],
                    "years_until_expiry": years_until_expiry,
                    "assignee": patent["assignee"],
                }

                if years_until_expiry < 0:
                    already_expired.append(patent_summary)
                elif years_until_expiry <= 2:
                    expiring_soon.append(patent_summary)
                elif years_until_expiry <= 5:
                    expiring_medium.append(patent_summary)
                else:
                    expiring_long.append(patent_summary)

            except (ValueError, KeyError, IndexError):
                continue

        return {
            "expiring_soon_count": len(expiring_soon),
            "expiring_medium_count": len(expiring_medium),
            "expiring_long_count": len(expiring_long),
            "expired_count": len(already_expired),
            "expiring_soon": expiring_soon,
            "expiring_medium": expiring_medium,
            "expiring_long": expiring_long,
            "expired": already_expired,
            "opportunity_window": (
                "High"
                if len(expiring_soon) > 0
                else "Medium" if len(expiring_medium) > 0 else "Low"
            ),
        }

    def _assess_fto(
        self, patents: List[Dict], drugs: List[str] = None
    ) -> Dict[str, Any]:
        """
        Assess Freedom to Operate (FTO) risks

        Args:
            patents: List of patent dictionaries
            drugs: List of drugs being analyzed

        Returns:
            FTO risk assessment
        """
        current_year = datetime.now().year

        # Count active patents
        active_patents = []
        for patent in patents:
            try:
                expiry_year = int(patent["expiry_date"].split("-")[0])
                if expiry_year >= current_year and patent["status"] == "Active":
                    active_patents.append(patent)
            except (ValueError, KeyError, IndexError):
                continue

        # Count unique assignees (competitors)
        assignees = [p["assignee"] for p in active_patents]
        unique_assignees = set(assignees)

        # Determine risk level
        active_count = len(active_patents)
        if active_count == 0:
            risk_level = "Low"
            risk_description = "No active patents blocking development"
        elif active_count <= 3:
            risk_level = "Medium"
            risk_description = f"{active_count} active patent(s) - manageable FTO risk"
        else:
            risk_level = "High"
            risk_description = (
                f"{active_count} active patents - significant FTO concerns"
            )

        return {
            "risk_level": risk_level,
            "risk_description": risk_description,
            "active_patents_count": active_count,
            "unique_patent_holders": len(unique_assignees),
            "top_patent_holders": [
                {"assignee": assignee, "patent_count": assignees.count(assignee)}
                for assignee in sorted(
                    set(assignees), key=lambda x: assignees.count(x), reverse=True
                )[:5]
            ],
            "blocking_patents": [
                {
                    "patent_number": p["patent_number"],
                    "title": p["title"],
                    "assignee": p["assignee"],
                    "expiry_date": p["expiry_date"],
                }
                for p in active_patents[:10]  # Top 10 blocking patents
            ],
        }

    def _analyze_competitive_landscape(self, patents: List[Dict]) -> Dict[str, Any]:
        """
        Analyze competitive patent landscape

        Args:
            patents: List of patent dictionaries

        Returns:
            Competitive landscape analysis
        """
        # Analyze assignees
        assignees = [p["assignee"] for p in patents]
        assignee_counts = Counter(assignees)

        # Analyze filing trends by year
        filing_years = []
        for patent in patents:
            try:
                filing_year = int(patent["filing_date"].split("-")[0])
                filing_years.append(filing_year)
            except (ValueError, KeyError, IndexError):
                continue

        year_counts = Counter(filing_years)

        # Recent activity (last 5 years)
        current_year = datetime.now().year
        recent_filings = sum(1 for year in filing_years if year >= current_year - 5)

        return {
            "total_assignees": len(assignee_counts),
            "top_assignees": [
                {"assignee": assignee, "patent_count": count}
                for assignee, count in assignee_counts.most_common(10)
            ],
            "market_leader": (
                assignee_counts.most_common(1)[0][0] if assignee_counts else "Unknown"
            ),
            "filing_trend": {
                "recent_filings_5yr": recent_filings,
                "annual_breakdown": dict(
                    sorted(year_counts.items(), reverse=True)[:10]
                ),
                "innovation_momentum": (
                    "High"
                    if recent_filings > len(patents) * 0.3
                    else "Moderate" if recent_filings > 0 else "Low"
                ),
            },
            "market_concentration": self._calculate_concentration(assignee_counts),
        }

    def _calculate_concentration(self, assignee_counts: Counter) -> str:
        """
        Calculate market concentration based on patent holdings

        Args:
            assignee_counts: Counter of patents per assignee

        Returns:
            Concentration description
        """
        if not assignee_counts:
            return "Unknown"

        total = sum(assignee_counts.values())
        top_3_share = (
            sum(count for _, count in assignee_counts.most_common(3)) / total
            if total > 0
            else 0
        )

        if top_3_share > 0.6:
            return "High - Dominated by few players"
        elif top_3_share > 0.4:
            return "Moderate - Several key players"
        else:
            return "Low - Fragmented landscape"

    def _identify_white_space(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify white space opportunities

        Args:
            results: Analysis results

        Returns:
            List of white space opportunities
        """
        opportunities = []

        # Opportunity 1: Expiring patents
        if results["expiry_timeline"]["expiring_soon_count"] > 0:
            opportunities.append(
                {
                    "type": "Patent Expiry",
                    "description": f"{results['expiry_timeline']['expiring_soon_count']} patent(s) expiring within 2 years",
                    "opportunity": "Generic development or formulation improvements",
                    "priority": "High",
                    "patents": results["expiry_timeline"]["expiring_soon"][:3],
                }
            )

        # Opportunity 2: Low FTO risk areas
        if results["fto_assessment"]["risk_level"] == "Low":
            opportunities.append(
                {
                    "type": "Open Innovation Space",
                    "description": "Minimal patent barriers in this area",
                    "opportunity": "Novel formulations or delivery mechanisms",
                    "priority": "High",
                }
            )

        # Opportunity 3: Areas with low recent activity
        if (
            results["competitive_landscape"]["filing_trend"]["innovation_momentum"]
            == "Low"
        ):
            opportunities.append(
                {
                    "type": "Reduced Competition",
                    "description": "Low recent patent activity suggests reduced competitive focus",
                    "opportunity": "Market entry with differentiated product",
                    "priority": "Medium",
                }
            )

        return opportunities

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate executive summary of patent analysis

        Args:
            results: Analysis results dictionary

        Returns:
            Summary string
        """
        total = results["total_patents_found"]

        if total == 0:
            return "No patents found matching the search criteria."

        summary_parts = [f"Found {total} patent(s)."]

        if results["expiry_timeline"]:
            expiring_soon = results["expiry_timeline"]["expiring_soon_count"]
            if expiring_soon > 0:
                summary_parts.append(
                    f"{expiring_soon} patent(s) expiring within 2 years - window of opportunity."
                )

        if results["fto_assessment"]:
            risk = results["fto_assessment"]["risk_level"]
            summary_parts.append(f"FTO risk: {risk}.")

        if results["white_space_opportunities"]:
            summary_parts.append(
                f"{len(results['white_space_opportunities'])} white space opportunity(ies) identified."
            )

        return " ".join(summary_parts)

    def get_patent_details(self, patent_number: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific patent

        Args:
            patent_number: Patent number (e.g., US1234567B2)

        Returns:
            Detailed patent information
        """
        # Placeholder - would require individual patent lookup
        return {
            "patent_number": patent_number,
            "note": "Detailed patent information requires individual lookup",
            "url": f"https://patents.google.com/patent/{patent_number}",
        }

    def compare_patent_positions(self, drugs: List[str]) -> Dict[str, Any]:
        """
        Compare patent positions for multiple drugs

        Args:
            drugs: List of drug names

        Returns:
            Comparative patent analysis
        """
        if self.verbose:
            print(f"\n[Patent Agent] Comparing patent positions: {drugs}")

        comparisons = []

        for drug in drugs:
            result = self.search_patents(drugs=[drug], max_results=20)

            comparisons.append(
                {
                    "drug": drug,
                    "total_patents": result["total_patents_found"],
                    "expiring_soon": result["expiry_timeline"].get(
                        "expiring_soon_count", 0
                    ),
                    "fto_risk": result["fto_assessment"].get("risk_level", "Unknown"),
                    "opportunities": len(result["white_space_opportunities"]),
                }
            )

        # Sort by opportunity (expiring patents + white space)
        comparisons.sort(
            key=lambda x: x["expiring_soon"] + x["opportunities"], reverse=True
        )

        return {
            "drugs_compared": len(comparisons),
            "comparison_table": comparisons,
            "best_opportunity": comparisons[0]["drug"] if comparisons else None,
            "summary": self._generate_comparison_summary(comparisons),
        }

    def _generate_comparison_summary(self, comparisons: List[Dict]) -> str:
        """Generate summary of patent comparison"""
        if not comparisons:
            return "No drugs to compare"

        total_patents = sum(c["total_patents"] for c in comparisons)
        best = comparisons[0]["drug"]

        return (
            f"Compared {len(comparisons)} drugs with {total_patents} total patents. "
            f"Best opportunity: {best} with {comparisons[0]['expiring_soon']} expiring patents."
        )


# Convenience function
def get_patent_agent(verbose: bool = True) -> PatentAgent:
    """
    Get instance of Patent Agent

    Args:
        verbose: Whether to enable verbose logging

    Returns:
        Initialized PatentAgent instance
    """
    return PatentAgent(verbose=verbose)


# Test the agent
if __name__ == "__main__":
    print("=" * 70)
    print("PATENT AGENT - TEST SUITE")
    print("=" * 70)

    # Initialize agent
    agent = get_patent_agent(verbose=True)

    # Test 1: Search patents for drug
    print("\n" + "=" * 70)
    print("TEST 1: Patent Search - Ibuprofen")
    print("=" * 70)
    result = agent.search_patents(drugs=["Ibuprofen"], max_results=10)
    print(f"\nSummary: {result['summary']}")
    print(f"Total patents: {result['total_patents_found']}")
    print(f"\nExpiry Timeline: {json.dumps(result['expiry_timeline'], indent=2)}")
    print(f"\nFTO Assessment: {json.dumps(result['fto_assessment'], indent=2)}")

    # Test 2: White space opportunities
    print("\n" + "=" * 70)
    print("TEST 2: White Space Opportunities")
    print("=" * 70)
    if result["white_space_opportunities"]:
        print(json.dumps(result["white_space_opportunities"], indent=2))
    else:
        print("No white space opportunities identified")

    # Test 3: Compare patent positions
    print("\n" + "=" * 70)
    print("TEST 3: Patent Position Comparison")
    print("=" * 70)
    comparison = agent.compare_patent_positions(
        ["Metformin", "Atorvastatin", "Ibuprofen"]
    )
    print(json.dumps(comparison, indent=2))

    print("\n✓ All Patent Agent tests completed!")