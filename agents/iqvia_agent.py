"""
IQVIA Insights Agent - Market Intelligence Specialist
Analyzes market trends, sales data, competition, and therapy area dynamics
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_fetchers import MockDataFetcher


class IQVIAAgent:
    """
    IQVIA Market Intelligence Agent

    Responsibilities:
    - Analyze market size, growth rates (CAGR), and trends
    - Identify competitive landscape and market share
    - Track therapy area dynamics and opportunities
    - Provide sales forecasts and prescription volume insights
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize IQVIA Agent

        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.data_fetcher = MockDataFetcher()

        if self.verbose:
            print(" IQVIA Agent initialized")

    def analyze_market(
        self,
        drugs: List[str] = None,
        therapeutic_areas: List[str] = None,
        time_scope: str = "recent",
    ) -> Dict[str, Any]:
        """
        Comprehensive market analysis for drugs or therapeutic areas

        Args:
            drugs: List of drug names to analyze
            therapeutic_areas: List of therapeutic areas to analyze
            time_scope: Time period ('recent', 'historical', 'all_time')

        Returns:
            Dictionary with market analysis results
        """
        if self.verbose:
            print(f"\n[IQVIA Agent] Analyzing market data...")
            print(f"  Drugs: {drugs}")
            print(f"  Therapeutic Areas: {therapeutic_areas}")
            print(f"  Time Scope: {time_scope}")

        results = {
            "summary": "",
            "drug_analyses": [],
            "therapeutic_area_analyses": [],
            "market_insights": [],
            "competitive_landscape": {},
        }

        # Analyze individual drugs
        if drugs:
            for drug_name in drugs:
                drug_analysis = self._analyze_single_drug(drug_name, time_scope)
                if drug_analysis:
                    results["drug_analyses"].append(drug_analysis)

        # Analyze therapeutic areas
        if therapeutic_areas:
            for area in therapeutic_areas:
                area_analysis = self._analyze_therapeutic_area(area, time_scope)
                if area_analysis:
                    results["therapeutic_area_analyses"].append(area_analysis)

        # Generate summary insights
        results["summary"] = self._generate_summary(results)
        results["market_insights"] = self._extract_market_insights(results)
        results["competitive_landscape"] = self._analyze_competition(
            drugs, therapeutic_areas
        )

        if self.verbose:
            print(
                f" Analysis complete - {len(results['drug_analyses'])} drugs, {len(results['therapeutic_area_analyses'])} therapeutic areas"
            )

        return results

    def _analyze_single_drug(
        self, drug_name: str, time_scope: str
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze market data for a single drug

        Args:
            drug_name: Name of the drug
            time_scope: Time period for analysis

        Returns:
            Dictionary with drug analysis or None if not found
        """
        # Get drug information
        drug_info = self.data_fetcher.get_drug_info(drug_name)
        if not drug_info:
            if self.verbose:
                print(f"  No data found for drug: {drug_name}")
            return None

        # Get market data
        market_data = self.data_fetcher.get_iqvia_market_data(drug_name)
        if not market_data:
            if self.verbose:
                print(f"  No IQVIA market data for: {drug_name}")
            return None

        # Calculate CAGR
        cagr = self.data_fetcher.calculate_cagr(drug_name)

        # Get latest data point
        data_points = market_data.get("data_points", [])
        if not data_points:
            return None

        latest_data = sorted(data_points, key=lambda x: x["year"], reverse=True)[0]
        oldest_data = sorted(data_points, key=lambda x: x["year"])[0]

        # Calculate growth metrics
        sales_growth = (
            latest_data["sales_usd_million"] - oldest_data["sales_usd_million"]
        )
        sales_growth_pct = (
            (sales_growth / oldest_data["sales_usd_million"]) * 100
            if oldest_data["sales_usd_million"] > 0
            else 0
        )

        analysis = {
            "drug_name": drug_name,
            "generic_name": drug_info.get("generic_name", drug_name),
            "therapeutic_area": drug_info.get("therapeutic_area", "Unknown"),
            "brand_names": drug_info.get("brand_names", []),
            "indication": drug_info.get("indication", "N/A"),
            "mechanism": drug_info.get("mechanism_of_action", "N/A"),
            "market_metrics": {
                "latest_year": latest_data["year"],
                "current_sales_usd_million": latest_data["sales_usd_million"],
                "current_prescriptions_million": latest_data["prescriptions_million"],
                "market_trend": latest_data["market_trend"],
                "cagr_percent": cagr,
                "total_growth_usd_million": round(sales_growth, 2),
                "growth_percentage": round(sales_growth_pct, 2),
                "years_analyzed": len(data_points),
            },
            "trend_analysis": self._analyze_trend(data_points),
            "historical_data": data_points,
        }

        return analysis

    def _analyze_therapeutic_area(
        self, therapeutic_area: str, time_scope: str
    ) -> Dict[str, Any]:
        """
        Analyze entire therapeutic area market dynamics

        Args:
            therapeutic_area: Name of therapeutic area
            time_scope: Time period for analysis

        Returns:
            Dictionary with therapeutic area analysis
        """
        # Get all drugs in this therapeutic area
        drugs_in_area = self.data_fetcher.get_drugs_by_therapeutic_area(
            therapeutic_area
        )

        if not drugs_in_area:
            if self.verbose:
                print(f"  No drugs found in therapeutic area: {therapeutic_area}")
            return {
                "therapeutic_area": therapeutic_area,
                "total_drugs": 0,
                "error": "No drugs found in this therapeutic area",
            }

        # Aggregate market data
        total_sales = 0
        total_prescriptions = 0
        drug_summaries = []

        for drug in drugs_in_area:
            drug_name = drug["name"]
            market_data = self.data_fetcher.get_iqvia_market_data(drug_name)

            if market_data and "data_points" in market_data:
                data_points = market_data["data_points"]
                if data_points:
                    latest = sorted(data_points, key=lambda x: x["year"], reverse=True)[
                        0
                    ]
                    total_sales += latest["sales_usd_million"]
                    total_prescriptions += latest["prescriptions_million"]

                    drug_summaries.append(
                        {
                            "drug_name": drug_name,
                            "sales_usd_million": latest["sales_usd_million"],
                            "market_share_percent": 0,  # Will calculate later
                            "trend": latest["market_trend"],
                        }
                    )

        # Calculate market shares
        for drug_summary in drug_summaries:
            drug_summary["market_share_percent"] = (
                round((drug_summary["sales_usd_million"] / total_sales * 100), 2)
                if total_sales > 0
                else 0
            )

        # Sort by market share
        drug_summaries.sort(key=lambda x: x["market_share_percent"], reverse=True)

        analysis = {
            "therapeutic_area": therapeutic_area,
            "total_drugs": len(drugs_in_area),
            "market_size_usd_million": round(total_sales, 2),
            "total_prescriptions_million": round(total_prescriptions, 2),
            "average_drug_sales_usd_million": (
                round(total_sales / len(drugs_in_area), 2) if drugs_in_area else 0
            ),
            "top_drugs": drug_summaries[:5],  # Top 5 drugs
            "market_leaders": [d["drug_name"] for d in drug_summaries[:3]],
            "competitive_intensity": self._calculate_competitive_intensity(
                drug_summaries
            ),
        }

        return analysis

    def _analyze_trend(self, data_points: List[Dict]) -> str:
        """
        Analyze trend direction from data points

        Args:
            data_points: List of historical data points

        Returns:
            Trend description string
        """
        if len(data_points) < 2:
            return "Insufficient data for trend analysis"

        sorted_points = sorted(data_points, key=lambda x: x["year"])

        # Check recent trend (last 2 years)
        if len(sorted_points) >= 2:
            recent_growth = (
                sorted_points[-1]["sales_usd_million"]
                - sorted_points[-2]["sales_usd_million"]
            )
            if recent_growth > 0:
                return "Growing - Positive momentum in recent period"
            elif recent_growth < 0:
                return "Declining - Negative momentum in recent period"
            else:
                return "Stable - No significant change in recent period"

        return "Trend unclear"

    def _calculate_competitive_intensity(self, drug_summaries: List[Dict]) -> str:
        """
        Calculate competitive intensity based on market concentration

        Args:
            drug_summaries: List of drug market summaries

        Returns:
            Competitive intensity description
        """
        if not drug_summaries:
            return "Unknown"

        # Calculate HHI (Herfindahl-Hirschman Index)
        hhi = sum(d["market_share_percent"] ** 2 for d in drug_summaries)

        if hhi > 2500:
            return "High Concentration - Dominated by few players"
        elif hhi > 1500:
            return "Moderate Concentration - Competitive but with leaders"
        else:
            return "Low Concentration - Highly fragmented market"

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate executive summary of market analysis

        Args:
            results: Analysis results dictionary

        Returns:
            Summary string
        """
        summary_parts = []

        if results["drug_analyses"]:
            num_drugs = len(results["drug_analyses"])
            total_sales = sum(
                d["market_metrics"]["current_sales_usd_million"]
                for d in results["drug_analyses"]
            )
            avg_cagr = (
                sum(
                    d["market_metrics"].get("cagr_percent", 0)
                    for d in results["drug_analyses"]
                )
                / num_drugs
                if num_drugs > 0
                else 0
            )

            summary_parts.append(
                f"Analyzed {num_drugs} drugs with combined sales of ${total_sales:.1f}M. "
                f"Average CAGR: {avg_cagr:.2f}%."
            )

        if results["therapeutic_area_analyses"]:
            num_areas = len(results["therapeutic_area_analyses"])
            summary_parts.append(f"Examined {num_areas} therapeutic area(s).")

        return (
            " ".join(summary_parts)
            if summary_parts
            else "No market data available for analysis."
        )

    def _extract_market_insights(self, results: Dict[str, Any]) -> List[str]:
        """
        Extract key market insights from analysis

        Args:
            results: Analysis results dictionary

        Returns:
            List of insight strings
        """
        insights = []

        # Identify high-growth drugs
        high_growth_drugs = [
            d
            for d in results["drug_analyses"]
            if d["market_metrics"].get("cagr_percent", 0) > 5
        ]

        if high_growth_drugs:
            drug_names = ", ".join([d["drug_name"] for d in high_growth_drugs[:3]])
            insights.append(f"High-growth opportunities identified in: {drug_names}")

        # Identify declining drugs
        declining_drugs = [
            d
            for d in results["drug_analyses"]
            if d["market_metrics"]["market_trend"] == "decreasing"
        ]

        if declining_drugs:
            insights.append(
                f"{len(declining_drugs)} drug(s) showing declining trends - potential repurposing candidates"
            )

        # Market size insights
        large_markets = [
            d
            for d in results["drug_analyses"]
            if d["market_metrics"]["current_sales_usd_million"] > 200
        ]

        if large_markets:
            insights.append(
                f"{len(large_markets)} blockbuster drug(s) with sales >$200M annually"
            )

        return insights

    def _analyze_competition(
        self, drugs: List[str] = None, therapeutic_areas: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze competitive landscape

        Args:
            drugs: List of drug names
            therapeutic_areas: List of therapeutic areas

        Returns:
            Competitive analysis dictionary
        """
        competition = {
            "total_competitors": 0,
            "market_leaders": [],
            "competitive_threats": [],
            "white_space_opportunities": [],
        }

        if therapeutic_areas:
            for area in therapeutic_areas:
                drugs_in_area = self.data_fetcher.get_drugs_by_therapeutic_area(area)
                competition["total_competitors"] += len(drugs_in_area)

                # Identify leaders (simplified)
                if len(drugs_in_area) >= 3:
                    competition["market_leaders"].extend(
                        [d["name"] for d in drugs_in_area[:3]]
                    )

        return competition

    def get_market_trends(self, drug_name: str, years: int = 5) -> Dict[str, Any]:
        """
        Get historical market trends for a specific drug

        Args:
            drug_name: Name of the drug
            years: Number of years to analyze

        Returns:
            Trend data dictionary
        """
        market_data = self.data_fetcher.get_iqvia_market_data(drug_name)

        if not market_data or "data_points" not in market_data:
            return {"error": f"No market data found for {drug_name}"}

        data_points = sorted(
            market_data["data_points"], key=lambda x: x["year"], reverse=True
        )[:years]

        return {
            "drug_name": drug_name,
            "years_analyzed": len(data_points),
            "trend_data": data_points,
            "cagr": self.data_fetcher.calculate_cagr(drug_name),
        }

    def compare_drugs(self, drug_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple drugs side-by-side

        Args:
            drug_names: List of drug names to compare

        Returns:
            Comparison dictionary
        """
        if self.verbose:
            print(f"\n[IQVIA Agent] Comparing drugs: {drug_names}")

        comparisons = []

        for drug_name in drug_names:
            analysis = self._analyze_single_drug(drug_name, "recent")
            if analysis:
                comparisons.append(
                    {
                        "drug_name": drug_name,
                        "sales_usd_million": analysis["market_metrics"][
                            "current_sales_usd_million"
                        ],
                        "cagr_percent": analysis["market_metrics"]["cagr_percent"],
                        "trend": analysis["market_metrics"]["market_trend"],
                        "therapeutic_area": analysis["therapeutic_area"],
                    }
                )

        # Sort by sales
        comparisons.sort(key=lambda x: x["sales_usd_million"], reverse=True)

        return {
            "drugs_compared": len(comparisons),
            "comparison_table": comparisons,
            "leader": comparisons[0]["drug_name"] if comparisons else None,
            "highest_growth": (
                max(comparisons, key=lambda x: x.get("cagr_percent", 0))["drug_name"]
                if comparisons
                else None
            ),
        }


# Convenience function
def get_iqvia_agent(verbose: bool = True) -> IQVIAAgent:
    """
    Get instance of IQVIA Agent

    Args:
        verbose: Whether to enable verbose logging

    Returns:
        Initialized IQVIAAgent instance
    """
    return IQVIAAgent(verbose=verbose)


# Test the agent
if __name__ == "__main__":
    print("=" * 70)
    print("IQVIA AGENT - TEST SUITE")
    print("=" * 70)

    # Initialize agent
    agent = get_iqvia_agent(verbose=True)

    # Test 1: Single drug analysis
    print("\n" + "=" * 70)
    print("TEST 1: Single Drug Analysis - Metformin")
    print("=" * 70)
    result = agent.analyze_market(drugs=["Metformin"])
    print(json.dumps(result, indent=2))

    # Test 2: Therapeutic area analysis
    print("\n" + "=" * 70)
    print("TEST 2: Therapeutic Area Analysis - Diabetes")
    print("=" * 70)
    result = agent.analyze_market(therapeutic_areas=["Diabetes"])
    print(json.dumps(result, indent=2))

    # Test 3: Compare drugs
    print("\n" + "=" * 70)
    print("TEST 3: Drug Comparison")
    print("=" * 70)
    comparison = agent.compare_drugs(["Metformin", "Insulin Glargine", "Sitagliptin"])
    print(json.dumps(comparison, indent=2))

    # Test 4: Market trends
    print("\n" + "=" * 70)
    print("TEST 4: Market Trends - Atorvastatin")
    print("=" * 70)
    trends = agent.get_market_trends("Atorvastatin", years=5)
    print(json.dumps(trends, indent=2))

    print("\n All IQVIA Agent tests completed!")
