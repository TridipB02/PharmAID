"""
EXIM Trade Agent - Import-Export Intelligence Specialist
Analyzes global trade patterns using real APIs (UN Comtrade, World Bank, USITC)
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_fetchers import get_mock_data_fetcher, get_real_exim_fetcher


class EXIMAgent:
    """
    EXIM Trade Intelligence Agent

    Uses Real APIs:
    - UN Comtrade (primary pharmaceutical trade data)
    - World Bank (aggregate trade indicators)
    - USITC DataWeb (US-specific data)
    - Mock data fallback if APIs unavailable
    """

    def __init__(self, verbose: bool = True):
        """Initialize EXIM Agent with real API sources"""
        self.verbose = verbose

        # Use real API fetcher
        self.data_fetcher = get_real_exim_fetcher(verbose=verbose)

        # Keep mock fetcher for drug metadata
        self.mock_fetcher = get_mock_data_fetcher()

        if self.verbose:
            print(" EXIM Agent initialized with real API sources")
            print("  - UN Comtrade API")
            print("  - World Bank API")
            print("  - USITC DataWeb API")
            print("  - Mock data fallback")

    def analyze_trade(
        self,
        drugs: List[str] = None,
        countries: List[str] = None,
        direction: str = "both",
    ) -> Dict[str, Any]:
        """
        Comprehensive trade analysis using real APIs

        Args:
            drugs: List of drug names to analyze
            countries: List of countries to focus on
            direction: Trade direction ("export", "import", or "both")

        Returns:
            Dictionary with trade analysis results
        """
        if self.verbose:
            print(f"\n[EXIM Agent] Analyzing trade data...")
            print(f"  Drugs: {drugs}")
            print(f"  Countries: {countries}")
            print(f"  Direction: {direction}")

        results = {
            "summary": "",
            "drug_analyses": [],
            "country_analyses": [],
            "trade_flows": {},
            "sourcing_insights": [],
            "price_analysis": {},
            "supply_chain_risks": [],
            "data_sources_used": [],
        }

        # Analyze individual drugs
        if drugs:
            for drug_name in drugs:
                drug_analysis = self._analyze_drug_trade_with_apis(
                    drug_name, countries, direction
                )
                if drug_analysis:
                    results["drug_analyses"].append(drug_analysis)
                    # Track data sources
                    if drug_analysis.get("data_sources"):
                        results["data_sources_used"].extend(
                            drug_analysis["data_sources"]
                        )

        # Remove duplicate data sources
        results["data_sources_used"] = list(set(results["data_sources_used"]))

        # Generate insights
        results["trade_flows"] = self._analyze_trade_flows(results["drug_analyses"])
        results["sourcing_insights"] = self._extract_sourcing_insights(results)
        results["supply_chain_risks"] = self._assess_supply_chain_risks(results)

        # Generate summary
        results["summary"] = self._generate_summary(results)

        if self.verbose:
            print(
                f" Analysis complete - {len(results['drug_analyses'])} drugs analyzed"
            )
            print(f"  Data sources used: {', '.join(results['data_sources_used'])}")

        return results

    def _analyze_drug_trade_with_apis(
        self, drug_name: str, countries: List[str] = None, direction: str = "both"
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze trade data for a single drug using real APIs ONLY

        """
        # Get drug info for metadata (from local database)
        drug_info = self.mock_fetcher.get_drug_info(drug_name)
        therapeutic_area = (
            drug_info.get("therapeutic_area", "Unknown") if drug_info else "Unknown"
        )

        # Get comprehensive trade data from real APIs
        country_code = countries[0] if countries else None
        trade_data = self.data_fetcher.get_comprehensive_trade_data(
            drug_name=drug_name, country=country_code
        )

        if not trade_data or not trade_data.get("data_sources"):
            if self.verbose:
                print(f"  ⚠ No trade data available for: {drug_name}")
            return None

        else:
            # Process real API data
            return self._process_real_api_data(
                drug_name, therapeutic_area, trade_data, countries, direction
            )

    def _process_real_api_data(
        self,
        drug_name: str,
        therapeutic_area: str,
        trade_data: Dict,
        countries: List[str] = None,
        direction: str = "both",
    ) -> Dict[str, Any]:
        """Process real API data from UN Comtrade, World Bank, USITC"""

        # Extract data from different sources
        comtrade_data = trade_data.get("un_comtrade_data", [])
        world_bank_data = trade_data.get("world_bank_data", [])

        if not comtrade_data and not world_bank_data:
            return {
                "drug_name": drug_name,
                "therapeutic_area": therapeutic_area,
                "trade_metrics": {
                    "total_import_value_usd": 0,
                    "total_export_value_usd": 0,
                    "trade_balance": "No Data",
                    "number_of_trade_records": 0,
                    "data_year": "N/A",
                },
                "top_trading_partners": [],
                "data_sources": trade_data.get("data_sources", []),
                "raw_data_summary": {
                    "un_comtrade_records": 0,
                    "world_bank_records": 0,
                    "usitc_records": 0,
                },
                "world_bank_indicators": [],
                "trade_records": [],
                "note": "No trade data available from any API source",
            }

        # Process UN Comtrade data (most detailed)
        total_import_value = 0
        total_export_value = 0
        trade_records = []

        for record in comtrade_data:
            flow = record.get("flowDesc", "Unknown")
            partner = record.get("partnerDesc", "Unknown")
            value = record.get("primaryValue", 0)

            if "Import" in flow:
                total_import_value += value
            elif "Export" in flow:
                total_export_value += value

            trade_records.append(
                {
                    "partner": partner,
                    "flow": flow,
                    "value_usd": value,
                    "year": record.get("period", "N/A"),
                }
            )

        # Process World Bank data (aggregate indicators)
        wb_summary = []
        for record in world_bank_data:
            wb_summary.append(
                {
                    "country": record.get("country", {}).get("value", "N/A"),
                    "year": record.get("date", "N/A"),
                    "value": record.get("value", 0),
                    "indicator": record.get("indicator", {}).get("value", "N/A"),
                }
            )

        # Determine trade balance
        trade_balance = (
            "Net Exporter"
            if total_export_value > total_import_value
            else (
                "Net Importer"
                if total_import_value > total_export_value
                else "Balanced"
            )
        )

        # Get top trading partners from Comtrade
        partner_values = defaultdict(float)
        for record in trade_records:
            partner_values[record["partner"]] += record["value_usd"]

        top_partners = sorted(partner_values.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        return {
            "drug_name": drug_name,
            "therapeutic_area": therapeutic_area,
            "trade_metrics": {
                "total_import_value_usd": round(total_import_value, 2),
                "total_export_value_usd": round(total_export_value, 2),
                "trade_balance": trade_balance,
                "number_of_trade_records": len(trade_records),
                "data_year": "2022",
            },
            "top_trading_partners": [
                {"country": country, "total_value_usd": round(value, 2)}
                for country, value in top_partners
            ],
            "data_sources": trade_data.get("data_sources", []),
            "raw_data_summary": {
                "un_comtrade_records": len(comtrade_data),
                "world_bank_records": len(world_bank_data),
            },
            "world_bank_indicators": wb_summary[:5] if wb_summary else [],
            "trade_records": trade_records[:10],  # Show top 10
        }

    def _analyze_trade_flows(self, drug_analyses: List[Dict]) -> Dict[str, Any]:
        """Analyze overall trade flows"""
        if not drug_analyses:
            return {}

        # Aggregate imports/exports
        total_imports = 0
        total_exports = 0

        for analysis in drug_analyses:
            metrics = analysis.get("trade_metrics", {})
            total_imports += metrics.get("total_import_value_usd", 0)
            total_exports += metrics.get("total_export_value_usd", 0)

        # Collect trading partners
        all_partners = []
        for analysis in drug_analyses:
            partners = analysis.get("top_trading_partners", [])
            all_partners.extend([p["country"] for p in partners])

        partner_counts = Counter(all_partners)

        return {
            "total_import_value_usd": round(total_imports, 2),
            "total_export_value_usd": round(total_exports, 2),
            "net_trade_position": (
                "Net Exporter" if total_exports > total_imports else "Net Importer"
            ),
            "most_active_partners": [
                {"country": country, "frequency": count}
                for country, count in partner_counts.most_common(10)
            ],
        }

    def _extract_sourcing_insights(self, results: Dict[str, Any]) -> List[str]:
        """Extract sourcing insights"""
        insights = []

        for drug_analysis in results["drug_analyses"]:
            drug_name = drug_analysis["drug_name"]
            balance = drug_analysis["trade_metrics"].get("trade_balance", "Unknown")

            if balance == "Net Importer":
                insights.append(
                    f"{drug_name}: Import dependency detected - consider local sourcing"
                )

            # Check data source
            sources = drug_analysis.get("data_sources", [])
            if "Mock Data (Fallback)" in sources:
                insights.append(
                    f"{drug_name}: Using fallback data - real APIs unavailable"
                )

        return insights

    def _assess_supply_chain_risks(
        self, results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Assess supply chain risks"""
        risks = []

        for drug_analysis in results["drug_analyses"]:
            drug_name = drug_analysis["drug_name"]
            balance = drug_analysis["trade_metrics"].get("trade_balance", "Unknown")

            if balance == "Net Importer":
                risks.append(
                    {
                        "drug": drug_name,
                        "risk_type": "Import Dependency",
                        "severity": "Medium",
                        "description": "Reliant on imports",
                        "mitigation": "Develop local manufacturing",
                    }
                )

            # Check if limited data
            data_sources = drug_analysis.get("data_sources", [])
            if "Mock Data (Fallback)" in data_sources:
                risks.append(
                    {
                        "drug": drug_name,
                        "risk_type": "Data Availability",
                        "severity": "Low",
                        "description": "Real-time trade data unavailable",
                        "mitigation": "Verify with customs authorities",
                    }
                )

        return risks

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary"""
        summary_parts = []

        num_drugs = len(results["drug_analyses"])
        if num_drugs > 0:
            summary_parts.append(f"Analyzed trade patterns for {num_drugs} drug(s).")

        if results["data_sources_used"]:
            sources_str = ", ".join(results["data_sources_used"])
            summary_parts.append(f"Data from: {sources_str}.")

        if results["trade_flows"]:
            position = results["trade_flows"].get("net_trade_position", "Unknown")
            summary_parts.append(f"Overall position: {position}.")

        if results["supply_chain_risks"]:
            high_risks = sum(
                1 for r in results["supply_chain_risks"] if r["severity"] == "High"
            )
            if high_risks > 0:
                summary_parts.append(f"⚠ {high_risks} high-severity risk(s) detected.")

        return " ".join(summary_parts) if summary_parts else "No trade data available."

    def compare_trade_positions(self, drugs: List[str]) -> Dict[str, Any]:
        """Compare trade positions for multiple drugs"""
        if self.verbose:
            print(f"\n[EXIM Agent] Comparing trade positions: {drugs}")

        comparisons = []

        for drug in drugs:
            analysis = self._analyze_drug_trade_with_apis(drug, None, "both")
            if analysis:
                metrics = analysis.get("trade_metrics", {})
                comparisons.append(
                    {
                        "drug": drug,
                        "total_import_value_usd": metrics.get(
                            "total_import_value_usd", 0
                        ),
                        "total_export_value_usd": metrics.get(
                            "total_export_value_usd", 0
                        ),
                        "trade_balance": metrics.get("trade_balance", "Unknown"),
                        "data_sources": ", ".join(analysis.get("data_sources", [])),
                    }
                )

        # Sort by total trade value
        comparisons.sort(
            key=lambda x: x["total_import_value_usd"] + x["total_export_value_usd"],
            reverse=True,
        )

        return {
            "drugs_compared": len(comparisons),
            "comparison_table": comparisons,
            "highest_trade_volume": comparisons[0]["drug"] if comparisons else None,
        }


def get_exim_agent(verbose: bool = True) -> EXIMAgent:
    """Get instance of EXIM Agent with real API support"""
    return EXIMAgent(verbose=verbose)
