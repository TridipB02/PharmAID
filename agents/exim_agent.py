"""
EXIM Trade Agent - Import-Export Intelligence Specialist
Analyzes global trade patterns, supply chain dynamics, and sourcing insights
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from collections import Counter, defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_fetchers import MockDataFetcher


class EXIMAgent:
    """
    EXIM Trade Intelligence Agent
    
    Responsibilities:
    - Analyze import/export volumes and trends
    - Track country-wise trade patterns
    - Identify sourcing opportunities and risks
    - Monitor price dynamics across regions
    - Assess supply chain dependencies
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize EXIM Agent
        
        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.data_fetcher = MockDataFetcher()
        
        if self.verbose:
            print("✓ EXIM Agent initialized")
    
    def analyze_trade(
        self,
        drugs: List[str] = None,
        countries: List[str] = None,
        direction: str = "both"  # "export", "import", or "both"
    ) -> Dict[str, Any]:
        """
        Comprehensive trade analysis for drugs and countries
        
        Args:
            drugs: List of drug names to analyze
            countries: List of countries to focus on
            direction: Trade direction to analyze
        
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
            "supply_chain_risks": []
        }
        
        # Analyze individual drugs
        if drugs:
            for drug_name in drugs:
                drug_analysis = self._analyze_drug_trade(drug_name, countries, direction)
                if drug_analysis:
                    results["drug_analyses"].append(drug_analysis)
        
        # Analyze by country
        if countries:
            for country in countries:
                country_analysis = self._analyze_country_trade(country, drugs)
                if country_analysis:
                    results["country_analyses"].append(country_analysis)
        
        # Generate insights
        results["trade_flows"] = self._analyze_trade_flows(results["drug_analyses"])
        results["sourcing_insights"] = self._extract_sourcing_insights(results)
        results["price_analysis"] = self._analyze_price_trends(results["drug_analyses"])
        results["supply_chain_risks"] = self._assess_supply_chain_risks(results)
        
        # Generate summary
        results["summary"] = self._generate_summary(results)
        
        if self.verbose:
            print(f"✓ Analysis complete - {len(results['drug_analyses'])} drugs analyzed")
        
        return results
    
    def _analyze_drug_trade(
        self, 
        drug_name: str, 
        countries: List[str] = None,
        direction: str = "both"
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze trade data for a single drug
        
        Args:
            drug_name: Name of the drug
            countries: List of countries to focus on (None = all)
            direction: Trade direction
        
        Returns:
            Dictionary with drug trade analysis or None
        """
        # Get drug info
        drug_info = self.data_fetcher.get_drug_info(drug_name)
        if not drug_info:
            if self.verbose:
                print(f"  ⚠ No data found for drug: {drug_name}")
            return None
        
        # Get trade data
        trade_data = self.data_fetcher.get_exim_trade_data(drug_name)
        if not trade_data:
            if self.verbose:
                print(f"  ⚠ No EXIM trade data for: {drug_name}")
            return None
        
        # Filter by direction and countries
        all_trades = trade_data.get('trade_data', [])
        if not all_trades:
            return None
        
        filtered_trades = []
        for trade in all_trades:
            # Filter by direction
            if direction != "both" and trade['direction'] != direction:
                continue
            # Filter by countries
            if countries and trade['country'].lower() not in [c.lower() for c in countries]:
                continue
            filtered_trades.append(trade)
        
        if not filtered_trades:
            return None
        
        # Calculate metrics
        total_volume = sum(t['quantity_kg'] for t in filtered_trades)
        total_value = sum(t['total_value_usd'] for t in filtered_trades)
        avg_price = total_value / total_volume if total_volume > 0 else 0
        
        # Analyze by country
        country_breakdown = defaultdict(lambda: {"volume": 0, "value": 0, "trades": []})
        for trade in filtered_trades:
            country = trade['country']
            country_breakdown[country]["volume"] += trade['quantity_kg']
            country_breakdown[country]["value"] += trade['total_value_usd']
            country_breakdown[country]["trades"].append(trade)
        
        # Top countries
        top_countries = sorted(
            country_breakdown.items(),
            key=lambda x: x[1]["volume"],
            reverse=True
        )[:5]
        
        # Analyze exports vs imports
        exports = [t for t in filtered_trades if t['direction'] == 'export']
        imports = [t for t in filtered_trades if t['direction'] == 'import']
        
        export_volume = sum(t['quantity_kg'] for t in exports)
        import_volume = sum(t['quantity_kg'] for t in imports)
        
        trade_balance = "Net Exporter" if export_volume > import_volume else "Net Importer" if import_volume > export_volume else "Balanced"
        
        analysis = {
            "drug_name": drug_name,
            "therapeutic_area": drug_info.get('therapeutic_area', 'Unknown'),
            "trade_metrics": {
                "total_volume_kg": round(total_volume, 2),
                "total_value_usd": round(total_value, 2),
                "average_price_per_kg": round(avg_price, 2),
                "number_of_trades": len(filtered_trades),
                "export_volume_kg": round(export_volume, 2),
                "import_volume_kg": round(import_volume, 2),
                "trade_balance": trade_balance
            },
            "top_trading_partners": [
                {
                    "country": country,
                    "volume_kg": round(data["volume"], 2),
                    "value_usd": round(data["value"], 2),
                    "average_price": round(data["value"] / data["volume"], 2) if data["volume"] > 0 else 0,
                    "trade_count": len(data["trades"])
                }
                for country, data in top_countries
            ],
            "price_range": self._calculate_price_range(filtered_trades),
            "trade_data": filtered_trades
        }
        
        return analysis
    
    def _analyze_country_trade(
        self, 
        country: str, 
        drugs: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze trade patterns for a specific country
        
        Args:
            country: Country name
            drugs: List of drugs to focus on (None = all available)
        
        Returns:
            Dictionary with country trade analysis
        """
        # This would require aggregating across all drugs
        # Simplified implementation for now
        return {
            "country": country,
            "note": "Country-specific aggregation requires full drug scan",
            "drugs_analyzed": drugs if drugs else []
        }
    
    def _calculate_price_range(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Calculate price range from trade data
        
        Args:
            trades: List of trade dictionaries
        
        Returns:
            Price range statistics
        """
        prices = [t['unit_price_usd'] for t in trades]
        
        if not prices:
            return {"min": 0, "max": 0, "average": 0}
        
        return {
            "min_price_usd": round(min(prices), 2),
            "max_price_usd": round(max(prices), 2),
            "average_price_usd": round(sum(prices) / len(prices), 2),
            "price_variance": round(max(prices) - min(prices), 2)
        }
    
    def _analyze_trade_flows(self, drug_analyses: List[Dict]) -> Dict[str, Any]:
        """
        Analyze overall trade flows
        
        Args:
            drug_analyses: List of drug analysis dictionaries
        
        Returns:
            Trade flow analysis
        """
        if not drug_analyses:
            return {}
        
        # Aggregate by direction
        total_exports = sum(
            d['trade_metrics']['export_volume_kg'] 
            for d in drug_analyses
        )
        total_imports = sum(
            d['trade_metrics']['import_volume_kg'] 
            for d in drug_analyses
        )
        
        # Collect all trading partners
        all_partners = []
        for analysis in drug_analyses:
            all_partners.extend([
                partner['country'] 
                for partner in analysis['top_trading_partners']
            ])
        
        partner_counts = Counter(all_partners)
        
        return {
            "total_export_volume_kg": round(total_exports, 2),
            "total_import_volume_kg": round(total_imports, 2),
            "net_trade_position": "Net Exporter" if total_exports > total_imports else "Net Importer",
            "most_active_partners": [
                {"country": country, "trade_frequency": count}
                for country, count in partner_counts.most_common(10)
            ],
            "trade_concentration": self._calculate_trade_concentration(partner_counts)
        }
    
    def _calculate_trade_concentration(self, partner_counts: Counter) -> str:
        """
        Calculate trade concentration
        
        Args:
            partner_counts: Counter of trade partners
        
        Returns:
            Concentration description
        """
        if not partner_counts:
            return "Unknown"
        
        total = sum(partner_counts.values())
        top_3_share = sum(count for _, count in partner_counts.most_common(3)) / total if total > 0 else 0
        
        if top_3_share > 0.7:
            return "High - Trade concentrated with few partners"
        elif top_3_share > 0.4:
            return "Moderate - Diversified across key partners"
        else:
            return "Low - Highly diversified trade network"
    
    def _extract_sourcing_insights(self, results: Dict[str, Any]) -> List[str]:
        """
        Extract sourcing insights from analysis
        
        Args:
            results: Analysis results
        
        Returns:
            List of insight strings
        """
        insights = []
        
        for drug_analysis in results['drug_analyses']:
            drug_name = drug_analysis['drug_name']
            
            # Check trade balance
            balance = drug_analysis['trade_metrics']['trade_balance']
            if balance == "Net Importer":
                insights.append(
                    f"{drug_name}: Strong import dependency - consider local manufacturing"
                )
            
            # Check price variance
            price_range = drug_analysis.get('price_range', {})
            variance = price_range.get('price_variance', 0)
            if variance > 500:  # Significant price difference
                insights.append(
                    f"{drug_name}: High price variance (${variance}/kg) across markets - arbitrage opportunity"
                )
            
            # Check concentration
            if len(drug_analysis['top_trading_partners']) <= 2:
                insights.append(
                    f"{drug_name}: Trade concentrated with ≤2 partners - supply chain risk"
                )
        
        return insights
    
    def _analyze_price_trends(self, drug_analyses: List[Dict]) -> Dict[str, Any]:
        """
        Analyze price trends across drugs
        
        Args:
            drug_analyses: List of drug analyses
        
        Returns:
            Price analysis
        """
        if not drug_analyses:
            return {}
        
        price_data = []
        for analysis in drug_analyses:
            price_data.append({
                "drug": analysis['drug_name'],
                "average_price": analysis['trade_metrics']['average_price_per_kg'],
                "price_range": analysis.get('price_range', {})
            })
        
        # Sort by price
        price_data.sort(key=lambda x: x['average_price'], reverse=True)
        
        return {
            "drugs_analyzed": len(price_data),
            "highest_priced": price_data[0] if price_data else None,
            "lowest_priced": price_data[-1] if price_data else None,
            "average_across_all": round(
                sum(d['average_price'] for d in price_data) / len(price_data), 2
            ) if price_data else 0,
            "price_data": price_data
        }
    
    def _assess_supply_chain_risks(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Assess supply chain risks
        
        Args:
            results: Analysis results
        
        Returns:
            List of risk assessments
        """
        risks = []
        
        for drug_analysis in results['drug_analyses']:
            drug_name = drug_analysis['drug_name']
            
            # Risk 1: Import dependency
            balance = drug_analysis['trade_metrics']['trade_balance']
            import_vol = drug_analysis['trade_metrics']['import_volume_kg']
            
            if balance == "Net Importer" and import_vol > 1000:
                risks.append({
                    "drug": drug_name,
                    "risk_type": "Import Dependency",
                    "severity": "High" if import_vol > 5000 else "Medium",
                    "description": f"Heavily reliant on imports ({import_vol}kg)",
                    "mitigation": "Develop local manufacturing capacity"
                })
            
            # Risk 2: Limited supplier base
            partner_count = len(drug_analysis['top_trading_partners'])
            if partner_count <= 2:
                risks.append({
                    "drug": drug_name,
                    "risk_type": "Supplier Concentration",
                    "severity": "High",
                    "description": f"Only {partner_count} trading partner(s)",
                    "mitigation": "Diversify supplier base"
                })
            
            # Risk 3: Price volatility
            price_range = drug_analysis.get('price_range', {})
            variance = price_range.get('price_variance', 0)
            avg_price = price_range.get('average_price_usd', 1)
            
            if avg_price > 0 and (variance / avg_price) > 0.3:  # >30% variance
                risks.append({
                    "drug": drug_name,
                    "risk_type": "Price Volatility",
                    "severity": "Medium",
                    "description": f"High price variance (${variance}/kg)",
                    "mitigation": "Establish long-term contracts with fixed pricing"
                })
        
        return risks
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate executive summary
        
        Args:
            results: Analysis results
        
        Returns:
            Summary string
        """
        summary_parts = []
        
        if results['drug_analyses']:
            num_drugs = len(results['drug_analyses'])
            summary_parts.append(f"Analyzed trade patterns for {num_drugs} drug(s).")
        
        if results['trade_flows']:
            position = results['trade_flows'].get('net_trade_position', 'Unknown')
            summary_parts.append(f"Overall position: {position}.")
        
        if results['sourcing_insights']:
            summary_parts.append(f"{len(results['sourcing_insights'])} sourcing insight(s) identified.")
        
        if results['supply_chain_risks']:
            high_risks = sum(1 for r in results['supply_chain_risks'] if r['severity'] == 'High')
            if high_risks > 0:
                summary_parts.append(f"⚠ {high_risks} high-severity supply chain risk(s) detected.")
        
        return " ".join(summary_parts) if summary_parts else "No trade data available for analysis."
    
    def compare_trade_positions(self, drugs: List[str]) -> Dict[str, Any]:
        """
        Compare trade positions for multiple drugs
        
        Args:
            drugs: List of drug names
        
        Returns:
            Comparative trade analysis
        """
        if self.verbose:
            print(f"\n[EXIM Agent] Comparing trade positions: {drugs}")
        
        comparisons = []
        
        for drug in drugs:
            analysis = self._analyze_drug_trade(drug, None, "both")
            if analysis:
                comparisons.append({
                    "drug": drug,
                    "total_volume_kg": analysis['trade_metrics']['total_volume_kg'],
                    "total_value_usd": analysis['trade_metrics']['total_value_usd'],
                    "trade_balance": analysis['trade_metrics']['trade_balance'],
                    "top_partner": analysis['top_trading_partners'][0]['country'] if analysis['top_trading_partners'] else "N/A"
                })
        
        # Sort by volume
        comparisons.sort(key=lambda x: x['total_volume_kg'], reverse=True)
        
        return {
            "drugs_compared": len(comparisons),
            "comparison_table": comparisons,
            "highest_volume": comparisons[0]['drug'] if comparisons else None,
            "summary": self._generate_comparison_summary(comparisons)
        }
    
    def _generate_comparison_summary(self, comparisons: List[Dict]) -> str:
        """Generate summary of trade comparison"""
        if not comparisons:
            return "No drugs to compare"
        
        total_volume = sum(c['total_volume_kg'] for c in comparisons)
        leader = comparisons[0]['drug']
        
        return f"Compared {len(comparisons)} drugs with {total_volume:.0f}kg total trade volume. " \
               f"Leader: {leader}."


# Convenience function
def get_exim_agent(verbose: bool = True) -> EXIMAgent:
    """
    Get instance of EXIM Agent
    
    Args:
        verbose: Whether to enable verbose logging
    
    Returns:
        Initialized EXIMAgent instance
    """
    return EXIMAgent(verbose=verbose)


# Test the agent
if __name__ == "__main__":
    print("="*70)
    print("EXIM AGENT - TEST SUITE")
    print("="*70)
    
    # Initialize agent
    agent = get_exim_agent(verbose=True)
    
    # Test 1: Single drug trade analysis
    print("\n" + "="*70)
    print("TEST 1: Trade Analysis - Metformin")
    print("="*70)
    result = agent.analyze_trade(drugs=["Metformin"])
    print(json.dumps(result, indent=2))
    
    # Test 2: Compare trade positions
    print("\n" + "="*70)
    print("TEST 2: Trade Position Comparison")
    print("="*70)
    comparison = agent.compare_trade_positions(["Metformin", "Insulin Glargine", "Atorvastatin"])
    print(json.dumps(comparison, indent=2))
    
    print("\n✓ All EXIM Agent tests completed!")