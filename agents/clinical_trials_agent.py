"""
Clinical Trials Agent - Clinical Pipeline Analyst
Tracks ongoing trials, sponsors, and development pipeline from ClinicalTrials.gov
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from collections import Counter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_fetchers import ClinicalTrialsFetcher


class ClinicalTrialsAgent:
    """
    Clinical Trials Intelligence Agent
    
    Responsibilities:
    - Search ongoing clinical trials by drug, disease, or indication
    - Analyze trial phases and pipeline stages
    - Identify sponsors and research institutions
    - Track trial status and completion timelines
    - Identify repurposing opportunities from off-label trials
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize Clinical Trials Agent
        
        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.fetcher = ClinicalTrialsFetcher()
        
        if self.verbose:
            print("✓ Clinical Trials Agent initialized")
    
    def search_trials(
        self,
        drugs: List[str] = None,
        diseases: List[str] = None,
        therapeutic_areas: List[str] = None,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Comprehensive clinical trials search and analysis
        
        Args:
            drugs: List of drug/intervention names
            diseases: List of disease/condition names
            therapeutic_areas: List of therapeutic areas (used as keywords)
            max_results: Maximum number of trials to retrieve
        
        Returns:
            Dictionary with clinical trials analysis
        """
        if self.verbose:
            print(f"\n[Clinical Trials Agent] Searching trials...")
            print(f"  Drugs: {drugs}")
            print(f"  Diseases: {diseases}")
            print(f"  Therapeutic Areas: {therapeutic_areas}")
        
        results = {
            "summary": "",
            "total_trials_found": 0,
            "trials_by_drug": {},
            "trials_by_disease": {},
            "phase_distribution": {},
            "sponsor_analysis": {},
            "status_summary": {},
            "key_findings": [],
            "repurposing_opportunities": [],
            "detailed_trials": []
        }
        
        all_trials = []
        
        # Search by drugs
        if drugs:
            for drug in drugs:
                trials = self.fetcher.search_trials(intervention=drug, max_results=max_results)
                if trials:
                    results["trials_by_drug"][drug] = {
                        "count": len(trials),
                        "trials": trials
                    }
                    all_trials.extend(trials)
                    
                    if self.verbose:
                        print(f"  Found {len(trials)} trials for drug: {drug}")
        
        # Search by diseases/conditions
        if diseases:
            for disease in diseases:
                trials = self.fetcher.search_trials(condition=disease, max_results=max_results)
                if trials:
                    results["trials_by_disease"][disease] = {
                        "count": len(trials),
                        "trials": trials
                    }
                    all_trials.extend(trials)
                    
                    if self.verbose:
                        print(f"  Found {len(trials)} trials for disease: {disease}")
        
        # Search by therapeutic areas (as condition keywords)
        if therapeutic_areas:
            for area in therapeutic_areas:
                trials = self.fetcher.search_trials(condition=area, max_results=max_results)
                if trials:
                    all_trials.extend(trials)
                    
                    if self.verbose:
                        print(f"  Found {len(trials)} trials for area: {area}")
        
        # Remove duplicates (based on NCT ID)
        unique_trials = {trial['nct_id']: trial for trial in all_trials}.values()
        results["detailed_trials"] = list(unique_trials)
        results["total_trials_found"] = len(results["detailed_trials"])
        
        # Analyze trials
        if results["detailed_trials"]:
            results["phase_distribution"] = self._analyze_phases(results["detailed_trials"])
            results["sponsor_analysis"] = self._analyze_sponsors(results["detailed_trials"])
            results["status_summary"] = self._analyze_status(results["detailed_trials"])
            results["key_findings"] = self._extract_key_findings(results)
            results["repurposing_opportunities"] = self._identify_repurposing_opportunities(results, drugs)
        
        # Generate summary
        results["summary"] = self._generate_summary(results)
        
        if self.verbose:
            print(f"✓ Search complete - {results['total_trials_found']} unique trials found")
        
        return results
    
    def _analyze_phases(self, trials: List[Dict]) -> Dict[str, Any]:
        """
        Analyze distribution of trial phases
        
        Args:
            trials: List of trial dictionaries
        
        Returns:
            Phase distribution analysis
        """
        phases = [trial['phase'] for trial in trials if trial['phase'] != 'N/A']
        phase_counts = Counter(phases)
        
        total = len(phases)
        distribution = {
            phase: {
                "count": count,
                "percentage": round((count / total * 100), 2) if total > 0 else 0
            }
            for phase, count in phase_counts.items()
        }
        
        return {
            "distribution": distribution,
            "total_with_phase": total,
            "most_common_phase": phase_counts.most_common(1)[0][0] if phase_counts else "N/A",
            "advanced_trials_count": sum(1 for p in phases if "Phase 3" in p or "Phase 4" in p)
        }
    
    def _analyze_sponsors(self, trials: List[Dict]) -> Dict[str, Any]:
        """
        Analyze trial sponsors and institutions
        
        Args:
            trials: List of trial dictionaries
        
        Returns:
            Sponsor analysis
        """
        sponsors = [trial['sponsor'] for trial in trials if trial['sponsor'] != 'N/A']
        sponsor_counts = Counter(sponsors)
        
        # Categorize sponsors
        industry_keywords = ['pharma', 'pharmaceutical', 'inc', 'ltd', 'corporation', 'therapeutics', 'bioscience']
        academic_keywords = ['university', 'hospital', 'medical center', 'institute', 'college']
        
        industry_sponsors = []
        academic_sponsors = []
        other_sponsors = []
        
        for sponsor in sponsors:
            sponsor_lower = sponsor.lower()
            if any(kw in sponsor_lower for kw in industry_keywords):
                industry_sponsors.append(sponsor)
            elif any(kw in sponsor_lower for kw in academic_keywords):
                academic_sponsors.append(sponsor)
            else:
                other_sponsors.append(sponsor)
        
        return {
            "total_unique_sponsors": len(sponsor_counts),
            "top_sponsors": [
                {"sponsor": sponsor, "trial_count": count}
                for sponsor, count in sponsor_counts.most_common(10)
            ],
            "sponsor_categories": {
                "industry": len(set(industry_sponsors)),
                "academic": len(set(academic_sponsors)),
                "other": len(set(other_sponsors))
            },
            "industry_vs_academic_ratio": round(
                len(industry_sponsors) / len(academic_sponsors), 2
            ) if academic_sponsors else "N/A"
        }
    
    def _analyze_status(self, trials: List[Dict]) -> Dict[str, Any]:
        """
        Analyze trial status distribution
        
        Args:
            trials: List of trial dictionaries
        
        Returns:
            Status analysis
        """
        statuses = [trial['status'] for trial in trials]
        status_counts = Counter(statuses)
        
        total = len(statuses)
        
        return {
            "distribution": {
                status: {
                    "count": count,
                    "percentage": round((count / total * 100), 2) if total > 0 else 0
                }
                for status, count in status_counts.items()
            },
            "active_trials": status_counts.get('Recruiting', 0) + status_counts.get('Active, not recruiting', 0),
            "completed_trials": status_counts.get('Completed', 0),
            "ongoing_percentage": round(
                ((status_counts.get('Recruiting', 0) + status_counts.get('Active, not recruiting', 0)) / total * 100), 2
            ) if total > 0 else 0
        }
    
    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """
        Extract key findings from trial analysis
        
        Args:
            results: Analysis results dictionary
        
        Returns:
            List of key finding strings
        """
        findings = []
        
        total = results['total_trials_found']
        if total == 0:
            return ["No trials found for the specified criteria"]
        
        # Phase findings
        if results['phase_distribution']:
            most_common = results['phase_distribution']['most_common_phase']
            advanced = results['phase_distribution']['advanced_trials_count']
            findings.append(f"Most trials in {most_common} with {advanced} in advanced stages (Phase 3/4)")
        
        # Sponsor findings
        if results['sponsor_analysis']:
            top_sponsor = results['sponsor_analysis']['top_sponsors'][0] if results['sponsor_analysis']['top_sponsors'] else None
            if top_sponsor:
                findings.append(f"Leading sponsor: {top_sponsor['sponsor']} with {top_sponsor['trial_count']} trials")
        
        # Status findings
        if results['status_summary']:
            active = results['status_summary']['active_trials']
            findings.append(f"{active} active/recruiting trials representing strong ongoing research interest")
        
        return findings
    
    def _identify_repurposing_opportunities(
        self, 
        results: Dict[str, Any], 
        drugs: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify potential drug repurposing opportunities
        
        Args:
            results: Analysis results
            drugs: List of drugs being searched
        
        Returns:
            List of repurposing opportunities
        """
        opportunities = []
        
        if not drugs:
            return opportunities
        
        for drug in drugs:
            if drug in results['trials_by_drug']:
                trials = results['trials_by_drug'][drug]['trials']
                
                # Look for trials in different indications
                conditions = set()
                for trial in trials:
                    conditions.update(trial.get('conditions', []))
                
                if len(conditions) > 1:
                    opportunities.append({
                        "drug": drug,
                        "trial_count": len(trials),
                        "conditions_being_studied": list(conditions),
                        "repurposing_potential": "High" if len(conditions) >= 3 else "Moderate",
                        "note": f"Being studied for {len(conditions)} different conditions"
                    })
        
        return opportunities
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate executive summary of clinical trials analysis
        
        Args:
            results: Analysis results dictionary
        
        Returns:
            Summary string
        """
        total = results['total_trials_found']
        
        if total == 0:
            return "No clinical trials found matching the search criteria."
        
        summary_parts = [f"Found {total} clinical trial(s)."]
        
        if results['phase_distribution']:
            advanced = results['phase_distribution']['advanced_trials_count']
            summary_parts.append(f"{advanced} in advanced stages (Phase 3/4).")
        
        if results['status_summary']:
            active = results['status_summary']['active_trials']
            summary_parts.append(f"{active} actively recruiting or ongoing.")
        
        if results['repurposing_opportunities']:
            summary_parts.append(
                f"{len(results['repurposing_opportunities'])} potential repurposing opportunity(ies) identified."
            )
        
        return " ".join(summary_parts)
    
    def get_trial_details(self, nct_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific trial
        
        Args:
            nct_id: NCT identifier for the trial
        
        Returns:
            Detailed trial information
        """
        # Note: This would require a separate API call in production
        # For now, we return a placeholder
        return {
            "nct_id": nct_id,
            "note": "Detailed trial information requires individual API call",
            "url": f"https://clinicaltrials.gov/study/{nct_id}"
        }
    
    def analyze_pipeline(
        self, 
        intervention: str
    ) -> Dict[str, Any]:
        """
        Analyze development pipeline for a specific drug/intervention
        
        Args:
            intervention: Drug or intervention name
        
        Returns:
            Pipeline analysis
        """
        if self.verbose:
            print(f"\n[Clinical Trials Agent] Analyzing pipeline for: {intervention}")
        
        trials = self.fetcher.search_trials(intervention=intervention, max_results=50)
        
        if not trials:
            return {
                "intervention": intervention,
                "pipeline_status": "No active trials found",
                "development_stage": "Unknown"
            }
        
        # Analyze by phase
        phase_counts = Counter(trial['phase'] for trial in trials if trial['phase'] != 'N/A')
        
        # Determine development stage
        if any("Phase 4" in p for p in phase_counts.keys()):
            stage = "Post-Marketing"
        elif any("Phase 3" in p for p in phase_counts.keys()):
            stage = "Late-Stage Development"
        elif any("Phase 2" in p for p in phase_counts.keys()):
            stage = "Mid-Stage Development"
        elif any("Phase 1" in p for p in phase_counts.keys()):
            stage = "Early-Stage Development"
        else:
            stage = "Preclinical/Unknown"
        
        return {
            "intervention": intervention,
            "total_trials": len(trials),
            "development_stage": stage,
            "phase_breakdown": dict(phase_counts),
            "active_trials": sum(1 for t in trials if t['status'] in ['Recruiting', 'Active, not recruiting']),
            "completed_trials": sum(1 for t in trials if t['status'] == 'Completed'),
            "pipeline_health": self._assess_pipeline_health(trials)
        }
    
    def _assess_pipeline_health(self, trials: List[Dict]) -> str:
        """
        Assess overall pipeline health
        
        Args:
            trials: List of trial dictionaries
        
        Returns:
            Health assessment string
        """
        if not trials:
            return "No Pipeline"
        
        active_count = sum(1 for t in trials if t['status'] in ['Recruiting', 'Active, not recruiting'])
        total_count = len(trials)
        
        active_ratio = active_count / total_count if total_count > 0 else 0
        
        # Check for advanced phase trials
        has_advanced = any("Phase 3" in t['phase'] or "Phase 4" in t['phase'] for t in trials)
        
        if active_ratio > 0.5 and has_advanced:
            return "Strong - Multiple active trials including advanced phases"
        elif active_ratio > 0.3:
            return "Moderate - Decent pipeline activity"
        elif active_count > 0:
            return "Weak - Limited active development"
        else:
            return "Dormant - No active trials"
    
    def compare_pipelines(self, interventions: List[str]) -> Dict[str, Any]:
        """
        Compare development pipelines for multiple drugs
        
        Args:
            interventions: List of drug/intervention names
        
        Returns:
            Comparative pipeline analysis
        """
        if self.verbose:
            print(f"\n[Clinical Trials Agent] Comparing pipelines: {interventions}")
        
        comparisons = []
        
        for intervention in interventions:
            pipeline = self.analyze_pipeline(intervention)
            comparisons.append({
                "intervention": intervention,
                "total_trials": pipeline.get('total_trials', 0),
                "development_stage": pipeline.get('development_stage', 'Unknown'),
                "active_trials": pipeline.get('active_trials', 0),
                "pipeline_health": pipeline.get('pipeline_health', 'Unknown')
            })
        
        # Sort by total trials
        comparisons.sort(key=lambda x: x['total_trials'], reverse=True)
        
        return {
            "interventions_compared": len(comparisons),
            "comparison_table": comparisons,
            "most_active_pipeline": comparisons[0]['intervention'] if comparisons else None,
            "summary": self._generate_comparison_summary(comparisons)
        }
    
    def _generate_comparison_summary(self, comparisons: List[Dict]) -> str:
        """Generate summary of pipeline comparison"""
        if not comparisons:
            return "No pipelines to compare"
        
        total_trials = sum(c['total_trials'] for c in comparisons)
        leader = comparisons[0]['intervention']
        
        return f"Compared {len(comparisons)} pipelines with {total_trials} total trials. " \
               f"Leader: {leader} with {comparisons[0]['total_trials']} trials."


# Convenience function
def get_clinical_trials_agent(verbose: bool = True) -> ClinicalTrialsAgent:
    """
    Get instance of Clinical Trials Agent
    
    Args:
        verbose: Whether to enable verbose logging
    
    Returns:
        Initialized ClinicalTrialsAgent instance
    """
    return ClinicalTrialsAgent(verbose=verbose)


# Test the agent
if __name__ == "__main__":
    print("="*70)
    print("CLINICAL TRIALS AGENT - TEST SUITE")
    print("="*70)
    
    # Initialize agent
    agent = get_clinical_trials_agent(verbose=True)
    
    # Test 1: Search trials by drug
    print("\n" + "="*70)
    print("TEST 1: Search Trials - Metformin")
    print("="*70)
    result = agent.search_trials(drugs=["Metformin"], max_results=10)
    print(f"\nSummary: {result['summary']}")
    print(f"Total trials: {result['total_trials_found']}")
    print(f"Key findings: {result['key_findings']}")
    
    # Test 2: Search by disease
    print("\n" + "="*70)
    print("TEST 2: Search Trials - Diabetes")
    print("="*70)
    result = agent.search_trials(diseases=["Diabetes"], max_results=10)
    print(f"\nSummary: {result['summary']}")
    print(f"Phase distribution: {json.dumps(result['phase_distribution'], indent=2)}")
    
    # Test 3: Pipeline analysis
    print("\n" + "="*70)
    print("TEST 3: Pipeline Analysis - Ibuprofen")
    print("="*70)
    pipeline = agent.analyze_pipeline("Ibuprofen")
    print(json.dumps(pipeline, indent=2))
    
    # Test 4: Compare pipelines
    print("\n" + "="*70)
    print("TEST 4: Pipeline Comparison")
    print("="*70)
    comparison = agent.compare_pipelines(["Metformin", "Insulin", "Atorvastatin"])
    print(json.dumps(comparison, indent=2))
    
    print("\n✓ All Clinical Trials Agent tests completed!")