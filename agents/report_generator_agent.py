"""
Report Generator Agent - Documentation and Reporting Expert
Generates comprehensive PDF/Excel reports with visualizations
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import REPORTS_DIR

# Import reporting libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠ reportlab not available - PDF generation will be limited")


class ReportGeneratorAgent:
    """
    Report Generator Agent
    
    Responsibilities:
    - Generate PDF reports with formatted text and tables
    - Create Excel data exports
    - Add charts and visualizations
    - Format agent responses into downloadable documents
    - Maintain report archive
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize Report Generator Agent
        
        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.reports_directory = REPORTS_DIR
        
        # Ensure directory exists
        self.reports_directory.mkdir(exist_ok=True, parents=True)
        
        if self.verbose:
            print("✓ Report Generator Agent initialized")
            print(f"  Reports directory: {self.reports_directory}")
    
    def generate_report(
        self,
        query: str,
        agent_responses: List[Dict[str, Any]],
        synthesized_response: str,
        report_format: str = "pdf"  # "pdf" or "excel"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report from agent responses
        
        Args:
            query: Original user query
            agent_responses: List of responses from worker agents
            synthesized_response: Final synthesized response text
            report_format: Format to generate ("pdf" or "excel")
        
        Returns:
            Dictionary with report generation results
        """
        if self.verbose:
            print(f"\n[Report Generator Agent] Generating {report_format.upper()} report...")
        
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:30])
        filename = f"report_{safe_query}_{timestamp}.{report_format}"
        filepath = self.reports_directory / filename
        
        result = {
            "success": False,
            "format": report_format,
            "filename": filename,
            "filepath": str(filepath),
            "size_bytes": 0,
            "generation_time": None
        }
        
        try:
            start_time = datetime.now()
            
            if report_format == "pdf":
                success = self._generate_pdf_report(
                    filepath,
                    query,
                    agent_responses,
                    synthesized_response
                )
            elif report_format == "excel":
                success = self._generate_excel_report(
                    filepath,
                    query,
                    agent_responses,
                    synthesized_response
                )
            else:
                raise ValueError(f"Unsupported format: {report_format}")
            
            if success and filepath.exists():
                result["success"] = True
                result["size_bytes"] = filepath.stat().st_size
                result["size_mb"] = round(result["size_bytes"] / (1024*1024), 3)
                result["generation_time"] = (datetime.now() - start_time).total_seconds()
                
                if self.verbose:
                    print(f"✓ Report generated: {filename}")
                    print(f"  Size: {result['size_mb']} MB")
                    print(f"  Time: {result['generation_time']:.2f}s")
            else:
                result["error"] = "Report file was not created"
        
        except Exception as e:
            result["error"] = str(e)
            if self.verbose:
                print(f"✗ Error generating report: {e}")
        
        return result
    
    def _generate_pdf_report(
        self,
        filepath: Path,
        query: str,
        agent_responses: List[Dict[str, Any]],
        synthesized_response: str
    ) -> bool:
        """
        Generate PDF report using reportlab
        
        Args:
            filepath: Output file path
            query: User query
            agent_responses: Agent responses
            synthesized_response: Synthesized text
        
        Returns:
            Success boolean
        """
        if not REPORTLAB_AVAILABLE:
            # Fallback: Generate simple text file
            return self._generate_text_report(filepath, query, agent_responses, synthesized_response)
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                str(filepath),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Container for elements
            elements = []
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f77b4'),
                spaceAfter=30
            )
            
            # Title
            elements.append(Paragraph("Pharma Agentic AI - Analysis Report", title_style))
            elements.append(Spacer(1, 12))
            
            # Metadata
            metadata_data = [
                ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Query:", query],
                ["Agents Used:", len(agent_responses)]
            ]
            metadata_table = Table(metadata_data, colWidths=[1.5*inch, 5*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.grey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(metadata_table)
            elements.append(Spacer(1, 20))
            
            # Executive Summary
            elements.append(Paragraph("Executive Summary", styles['Heading2']))
            elements.append(Spacer(1, 12))
            
            # Split synthesized response into paragraphs
            for para in synthesized_response.split('\n\n'):
                if para.strip():
                    # Handle markdown headers
                    if para.startswith('##'):
                        elements.append(Paragraph(para.replace('##', '').strip(), styles['Heading3']))
                    elif para.startswith('#'):
                        elements.append(Paragraph(para.replace('#', '').strip(), styles['Heading2']))
                    else:
                        elements.append(Paragraph(para.strip(), styles['BodyText']))
                    elements.append(Spacer(1, 12))
            
            # Agent Responses Section
            elements.append(PageBreak())
            elements.append(Paragraph("Detailed Agent Responses", styles['Heading2']))
            elements.append(Spacer(1, 12))
            
            for i, response in enumerate(agent_responses, 1):
                agent_name = response.get('agent', 'Unknown').upper()
                success = response.get('success', False)
                
                # Agent header
                elements.append(Paragraph(f"{i}. {agent_name} Agent", styles['Heading3']))
                elements.append(Spacer(1, 6))
                
                # Status
                status_text = "✓ Success" if success else "✗ Failed"
                status_color = colors.green if success else colors.red
                elements.append(Paragraph(f"Status: {status_text}", styles['BodyText']))
                elements.append(Spacer(1, 6))
                
                # Data summary
                if success and response.get('data'):
                    data_str = json.dumps(response['data'], indent=2)[:500]  # Limit to 500 chars
                    elements.append(Paragraph(f"<pre>{data_str}...</pre>", styles['Code']))
                else:
                    error_msg = response.get('error', 'No data available')
                    elements.append(Paragraph(f"Error: {error_msg}", styles['BodyText']))
                
                elements.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(elements)
            return True
        
        except Exception as e:
            if self.verbose:
                print(f"✗ PDF generation error: {e}")
            return False
    
    def _generate_text_report(
        self,
        filepath: Path,
        query: str,
        agent_responses: List[Dict[str, Any]],
        synthesized_response: str
    ) -> bool:
        """
        Generate simple text report (fallback if reportlab unavailable)
        
        Args:
            filepath: Output file path (will be .txt)
            query: User query
            agent_responses: Agent responses
            synthesized_response: Synthesized text
        
        Returns:
            Success boolean
        """
        try:
            # Change extension to .txt
            txt_filepath = filepath.with_suffix('.txt')
            
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("PHARMA AGENTIC AI - ANALYSIS REPORT\n")
                f.write("="*70 + "\n\n")
                
                # Metadata
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Query: {query}\n")
                f.write(f"Agents Used: {len(agent_responses)}\n")
                f.write("\n" + "="*70 + "\n\n")
                
                # Executive Summary
                f.write("EXECUTIVE SUMMARY\n")
                f.write("-"*70 + "\n\n")
                f.write(synthesized_response)
                f.write("\n\n" + "="*70 + "\n\n")
                
                # Agent Responses
                f.write("DETAILED AGENT RESPONSES\n")
                f.write("-"*70 + "\n\n")
                
                for i, response in enumerate(agent_responses, 1):
                    agent_name = response.get('agent', 'Unknown').upper()
                    success = response.get('success', False)
                    
                    f.write(f"{i}. {agent_name} AGENT\n")
                    f.write("-"*40 + "\n")
                    f.write(f"Status: {'Success' if success else 'Failed'}\n\n")
                    
                    if success and response.get('data'):
                        f.write("Data:\n")
                        f.write(json.dumps(response['data'], indent=2))
                        f.write("\n")
                    else:
                        f.write(f"Error: {response.get('error', 'No data')}\n")
                    
                    f.write("\n" + "-"*40 + "\n\n")
                
                f.write("="*70 + "\n")
                f.write("END OF REPORT\n")
                f.write("="*70 + "\n")
            
            return True
        
        except Exception as e:
            if self.verbose:
                print(f"✗ Text report generation error: {e}")
            return False
    
    def _generate_excel_report(
        self,
        filepath: Path,
        query: str,
        agent_responses: List[Dict[str, Any]],
        synthesized_response: str
    ) -> bool:
        """
        Generate Excel report with multiple sheets
        
        Args:
            filepath: Output file path
            query: User query
            agent_responses: Agent responses
            synthesized_response: Synthesized text
        
        Returns:
            Success boolean
        """
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Sheet 1: Summary
                summary_data = {
                    'Field': ['Generated', 'Query', 'Agents Used', 'Successful Agents'],
                    'Value': [
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        query,
                        len(agent_responses),
                        sum(1 for r in agent_responses if r.get('success', False))
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 2: Synthesized Response
                response_data = {
                    'Section': ['Executive Summary'],
                    'Content': [synthesized_response]
                }
                response_df = pd.DataFrame(response_data)
                response_df.to_excel(writer, sheet_name='Analysis', index=False)
                
                # Sheet 3: Agent Results
                agent_data = []
                for response in agent_responses:
                    agent_data.append({
                        'Agent': response.get('agent', 'Unknown').upper(),
                        'Status': 'Success' if response.get('success', False) else 'Failed',
                        'Error': response.get('error', 'N/A'),
                        'Data Summary': str(response.get('data', 'N/A'))[:500]
                    })
                
                agents_df = pd.DataFrame(agent_data)
                agents_df.to_excel(writer, sheet_name='Agent Results', index=False)
                
                # Sheet 4: Data tables (if available)
                # Extract tabular data from agent responses
                for i, response in enumerate(agent_responses):
                    if response.get('success') and response.get('data'):
                        data = response['data']
                        agent_name = response.get('agent', f'Agent{i}')
                        
                        # Try to convert to DataFrame
                        try:
                            if isinstance(data, dict):
                                # Look for list data that can be tabulated
                                for key, value in data.items():
                                    if isinstance(value, list) and value and isinstance(value[0], dict):
                                        df = pd.DataFrame(value)
                                        sheet_name = f"{agent_name}_{key}"[:31]  # Excel limit
                                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        except:
                            pass  # Skip if conversion fails
            
            return True
        
        except Exception as e:
            if self.verbose:
                print(f"✗ Excel generation error: {e}")
            return False
    
    def list_reports(self) -> Dict[str, Any]:
        """
        List all generated reports
        
        Returns:
            Dictionary with report listing
        """
        if self.verbose:
            print(f"\n[Report Generator Agent] Listing reports...")
        
        reports = []
        
        for file_path in self.reports_directory.iterdir():
            if file_path.is_file():
                report_info = {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "size_mb": round(file_path.stat().st_size / (1024*1024), 3),
                    "created": datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    "format": file_path.suffix.lower().replace('.', '')
                }
                reports.append(report_info)
        
        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x['created'], reverse=True)
        
        return {
            "total_reports": len(reports),
            "reports": reports,
            "directory": str(self.reports_directory)
        }
    
    def delete_report(self, filename: str) -> Dict[str, Any]:
        """
        Delete a specific report
        
        Args:
            filename: Name of the report file
        
        Returns:
            Result dictionary
        """
        filepath = self.reports_directory / filename
        
        if not filepath.exists():
            return {
                "success": False,
                "error": "Report not found"
            }
        
        try:
            filepath.unlink()
            if self.verbose:
                print(f"✓ Deleted report: {filename}")
            return {
                "success": True,
                "filename": filename
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def clean_old_reports(self, days: int = 30) -> Dict[str, Any]:
        """
        Clean reports older than specified days
        
        Args:
            days: Age threshold in days
        
        Returns:
            Cleanup result dictionary
        """
        if self.verbose:
            print(f"\n[Report Generator Agent] Cleaning reports older than {days} days...")
        
        threshold = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted_count = 0
        deleted_files = []
        
        for file_path in self.reports_directory.iterdir():
            if file_path.is_file():
                if file_path.stat().st_mtime < threshold:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        deleted_files.append(file_path.name)
                    except Exception as e:
                        if self.verbose:
                            print(f"  ⚠ Could not delete {file_path.name}: {e}")
        
        if self.verbose:
            print(f"✓ Deleted {deleted_count} old report(s)")
        
        return {
            "deleted_count": deleted_count,
            "deleted_files": deleted_files,
            "threshold_days": days
        }
    
    def get_report_path(self, filename: str) -> Optional[str]:
        """
        Get full path for a report file
        
        Args:
            filename: Report filename
        
        Returns:
            Full path string or None if not found
        """
        filepath = self.reports_directory / filename
        return str(filepath) if filepath.exists() else None


# Convenience function
def get_report_generator_agent(verbose: bool = True) -> ReportGeneratorAgent:
    """
    Get instance of Report Generator Agent
    
    Args:
        verbose: Whether to enable verbose logging
    
    Returns:
        Initialized ReportGeneratorAgent instance
    """
    return ReportGeneratorAgent(verbose=verbose)


# Test the agent
if __name__ == "__main__":
    print("="*70)
    print("REPORT GENERATOR AGENT - TEST SUITE")
    print("="*70)
    
    # Initialize agent
    agent = get_report_generator_agent(verbose=True)
    
    # Test 1: Generate PDF report
    print("\n" + "="*70)
    print("TEST 1: Generate PDF Report")
    print("="*70)
    
    sample_query = "What are the market trends for Metformin?"
    sample_responses = [
        {
            "agent": "iqvia",
            "success": True,
            "data": {
                "drug_name": "Metformin",
                "market_size": 500,
                "cagr": 5.2
            }
        },
        {
            "agent": "clinical_trials",
            "success": True,
            "data": {
                "total_trials": 145,
                "active_trials": 32
            }
        }
    ]
    sample_synthesis = """
    # Executive Summary
    
    Metformin shows strong market performance with $500M market size and 5.2% CAGR.
    
    ## Key Findings
    - 145 clinical trials found, 32 currently active
    - Strong research interest in repurposing opportunities
    
    ## Recommendations
    - Consider lifecycle management strategies
    - Explore new indications in oncology
    """
    
    pdf_result = agent.generate_report(
        query=sample_query,
        agent_responses=sample_responses,
        synthesized_response=sample_synthesis,
        report_format="pdf"
    )
    print(json.dumps(pdf_result, indent=2))
    
    # Test 2: Generate Excel report
    print("\n" + "="*70)
    print("TEST 2: Generate Excel Report")
    print("="*70)
    
    excel_result = agent.generate_report(
        query=sample_query,
        agent_responses=sample_responses,
        synthesized_response=sample_synthesis,
        report_format="excel"
    )
    print(json.dumps(excel_result, indent=2))
    
    # Test 3: List reports
    print("\n" + "="*70)
    print("TEST 3: List All Reports")
    print("="*70)
    
    listing = agent.list_reports()
    print(json.dumps(listing, indent=2))
    
    # Test 4: Get report path
    print("\n" + "="*70)
    print("TEST 4: Get Report Path")
    print("="*70)
    
    if listing['reports']:
        first_report = listing['reports'][0]['filename']
        path = agent.get_report_path(first_report)
        print(f"Report path: {path}")
    
    print("\n✓ All Report Generator Agent tests completed!")