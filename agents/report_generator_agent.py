"""
Enhanced Report Generator Agent - Professional PDF with Charts
STANDALONE VERSION 
"""

import json
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import REPORTS_DIR

# Import reporting libraries
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (
        PageBreak, Paragraph, SimpleDocTemplate, Spacer, 
        Table, TableStyle, Image, HRFlowable
    )
    REPORTLAB_AVAILABLE = True
    print(" ReportLab imported successfully")
except ImportError as e:
    REPORTLAB_AVAILABLE = False
    print(f" ReportLab import failed: {e}")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    print(" Matplotlib imported successfully")
except ImportError as e:
    MATPLOTLIB_AVAILABLE = False
    print(f" Matplotlib import failed: {e}")


class ProfessionalReportGenerator:
    """Enhanced Report Generator with Professional Formatting and Charts"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Ensure reports directory exists
        self.reports_directory = REPORTS_DIR
        try:
            self.reports_directory.mkdir(exist_ok=True, parents=True)
            if self.verbose:
                print(f" Reports directory: {self.reports_directory}")
                print(f"  Exists: {self.reports_directory.exists()}")
                print(f"  Writable: {self.reports_directory.is_dir()}")
        except Exception as e:
            print(f" Error with reports directory: {e}")
            # Fallback
            self.reports_directory = Path("./reports")
            self.reports_directory.mkdir(exist_ok=True, parents=True)
        
        # Set matplotlib styling
        if MATPLOTLIB_AVAILABLE:
            try:
                sns.set_style("whitegrid")
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
            except:
                pass
        
        if self.verbose:
            print(f" Report Generator initialized")
            print(f"  ReportLab: {REPORTLAB_AVAILABLE}")
            print(f"  Matplotlib: {MATPLOTLIB_AVAILABLE}")

    def generate_report(
        self,
        query: str,
        agent_responses: List[Dict[str, Any]],
        synthesized_response: str,
        report_format: str = "pdf",
    ) -> Dict[str, Any]:
        """Generate professional report"""
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"GENERATING {report_format.upper()} REPORT")
            print(f"{'='*60}")

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:30])
        filename = f"pharma_report_{safe_query}_{timestamp}.{report_format}"
        filepath = self.reports_directory / filename

        result = {
            "success": False,
            "format": report_format,
            "filename": filename,
            "filepath": str(filepath),
            "size_bytes": 0,
            "generation_time": None,
            "error": None
        }

        try:
            start_time = datetime.now()

            if report_format == "pdf":
                success = self._generate_pdf(
                    filepath, query, agent_responses, synthesized_response
                )
            elif report_format == "excel":
                success = self._generate_excel(
                    filepath, query, agent_responses, synthesized_response
                )
            else:
                raise ValueError(f"Unsupported format: {report_format}")

            if success and filepath.exists():
                result["success"] = True
                result["size_bytes"] = filepath.stat().st_size
                result["size_mb"] = round(result["size_bytes"] / (1024 * 1024), 3)
                result["generation_time"] = (datetime.now() - start_time).total_seconds()

                if self.verbose:
                    print(f" Report generated: {filename}")
                    print(f"  Size: {result['size_mb']} MB")
                    print(f"  Time: {result['generation_time']:.2f}s")
            else:
                result["error"] = "Report file was not created"
                if self.verbose:
                    print(f" File not created: {filepath}")
                    print(f"  Exists: {filepath.exists()}")

        except Exception as e:
            result["error"] = str(e)
            if self.verbose:
                print(f" Error: {e}")
                import traceback
                traceback.print_exc()

        return result

    def _generate_pdf(self, filepath, query, agent_responses, synthesized_response):
        """Generate PDF report"""
        
        if not REPORTLAB_AVAILABLE:
            if self.verbose:
                print(" ReportLab not available, using text fallback")
            return self._generate_text(filepath, query, agent_responses, synthesized_response)

        try:
            if self.verbose:
                print(f"  Creating PDF: {filepath}")

            # Create document
            doc = SimpleDocTemplate(
                str(filepath),
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=1*inch,
                bottomMargin=0.75*inch,
            )

            elements = []
            styles = self._get_styles()

            # Cover page
            elements.extend(self._create_cover(query, styles))
            elements.append(PageBreak())

            # Executive summary
            elements.append(Paragraph("Executive Summary", styles['CustomHeading1']))
            elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1f77b4")))
            elements.append(Spacer(1, 0.2*inch))
            
            # Parse and add synthesized response with proper formatting
            summary_elements = self._parse_markdown_text(synthesized_response, styles)
            elements.extend(summary_elements)
            
            elements.append(PageBreak())

            # Charts section
            if MATPLOTLIB_AVAILABLE:
                elements.append(Paragraph("Data Visualizations", styles['CustomHeading1']))
                elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1f77b4")))
                elements.append(Spacer(1, 0.2*inch))
                
                charts = self._create_all_charts(agent_responses, styles)
                if charts:
                    elements.extend(charts)
                else:
                    elements.append(Paragraph("<i>No visualizations available</i>", styles['CustomItalic']))
                
                elements.append(PageBreak())

            # Agent responses
            elements.append(Paragraph("Detailed Analysis", styles['CustomHeading1']))
            elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1f77b4")))
            elements.append(Spacer(1, 0.2*inch))
            
            for i, resp in enumerate(agent_responses, 1):
                elements.extend(self._format_agent(resp, i, styles))
                if i < len(agent_responses):
                    elements.append(Spacer(1, 0.1*inch))
                    elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
                    elements.append(Spacer(1, 0.1*inch))

            # Build PDF
            if self.verbose:
                print(f"  Building PDF with {len(elements)} elements...")
            
            doc.build(elements, onFirstPage=self._add_page_num, onLaterPages=self._add_page_num)
            
            if self.verbose:
                print(f"  PDF built")
                print(f"  File exists: {filepath.exists()}")
            
            return filepath.exists()

        except Exception as e:
            if self.verbose:
                print(f"  PDF error: {e}")
                import traceback
                traceback.print_exc()
            
            # Fallback to text
            return self._generate_text(filepath, query, agent_responses, synthesized_response)

    def _get_styles(self):
        """Get paragraph styles"""
        styles = getSampleStyleSheet()
        
        # Use custom names to avoid conflicts with existing styles
        styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor("#1f77b4"),
            spaceAfter=0.15*inch,
            fontName='Helvetica-Bold',
        ))
        
        styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor("#2ca02c"),
            spaceAfter=0.1*inch,
            fontName='Helvetica-Bold',
        ))
        
        styles.add(ParagraphStyle(
            name='CustomBodyText',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=0.1*inch,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        styles.add(ParagraphStyle(
            name='CustomItalic',
            parent=styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Oblique',
            textColor=colors.HexColor("#666666")
        ))
        
        styles.add(ParagraphStyle(
            name='CustomHeading3',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor("#d62728"),
            spaceAfter=0.08*inch,
            spaceBefore=0.1*inch,
            fontName='Helvetica-Bold',
        ))
        
        styles.add(ParagraphStyle(
            name='CustomBullet',
            parent=styles['Normal'],
            fontSize=11,
            leftIndent=0.3*inch,
            bulletIndent=0.1*inch,
            spaceAfter=6,
            fontName='Helvetica'
        ))
        
        return styles

    def _parse_markdown_text(self, text, styles):
        """Parse markdown-style text to formatted PDF elements"""
        elements = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Headers
            if line.startswith('### '):
                header_text = line.replace('### ', '').strip()
                elements.append(Paragraph(f"<b>{header_text}</b>", styles['CustomHeading3']))
            elif line.startswith('## '):
                header_text = line.replace('## ', '').strip()
                elements.append(Paragraph(f"<b>{header_text}</b>", styles['CustomHeading2']))
            elif line.startswith('# '):
                header_text = line.replace('# ', '').strip()
                elements.append(Paragraph(f"<b>{header_text}</b>", styles['CustomHeading1']))
            
            # Bullet points
            elif line.startswith('- ') or line.startswith('* '):
                bullet_text = line[2:].strip()
                # Format bold items
                bullet_text = self._format_inline_markdown(bullet_text)
                elements.append(Paragraph(f"• {bullet_text}", styles['CustomBullet']))
            
            # Numbered lists
            elif len(line) > 2 and line[0].isdigit() and line[1:3] in ['. ', ') ']:
                list_text = line.split('. ', 1)[1] if '. ' in line else line.split(') ', 1)[1]
                list_text = self._format_inline_markdown(list_text)
                elements.append(Paragraph(f"{line.split()[0]} {list_text}", styles['CustomBullet']))
            
            # Bold standalone lines (like **Clinical Trials**)
            elif line.startswith('**') and line.endswith('**'):
                bold_text = line.strip('**')
                elements.append(Paragraph(f"<b>{bold_text}</b>", styles['CustomHeading3']))
            
            # Regular paragraphs
            else:
                formatted_text = self._format_inline_markdown(line)
                elements.append(Paragraph(formatted_text, styles['CustomBodyText']))
            
            elements.append(Spacer(1, 0.08*inch))
            i += 1
        
        return elements

    def _format_inline_markdown(self, text):
        """Format inline markdown (bold, italic)"""
        # Bold text: **text** or __text__
        import re
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
        
        # Italic text: *text* or _text_
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
        
        # Escape special characters for XML
        text = text.replace('&', '&amp;')
        
        return text

    def _create_cover(self, query, styles):
        """Create cover page"""
        elements = []
        elements.append(Spacer(1, 1*inch))
        elements.append(Paragraph(
            "<b>Pharmaceutical Intelligence Report</b>",
            ParagraphStyle(
                'Title',
                parent=styles['Normal'],
                fontSize=28,
                textColor=colors.HexColor("#1f77b4"),
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
        ))
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph(
            f"<i>{query}</i>",
            ParagraphStyle(
                'Subtitle',
                parent=styles['Normal'],
                fontSize=14,
                alignment=TA_CENTER,
                fontName='Helvetica-Oblique'
            )
        ))
        elements.append(Spacer(1, 0.5*inch))
        
        # Metadata table
        metadata = [
            ['Generated:', datetime.now().strftime("%B %d, %Y at %H:%M")],
            ['Powered By:', 'PharmAID Platform'],
        ]
        
        table = Table(metadata, colWidths=[2*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        
        elements.append(table)
        return elements

    def _create_all_charts(self, agent_responses, styles):
        """Create all charts"""
        if not MATPLOTLIB_AVAILABLE:
            return []
        
        elements = []
        
        for resp in agent_responses:
            if not resp.get("success"):
                continue
            
            agent = resp.get("agent", "")
            data = resp.get("data", {})
            
            if agent == "iqvia":
                charts = self._create_iqvia_charts(data)
                if charts:
                    elements.append(Paragraph("<b>Market Analysis</b>", styles['CustomHeading2']))
                    elements.extend(charts)
            
            elif agent == "clinical_trials":
                charts = self._create_ct_charts(data)
                if charts:
                    elements.append(Paragraph("<b>Clinical Trials</b>", styles['CustomHeading2']))
                    elements.extend(charts)
            
            elif agent == "patent":
                charts = self._create_patent_charts(data)
                if charts:
                    elements.append(Paragraph("<b>Patent Landscape</b>", styles['CustomHeading2']))
                    elements.extend(charts)
            
            elif agent == "exim":
                charts = self._create_exim_charts(data)
                if charts:
                    elements.append(Paragraph("<b>Trade Analysis</b>", styles['CustomHeading2']))
                    elements.extend(charts)
        
        return elements

    def _create_iqvia_charts(self, data):
        """Create IQVIA charts - ALL OF THEM"""
        charts = []
        
        try:
            analyses = data.get("drug_analyses", [])
            if not analyses:
                return charts
            
            # Chart 1: Market size bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            drugs = [d.get("drug_name", "Unknown") for d in analyses[:5]]
            sales = [d.get("market_metrics", {}).get("current_sales_usd_million", 0) for d in analyses[:5]]
            
            ax.bar(drugs, sales, color='#1f77b4', alpha=0.7)
            ax.set_ylabel('Sales (USD Million)', fontweight='bold')
            ax.set_title('Market Size Comparison', fontweight='bold', pad=15)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(sales):
                ax.text(i, v, f'${v:.0f}M', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            charts.append(Image(buf, width=6*inch, height=3*inch))
            charts.append(Spacer(1, 0.15*inch))
            
            # Chart 2: CAGR comparison
            if len(analyses) > 1:
                fig, ax = plt.subplots(figsize=(8, 4))
                cagr = [d.get("market_metrics", {}).get("cagr_percent", 0) for d in analyses[:5]]
                
                colors = ['#2ca02c' if c >= 0 else '#d62728' for c in cagr]
                bars = ax.barh(drugs, cagr, color=colors, alpha=0.7)
                
                ax.set_xlabel('CAGR (%)', fontweight='bold')
                ax.set_title('Growth Rate (CAGR) Comparison', fontweight='bold', pad=15)
                ax.grid(axis='x', alpha=0.3)
                ax.axvline(x=0, color='black', linewidth=0.8)
                
                # Add value labels
                for i, v in enumerate(cagr):
                    label_x = v + 0.5 if v >= 0 else v - 0.5
                    ax.text(label_x, i, f'{v:.1f}%', va='center', fontweight='bold')
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                
                charts.append(Image(buf, width=6*inch, height=3*inch))
                charts.append(Spacer(1, 0.15*inch))
        
        except Exception as e:
            if self.verbose:
                print(f"   IQVIA chart error: {e}")
        
        return charts

    def _create_ct_charts(self, data):
        """Create clinical trials charts - ALL OF THEM"""
        charts = []
        
        try:
            # Chart 1: Phase distribution pie chart
            phase_dist = data.get("phase_distribution", {}).get("distribution", {})
            if phase_dist:
                fig, ax = plt.subplots(figsize=(7, 5))
                phases = list(phase_dist.keys())
                counts = [phase_dist[p]["count"] for p in phases]
                
                colors_pie = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                wedges, texts, autotexts = ax.pie(counts, labels=phases, autopct='%1.1f%%',
                                                   colors=colors_pie[:len(phases)], startangle=90)
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax.set_title('Trial Phase Distribution', fontweight='bold', pad=15)
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                
                charts.append(Image(buf, width=5*inch, height=4*inch))
                charts.append(Spacer(1, 0.15*inch))
            
            # Chart 2: Status distribution bar chart
            status_summary = data.get("status_summary", {}).get("distribution", {})
            if status_summary:
                fig, ax = plt.subplots(figsize=(8, 4))
                
                statuses = list(status_summary.keys())
                counts_status = [status_summary[s]["count"] for s in statuses]
                
                bars = ax.bar(statuses, counts_status, color='#2ca02c', alpha=0.7)
                
                ax.set_ylabel('Number of Trials', fontweight='bold')
                ax.set_title('Trial Status Distribution', fontweight='bold', pad=15)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                
                charts.append(Image(buf, width=6*inch, height=3*inch))
                charts.append(Spacer(1, 0.15*inch))
        
        except Exception as e:
            if self.verbose:
                print(f"  CT chart error: {e}")
        
        return charts

    def _create_patent_charts(self, data):
        """Create patent charts - ALL OF THEM"""
        charts = []
        
        try:
            # Chart 1: Expiry timeline bar chart
            timeline = data.get("expiry_timeline", {})
            if timeline:
                fig, ax = plt.subplots(figsize=(8, 4))
                categories = ['Expired', 'Soon\n(0-2y)', 'Medium\n(2-5y)', 'Long\n(5+y)']
                counts = [
                    timeline.get("expired_count", 0),
                    timeline.get("expiring_soon_count", 0),
                    timeline.get("expiring_medium_count", 0),
                    timeline.get("expiring_long_count", 0)
                ]
                
                colors_exp = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c']
                bars = ax.bar(categories, counts, color=colors_exp)
                
                ax.set_ylabel('Number of Patents', fontweight='bold')
                ax.set_title('Patent Expiry Timeline', fontweight='bold', pad=15)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                
                charts.append(Image(buf, width=6*inch, height=3*inch))
                charts.append(Spacer(1, 0.15*inch))
            
            # Chart 2: Top patent holders
            competitive = data.get("competitive_landscape", {})
            top_assignees = competitive.get("top_assignees", [])
            
            if top_assignees and len(top_assignees) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                
                assignees = [a["assignee"][:30] + "..." if len(a["assignee"]) > 30 else a["assignee"] 
                            for a in top_assignees[:8]]
                patent_counts = [a["patent_count"] for a in top_assignees[:8]]
                
                bars = ax.barh(assignees, patent_counts, color='#1f77b4', alpha=0.7)
                
                ax.set_xlabel('Number of Patents', fontweight='bold')
                ax.set_title('Top Patent Holders', fontweight='bold', pad=15)
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, patent_counts)):
                    ax.text(val, i, f'  {val}', va='center', fontweight='bold')
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                
                charts.append(Image(buf, width=6*inch, height=4*inch))
                charts.append(Spacer(1, 0.15*inch))
        
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Patent chart error: {e}")
        
        return charts

    def _create_exim_charts(self, data):
        """Create EXIM charts - ALL OF THEM"""
        charts = []
        
        try:
            analyses = data.get("drug_analyses", [])
            if not analyses:
                return charts
            
            # Chart 1: Import/Export comparison
            fig, ax = plt.subplots(figsize=(8, 4))
            drugs = [d.get("drug_name", "Unknown") for d in analyses[:5]]
            imports = [d.get("trade_metrics", {}).get("total_import_value_usd", 0)/1e6 for d in analyses[:5]]
            exports = [d.get("trade_metrics", {}).get("total_export_value_usd", 0)/1e6 for d in analyses[:5]]
            
            x = range(len(drugs))
            width = 0.35
            ax.bar([i-width/2 for i in x], imports, width, label='Imports', color='#d62728', alpha=0.8)
            ax.bar([i+width/2 for i in x], exports, width, label='Exports', color='#2ca02c', alpha=0.8)
            
            ax.set_ylabel('Trade Value (USD Million)', fontweight='bold')
            ax.set_title('Import vs Export Comparison', fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(drugs, rotation=45, ha='right')
            ax.legend(loc='upper left', frameon=True, shadow=True)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            charts.append(Image(buf, width=6*inch, height=3*inch))
            charts.append(Spacer(1, 0.15*inch))
            
            # Chart 2: Trade balance
            if len(analyses) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                
                balance = []
                for d in analyses[:5]:
                    metrics = d.get("trade_metrics", {})
                    exp = metrics.get("total_export_value_usd", 0)
                    imp = metrics.get("total_import_value_usd", 0)
                    balance.append((exp - imp) / 1e6)
                
                colors_balance = ['#2ca02c' if b >= 0 else '#d62728' for b in balance]
                bars = ax.barh(drugs, balance, color=colors_balance, alpha=0.7)
                
                ax.set_xlabel('Net Trade Balance (USD Million)', fontweight='bold')
                ax.set_title('Trade Balance by Drug (Exports - Imports)', fontweight='bold', pad=15)
                ax.grid(axis='x', alpha=0.3)
                ax.axvline(x=0, color='black', linewidth=1)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, balance)):
                    label = f'${val:.1f}M'
                    if val >= 0:
                        ax.text(val, i, f'  {label}', va='center', fontweight='bold')
                    else:
                        ax.text(val, i, f'{label}  ', va='center', ha='right', fontweight='bold')
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                
                charts.append(Image(buf, width=6*inch, height=3*inch))
                charts.append(Spacer(1, 0.15*inch))
        
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ EXIM chart error: {e}")
        
        return charts

    def _format_agent(self, resp, index, styles):
        """Format agent response with better structure"""
        elements = []
        
        agent = resp.get("agent", "Unknown").upper()
        success = resp.get("success", False)
        data = resp.get("data", {})
        
        # Agent header
        elements.append(Paragraph(f"<b>{index}. {agent} Intelligence</b>", styles['CustomHeading2']))
        elements.append(Spacer(1, 0.05*inch))
        
        if success and data:
            # Summary
            summary = data.get("summary", "")
            if summary:
                elements.append(Paragraph(f"<b>Summary:</b> {summary}", styles['CustomBodyText']))
                elements.append(Spacer(1, 0.1*inch))
            
            # Special formatting for Clinical Trials
            if agent == "CLINICAL_TRIALS":
                elements.extend(self._format_clinical_trials_details(data, styles))
            
            # Special formatting for IQVIA
            elif agent == "IQVIA":
                elements.extend(self._format_iqvia_details(data, styles))
            
            # Special formatting for Patents
            elif agent == "PATENT":
                elements.extend(self._format_patent_details(data, styles))
            
            # Special formatting for EXIM
            elif agent == "EXIM":
                elements.extend(self._format_exim_details(data, styles))
            
            else:
                # Generic formatting for other agents
                elements.append(Paragraph("<i>Detailed analysis available in raw data</i>", styles['CustomItalic']))
        else:
            elements.append(Paragraph(
                f"<font color='red'><i>Error: {resp.get('error', 'No data available')}</i></font>",
                styles['CustomItalic']
            ))
        
        elements.append(Spacer(1, 0.15*inch))
        return elements

    def _format_clinical_trials_details(self, data, styles):
        """Format clinical trials with ONLY clean tables - no text duplication"""
        elements = []
        
        # Stats summary
        total = data.get("total_trials_found", 0)
        active = data.get("status_summary", {}).get("active_trials", 0)
        phase_dist = data.get("phase_distribution", {})
        advanced = phase_dist.get("advanced_trials_count", 0)
        
        stats_text = f"<b>Overview:</b> {total} trials found | {active} active | {advanced} in advanced phases (Phase 3/4)"
        elements.append(Paragraph(stats_text, styles['CustomBodyText']))
        elements.append(Spacer(1, 0.15*inch))
        
        # Show ALL trials in tables (no text format)
        trials = data.get("detailed_trials", [])
        if trials:
            elements.append(Paragraph(f"<b>Trial Details ({len(trials)} trials):</b>", styles['CustomHeading3']))
            elements.append(Spacer(1, 0.1*inch))
            
            # Create comprehensive table data with Paragraph objects for wrapping
            table_data = [[
                Paragraph('<b>Trial ID</b>', styles['CustomBodyText']),
                Paragraph('<b>Title</b>', styles['CustomBodyText']),
                Paragraph('<b>Phase</b>', styles['CustomBodyText']),
                Paragraph('<b>Status</b>', styles['CustomBodyText']),
                Paragraph('<b>Sponsor</b>', styles['CustomBodyText']),
                Paragraph('<b>Timeline</b>', styles['CustomBodyText'])
            ]]
            
            for trial in trials:
                title = trial.get('title', 'N/A')
                # Don't truncate - let it wrap
                
                timeline = f"{trial.get('start_date', 'N/A')[:7]} to {trial.get('completion_date', 'N/A')[:7]}"
                
                sponsor = trial.get('sponsor', 'N/A')
                # Don't truncate sponsor either
                
                table_data.append([
                    Paragraph(trial.get('nct_id', 'N/A'), styles['CustomBodyText']),
                    Paragraph(title, styles['CustomBodyText']),
                    Paragraph(trial.get('phase', 'N/A'), styles['CustomBodyText']),
                    Paragraph(trial.get('status', 'N/A'), styles['CustomBodyText']),
                    Paragraph(sponsor, styles['CustomBodyText']),
                    Paragraph(timeline, styles['CustomBodyText'])
                ])
            
            # Create wide table with proper wrapping
            trials_table = Table(table_data, colWidths=[0.8*inch, 2.5*inch, 0.7*inch, 0.9*inch, 1.3*inch, 0.8*inch])
            trials_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 3),
                ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f8ff")]),
            ]))
            
            elements.append(trials_table)
            elements.append(Spacer(1, 0.15*inch))
            
            # Additional details table for interventions/conditions
            if len(trials) > 0:
                elements.append(Paragraph("<b>Trial Interventions & Conditions:</b>", styles['CustomHeading3']))
                elements.append(Spacer(1, 0.05*inch))
                
                detail_data = [[
                    Paragraph('<b>Trial ID</b>', styles['CustomBodyText']),
                    Paragraph('<b>Interventions</b>', styles['CustomBodyText']),
                    Paragraph('<b>Conditions</b>', styles['CustomBodyText'])
                ]]
                
                for trial in trials[:10]:  # Show details for top 10
                    interventions = ', '.join(trial.get('interventions', [])[:3])
                    conditions = ', '.join(trial.get('conditions', [])[:3])
                    
                    detail_data.append([
                        Paragraph(trial.get('nct_id', 'N/A'), styles['CustomBodyText']),
                        Paragraph(interventions if interventions else 'N/A', styles['CustomBodyText']),
                        Paragraph(conditions if conditions else 'N/A', styles['CustomBodyText'])
                    ])
                
                detail_table = Table(detail_data, colWidths=[0.8*inch, 3*inch, 2.4*inch])
                detail_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2ca02c")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('LEFTPADDING', (0, 0), (-1, -1), 3),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#e8f4f8")]),
                ]))
                
                elements.append(detail_table)
                elements.append(Spacer(1, 0.1*inch))
        
        return elements

    def _format_iqvia_details(self, data, styles):
        """Format IQVIA market data - TABLE FORMAT ONLY"""
        elements = []
        
        drug_analyses = data.get("drug_analyses", [])
        if drug_analyses:
            elements.append(Paragraph("<b>Market Analysis Summary:</b>", styles['CustomHeading3']))
            elements.append(Spacer(1, 0.1*inch))
            
            # Comprehensive market table
            table_data = [['Drug', 'Therapeutic Area', 'Sales (USD M)', 'CAGR (%)', 'Trend', 'Prescriptions (M)']]
            
            for drug in drug_analyses:
                metrics = drug.get("market_metrics", {})
                table_data.append([
                    drug.get('drug_name', 'N/A'),
                    drug.get('therapeutic_area', 'N/A'),
                    f"${metrics.get('current_sales_usd_million', 0):.1f}",
                    f"{metrics.get('cagr_percent', 0):.1f}%",
                    metrics.get('market_trend', 'N/A'),
                    f"{metrics.get('current_prescriptions_million', 0):.1f}" if metrics.get('current_prescriptions_million') else 'N/A'
                ])
            
            market_table = Table(table_data, colWidths=[1.2*inch, 1.3*inch, 1*inch, 0.8*inch, 0.9*inch, 1*inch])
            market_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (2, 1), (5, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f8ff")]),
            ]))
            
            elements.append(market_table)
            elements.append(Spacer(1, 0.1*inch))
        
        return elements

    def _format_patent_details(self, data, styles):
        """Format patent landscape data - TABLE FORMAT ONLY"""
        elements = []
        
        total = data.get("total_patents_found", 0)
        expiry = data.get("expiry_timeline", {})
        fto = data.get("fto_assessment", {})
        
        patent_summary = f"<b>Overview:</b> {total} patents analyzed | FTO Risk: {fto.get('risk_level', 'Unknown')} | {expiry.get('expiring_soon_count', 0)} expiring within 2 years"
        elements.append(Paragraph(patent_summary, styles['CustomBodyText']))
        elements.append(Spacer(1, 0.15*inch))
        
        # Patent details table with text wrapping
        detailed_patents = data.get("detailed_patents", [])
        if detailed_patents:
            elements.append(Paragraph(f"<b>Patent Portfolio ({len(detailed_patents)} patents):</b>", styles['CustomHeading3']))
            elements.append(Spacer(1, 0.1*inch))
            
            table_data = [[
                Paragraph('<b>Patent #</b>', styles['CustomBodyText']),
                Paragraph('<b>Title</b>', styles['CustomBodyText']),
                Paragraph('<b>Assignee</b>', styles['CustomBodyText']),
                Paragraph('<b>Grant</b>', styles['CustomBodyText']),
                Paragraph('<b>Expiry</b>', styles['CustomBodyText']),
                Paragraph('<b>Status</b>', styles['CustomBodyText'])
            ]]
            
            for patent in detailed_patents[:15]:  # Show top 15
                title = patent.get('title', 'N/A')
                assignee = patent.get('assignee', 'N/A')
                
                table_data.append([
                    Paragraph(patent.get('patent_number', 'N/A'), styles['CustomBodyText']),
                    Paragraph(title, styles['CustomBodyText']),
                    Paragraph(assignee, styles['CustomBodyText']),
                    Paragraph(patent.get('grant_date', 'N/A')[:10], styles['CustomBodyText']),
                    Paragraph(patent.get('expiry_date', 'N/A')[:10], styles['CustomBodyText']),
                    Paragraph(patent.get('status', 'N/A'), styles['CustomBodyText'])
                ])
            
            patent_table = Table(table_data, colWidths=[0.9*inch, 2.2*inch, 1.3*inch, 0.7*inch, 0.7*inch, 0.7*inch])
            patent_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#d62728")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 3),
                ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#ffe6e6")]),
            ]))
            
            elements.append(patent_table)
            elements.append(Spacer(1, 0.1*inch))
        
        return elements

    def _format_exim_details(self, data, styles):
        """Format EXIM trade data - TABLE FORMAT ONLY"""
        elements = []
        
        drug_analyses = data.get("drug_analyses", [])
        if drug_analyses:
            elements.append(Paragraph("<b>Trade Analysis Summary:</b>", styles['CustomHeading3']))
            elements.append(Spacer(1, 0.1*inch))
            
            # Trade summary table
            table_data = [['Drug', 'Imports (USD)', 'Exports (USD)', 'Balance', 'Records', 'Year']]
            
            for drug in drug_analyses:
                metrics = drug.get("trade_metrics", {})
                table_data.append([
                    drug.get('drug_name', 'N/A'),
                    f"${metrics.get('total_import_value_usd', 0):,.0f}",
                    f"${metrics.get('total_export_value_usd', 0):,.0f}",
                    metrics.get('trade_balance', 'N/A'),
                    str(metrics.get('number_of_trade_records', 0)),
                    metrics.get('data_year', 'N/A')
                ])
            
            trade_table = Table(table_data, colWidths=[1.2*inch, 1.3*inch, 1.3*inch, 1*inch, 0.7*inch, 0.6*inch])
            trade_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2ca02c")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 1), (5, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#e8f5e9")]),
            ]))
            
            elements.append(trade_table)
            elements.append(Spacer(1, 0.15*inch))
            
            # Top trading partners table (if available)
            for drug in drug_analyses[:2]:  # Show partners for top 2 drugs
                partners = drug.get("top_trading_partners", [])
                if partners:
                    elements.append(Paragraph(f"<b>Top Partners for {drug.get('drug_name')}:</b>", styles['CustomHeading3']))
                    elements.append(Spacer(1, 0.05*inch))
                    
                    partner_data = [['Country', 'Trade Value (USD)', 'Flow']]
                    for partner in partners[:5]:
                        partner_data.append([
                            partner.get('country', 'N/A'),
                            f"${partner.get('total_value_usd', 0):,.0f}",
                            partner.get('flow', 'N/A') if 'flow' in partner else 'Mixed'
                        ])
                    
                    partner_table = Table(partner_data, colWidths=[2*inch, 1.5*inch, 1*inch])
                    partner_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#ff7f0e")),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff3e0")]),
                    ]))
                    
                    elements.append(partner_table)
                    elements.append(Spacer(1, 0.1*inch))
        
        return elements

    def _add_page_num(self, canvas, doc):
        """Add page number"""
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(7.5*inch, 0.5*inch, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    def _generate_text(self, filepath, query, agent_responses, synthesized_response):
        """Generate text fallback"""
        try:
            txt_path = filepath.with_suffix(".txt")
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("="*70 + "\n")
                f.write("PHARMACEUTICAL INTELLIGENCE REPORT\n")
                f.write("="*70 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Query: {query}\n\n")
                f.write("="*70 + "\n")
                f.write("EXECUTIVE SUMMARY\n")
                f.write("="*70 + "\n\n")
                f.write(synthesized_response)
                f.write("\n\n")
                f.write("="*70 + "\n")
                f.write("AGENT RESPONSES\n")
                f.write("="*70 + "\n\n")
                
                for i, resp in enumerate(agent_responses, 1):
                    f.write(f"{i}. {resp.get('agent', 'Unknown').upper()}\n")
                    f.write("-"*70 + "\n")
                    if resp.get("success"):
                        f.write(f"Summary: {resp.get('data', {}).get('summary', 'N/A')}\n")
                    else:
                        f.write(f"Error: {resp.get('error', 'Unknown')}\n")
                    f.write("\n")
            
            return txt_path.exists()
        except Exception as e:
            if self.verbose:
                print(f"Text generation error: {e}")
            return False

    def _generate_excel(self, filepath, query, agent_responses, synthesized_response):
        """Generate Excel report"""
        try:
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                # Summary
                pd.DataFrame({
                    "Field": ["Generated", "Query", "Agents"],
                    "Value": [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), query, len(agent_responses)]
                }).to_excel(writer, sheet_name="Summary", index=False)
                
                # Agents
                pd.DataFrame([{
                    "Agent": r.get("agent", "Unknown"),
                    "Status": "Success" if r.get("success") else "Failed",
                    "Summary": str(r.get("data", {}).get("summary", ""))[:500]
                } for r in agent_responses]).to_excel(writer, sheet_name="Agents", index=False)
            
            return filepath.exists()
        except Exception as e:
            if self.verbose:
                print(f" Excel error: {e}")
            return False


# Convenience function
def get_report_generator_agent(verbose: bool = True):
    """Get report generator instance"""
    return ProfessionalReportGenerator(verbose=verbose)


if __name__ == "__main__":
    print("="*60)
    print("TESTING REPORT GENERATOR")
    print("="*60)
    
    agent = get_report_generator_agent(verbose=True)
    
    result = agent.generate_report(
        query="Test Query",
        agent_responses=[{
            "agent": "iqvia",
            "success": True,
            "data": {"summary": "Test summary", "drug_analyses": []}
        }],
        synthesized_response="This is a test report.",
        report_format="pdf"
    )
    
    print("\n" + "="*60)
    print("RESULT:")
    print(json.dumps(result, indent=2))
    print("="*60)