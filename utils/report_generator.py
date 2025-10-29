"""
Report Generator Utilities
Helper functions for creating formatted reports, charts, and tables
Used by the Report Generator Agent
"""

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Table, TableStyle

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def create_chart_from_data(
    data: Dict[str, Any], chart_type: str = "bar"
) -> Optional[bytes]:
    """
    Create a chart from data and return as bytes

    Args:
        data: Dictionary with chart data
        chart_type: Type of chart ('bar', 'line', 'pie')

    Returns:
        Chart image as bytes or None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar" and "x" in data and "y" in data:
            ax.bar(data["x"], data["y"])
            ax.set_xlabel(data.get("xlabel", "X"))
            ax.set_ylabel(data.get("ylabel", "Y"))
            ax.set_title(data.get("title", "Chart"))

        elif chart_type == "line" and "x" in data and "y" in data:
            ax.plot(data["x"], data["y"], marker="o")
            ax.set_xlabel(data.get("xlabel", "X"))
            ax.set_ylabel(data.get("ylabel", "Y"))
            ax.set_title(data.get("title", "Chart"))

        elif chart_type == "pie" and "labels" in data and "values" in data:
            ax.pie(data["values"], labels=data["labels"], autopct="%1.1f%%")
            ax.set_title(data.get("title", "Chart"))

        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        print(f"Error creating chart: {e}")
        return None


def format_table_for_pdf(
    data: List[Dict[str, Any]], max_rows: int = 50
) -> Optional[Table]:
    """
    Format data as a ReportLab table

    Args:
        data: List of dictionaries
        max_rows: Maximum rows to include

    Returns:
        ReportLab Table object or None
    """
    if not REPORTLAB_AVAILABLE or not data:
        return None

    try:
        # Convert to list of lists
        headers = list(data[0].keys())
        rows = [headers]

        for item in data[:max_rows]:
            row = [str(item.get(h, "")) for h in headers]
            rows.append(row)

        # Create table
        table = Table(rows)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        return table

    except Exception as e:
        print(f"Error formatting table: {e}")
        return None


def sanitize_text_for_pdf(text: str) -> str:
    """
    Sanitize text for PDF generation (remove problematic characters)

    Args:
        text: Input text

    Returns:
        Sanitized text
    """
    # Remove or replace problematic characters
    replacements = {
        "<": "&lt;",
        ">": "&gt;",
        "&": "&amp;",
        '"': "&quot;",
        "'": "&apos;",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def create_excel_summary(data: Dict[str, Any]) -> pd.ExcelWriter:
    """
    Create Excel writer with formatted sheets

    Args:
        data: Dictionary with data for Excel

    Returns:
        ExcelWriter object
    """
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine="openpyxl")

    return writer


def extract_metrics_from_response(response: str) -> Dict[str, Any]:
    """
    Extract key metrics from synthesized response text

    Args:
        response: Synthesized response text

    Returns:
        Dictionary with extracted metrics
    """
    metrics = {
        "summary_length": len(response),
        "sections_found": [],
        "numbers_mentioned": [],
    }

    # Extract section headers
    lines = response.split("\n")
    for line in lines:
        if line.startswith("##"):
            metrics["sections_found"].append(line.strip("#").strip())

    # Extract numbers (basic)
    import re

    numbers = re.findall(r"\d+\.?\d*", response)
    metrics["numbers_mentioned"] = numbers[:10]  # First 10 numbers

    return metrics


def generate_filename(prefix: str, extension: str) -> str:
    """
    Generate a unique filename with timestamp

    Args:
        prefix: Filename prefix
        extension: File extension (without dot)

    Returns:
        Generated filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join(c if c.isalnum() else "_" for c in prefix[:30])
    return f"{safe_prefix}_{timestamp}.{extension}"


def format_currency(value: float, currency: str = "USD") -> str:
    """
    Format number as currency

    Args:
        value: Numeric value
        currency: Currency code

    Returns:
        Formatted string
    """
    if currency == "USD":
        return f"${value:,.2f}"
    elif currency == "EUR":
        return f"€{value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format number as percentage

    Args:
        value: Numeric value (0-100)
        decimals: Decimal places

    Returns:
        Formatted string
    """
    return f"{value:.{decimals}f}%"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def extract_agent_summaries(agent_responses: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Extract summaries from each agent response

    Args:
        agent_responses: List of agent response dictionaries

    Returns:
        Dictionary mapping agent names to summaries
    """
    summaries = {}

    for response in agent_responses:
        agent_name = response.get("agent", "Unknown")
        success = response.get("success", False)
        data = response.get("data", {})

        if success and isinstance(data, dict):
            summary = data.get("summary", "")
            if summary:
                summaries[agent_name] = summary
            else:
                # Try to create a basic summary
                summaries[agent_name] = f"Analysis completed for {agent_name}"
        else:
            summaries[agent_name] = f"No data from {agent_name}"

    return summaries


def validate_report_data(data: Dict[str, Any]) -> bool:
    """
    Validate that report data is complete

    Args:
        data: Report data dictionary

    Returns:
        True if valid, False otherwise
    """
    required_keys = ["query", "response", "agent_responses"]

    for key in required_keys:
        if key not in data:
            return False

    return True


# Helper class for report styling
class ReportStyles:
    """Standard styles for reports"""

    @staticmethod
    def get_header_style():
        """Get header paragraph style"""
        if not REPORTLAB_AVAILABLE:
            return None

        return ParagraphStyle(
            "CustomHeader",
            parent=getSampleStyleSheet()["Heading1"],
            fontSize=20,
            textColor=colors.HexColor("#1f77b4"),
            spaceAfter=20,
            alignment=1,  # Center
        )

    @staticmethod
    def get_subheader_style():
        """Get subheader paragraph style"""
        if not REPORTLAB_AVAILABLE:
            return None

        return ParagraphStyle(
            "CustomSubheader",
            parent=getSampleStyleSheet()["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#2ca02c"),
            spaceAfter=12,
        )

    @staticmethod
    def get_body_style():
        """Get body text paragraph style"""
        if not REPORTLAB_AVAILABLE:
            return None

        return ParagraphStyle(
            "CustomBody",
            parent=getSampleStyleSheet()["BodyText"],
            fontSize=11,
            spaceAfter=10,
        )
