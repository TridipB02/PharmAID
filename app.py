"""
Pharma Agentic AI - Enhanced Streamlit Web Interface
Main application file with PROFESSIONAL PDF reports and visualizations
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from agents.master_agent import get_master_agent
from config.settings import TEST_QUERIES, UI_CONFIG
from utils.data_fetchers import get_mock_data_fetcher

# Page configuration
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"],
)

# Custom CSS - Enhanced for professional look
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .query-box {
        background-color: #ffffff;
        color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 2px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .response-box {
        background-color: #ffffff;
        color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .agent-info {
        background-color: #e8f4f8;
        color: #000000;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border: 1px solid #cce0eb;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #155a8a;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-banner {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .download-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if "master_agent" not in st.session_state:
        st.session_state.master_agent = get_master_agent()
    if "mock_fetcher" not in st.session_state:
        st.session_state.mock_fetcher = get_mock_data_fetcher()
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "current_response" not in st.session_state:
        st.session_state.current_response = None


# Main header
def render_header():
    """Render the main header"""
    st.markdown(
        '<h1 class="main-header"> PharmAID</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 1.1rem;">Pharmaceutical Intelligence Platform</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


# Sidebar
def render_sidebar():
    """Render the sidebar with options"""
    st.sidebar.title(" Settings")

    # Sample queries
    st.sidebar.subheader(" Sample Queries")
    st.sidebar.write("Try these example queries:")

    selected_sample = st.sidebar.selectbox(
        "Choose a sample query:", [""] + TEST_QUERIES, key="sample_query_selector"
    )

    if selected_sample and st.sidebar.button("Use This Query"):
        st.session_state.selected_sample_query = selected_sample

    st.sidebar.markdown("---")

    # Query history
    st.sidebar.subheader(" Query History")
    if st.session_state.query_history:
        st.sidebar.write(f"Total queries: {len(st.session_state.query_history)}")
        if st.sidebar.button("Clear History"):
            st.session_state.query_history = []
            st.rerun()
    else:
        st.sidebar.write("No queries yet")

    st.sidebar.markdown("---")

    # System status
    st.sidebar.subheader(" System Status")
    st.sidebar.success(" Ollama Connected")
    st.sidebar.success(" APIs Ready")
    st.sidebar.info(f"Model: llama3.1:8b")


# Main query interface
def render_query_interface():
    """Render the main query interface"""
    st.subheader(" Ask Your Strategic Question")

    # Check if sample query was selected
    default_query = ""
    if hasattr(st.session_state, "selected_sample_query"):
        default_query = st.session_state.selected_sample_query
        del st.session_state.selected_sample_query

    # Query input
    user_query = st.text_area(
        "Enter your query about drug repurposing, market analysis, patents, clinical trials, etc.",
        value=default_query,
        height=100,
        placeholder="Example: What are the repurposing opportunities for Metformin in oncology?",
        key="user_query_input",
    )

    # Submit button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        submit_button = st.button(
            " Submit Query", type="primary", use_container_width=True
        )

    if submit_button and user_query.strip():
        process_query(user_query)


def process_query(user_query: str):
    """Process the user query"""
    with st.spinner(" Processing your query... This may take a moment."):
        try:
            # Get master agent
            master_agent = st.session_state.master_agent

            # Process query
            result = master_agent.process_query(user_query)

            # Store in session state
            st.session_state.current_response = result
            st.session_state.query_history.append(
                {
                    "query": user_query,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "result": result,
                }
            )

            st.success(" Query processed successfully!")

        except Exception as e:
            st.error(f" Error processing query: {str(e)}")
            st.exception(e)


def render_response():
    """Render unified summary - everything in ONE section"""
    if st.session_state.current_response is None:
        # Show helpful information card
        st.markdown("###  Pharmaceutical Intelligence Capabilities")
        
        st.info("""
        **Ask questions about:**
        
        🔹 **Market Dynamics** - Sales trends, CAGR analysis, competitive positioning
        🔹 **Clinical Development** - Pipeline analysis, trial phases, repurposing opportunities
        🔹 **Patent Landscape** - IP protection, expiry timelines, FTO analysis
        🔹 **Global Trade** - Import/export patterns, sourcing strategies
        🔹 **Scientific Evidence** - Literature analysis, treatment guidelines
        """)
        return

    response = st.session_state.current_response

    # =============================================
    # SINGLE "SUMMARY" SECTION WITH EVERYTHING
    # =============================================
    
    st.markdown("##  Summary")
    
    # Show the query
    st.markdown(
        f'<div class="query-box"><strong>Your Query:</strong> {response["query"]}</div>', 
        unsafe_allow_html=True
    )

    # Query analysis (collapsed by default)
    with st.expander(" View Query Analysis", expanded=False):
        parsed = response.get("parsed_query", {})
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Intent:**", parsed.get("intent", "N/A"))
            st.write("**Entities:**", ", ".join(parsed.get("entities", {}).get("drugs", [])) or "None")
        with col2:
            st.write("**Required Agents:**", ", ".join(parsed.get("required_agents", [])))
            st.write("**Keywords:**", ", ".join(parsed.get("keywords", [])[:5]))

    st.markdown("---")

    # =============================================
    # AI TEXT RESPONSE
    # =============================================
    st.markdown(
        f'<div class="response-box">{response["response"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # =============================================
    # INLINE DATA: CHARTS, TABLES, DATABASE INFO
    # =============================================
    
    agent_responses = response.get("agent_responses", [])
    
    for agent_response in agent_responses:
        if not agent_response.get("success"):
            continue
        
        agent = agent_response.get("agent", "")
        data = agent_response.get("data", {})
        
        # Render inline visualizations WITHOUT download buttons
        if agent == "iqvia":
            render_iqvia_inline(data)
        
        elif agent == "clinical_trials":
            render_clinical_trials_inline(data)
        
        elif agent == "patent":
            render_patent_inline(data)
        
        elif agent == "exim":
            render_exim_inline(data)
        
        elif agent == "web_intelligence":
            render_literature_inline(data)

    st.markdown("---")

    # =============================================
    # EXPORT OPTIONS (AT END)
    # =============================================
    st.markdown("###  Export Options")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(" Generate PDF Report", use_container_width=True):
            generate_professional_pdf_report(response)

    with col2:
        if st.button(" Generate Excel", use_container_width=True):
            generate_excel_report(response)

    with col3:
        if st.button(" New Query", use_container_width=True):
            st.session_state.current_response = None
            st.rerun()


def render_iqvia_inline(data: Dict):
    """Inline IQVIA data - charts + table (NO download button)"""
    
    drug_analyses = data.get("drug_analyses", [])
    
    if not drug_analyses:
        return
    
    st.markdown(f"####  Market Analysis ({len(drug_analyses)} Drugs)")
    
    # Create DataFrame
    market_df = pd.DataFrame([{
        "Drug": d.get("drug_name", "Unknown"),
        "Sales (USD M)": d.get("market_metrics", {}).get("current_sales_usd_million", 0),
        "CAGR (%)": d.get("market_metrics", {}).get("cagr_percent", 0),
        "Trend": d.get("market_metrics", {}).get("market_trend", "N/A"),
        "Therapeutic Area": d.get("therapeutic_area", "N/A")
    } for d in drug_analyses])
    
    # Charts side by side
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            market_df,
            x="Drug",
            y="Sales (USD M)",
            title="Sales Comparison",
            color="Trend",
            color_discrete_map={
                "increasing": "#2ecc71",
                "stable": "#f39c12",
                "decreasing": "#e74c3c"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            market_df,
            x="Drug",
            y="CAGR (%)",
            title="Growth Rate",
            color="CAGR (%)",
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Table (NO download button)
    st.dataframe(market_df, use_container_width=True)
    
    st.markdown("---")


def render_clinical_trials_inline(data: Dict):
    """Inline clinical trials - charts + table (NO download button)"""
    
    total = data.get("total_trials_found", 0)
    if total == 0:
        return
    
    st.markdown(f"####  Clinical Trials ({total} Trials)")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Trials", total)
    
    with col2:
        active = data.get("status_summary", {}).get("active_trials", 0)
        st.metric("Active Trials", active)
    
    with col3:
        advanced = data.get("phase_distribution", {}).get("advanced_trials_count", 0)
        st.metric("Advanced Phase", advanced)
    
    # Phase Distribution Chart
    phase_dist = data.get("phase_distribution", {}).get("distribution", {})
    if phase_dist:
        fig = go.Figure(data=[go.Pie(
            labels=list(phase_dist.keys()),
            values=[phase_dist[p]["count"] for p in phase_dist.keys()],
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(title=f"Trial Phase Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Table (NO download button)
    detailed_trials = data.get("detailed_trials", [])
    if detailed_trials:
        trial_df = pd.DataFrame([{
            "NCT ID": t.get("nct_id", "N/A"),
            "Title": t.get("title", "N/A")[:50] + "..." if len(t.get("title", "")) > 50 else t.get("title", "N/A"),
            "Phase": t.get("phase", "N/A"),
            "Status": t.get("status", "N/A"),
            "Sponsor": t.get("sponsor", "N/A")[:30] if len(t.get("sponsor", "")) > 30 else t.get("sponsor", "N/A"),
            "Start Date": t.get("start_date", "N/A")
        } for t in detailed_trials])
        
        st.dataframe(trial_df, use_container_width=True, height=400)
    
    st.markdown("---")


def render_patent_inline(data: Dict):
    """Inline patent data - charts + table (NO download button)"""
    
    total = data.get("total_patents_found", 0)
    if total == 0:
        return
    
    st.markdown(f"####  Patent Landscape ({total} Patents)")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patents", total)
    
    with col2:
        expiring_soon = data.get("expiry_timeline", {}).get("expiring_soon_count", 0)
        st.metric("Expiring Soon", expiring_soon)
    
    with col3:
        fto_risk = data.get("fto_assessment", {}).get("risk_level", "Unknown")
        st.metric("FTO Risk", fto_risk)
    
    with col4:
        active = data.get("fto_assessment", {}).get("active_patents_count", 0)
        st.metric("Active", active)
    
    # Expiry Timeline Chart
    expiry_timeline = data.get("expiry_timeline", {})
    if expiry_timeline:
        timeline_df = pd.DataFrame({
            "Category": ["Expired", "Soon (0-2y)", "Medium (2-5y)", "Long (5+y)"],
            "Count": [
                expiry_timeline.get("expired_count", 0),
                expiry_timeline.get("expiring_soon_count", 0),
                expiry_timeline.get("expiring_medium_count", 0),
                expiry_timeline.get("expiring_long_count", 0)
            ]
        })
        
        fig = px.bar(
            timeline_df,
            x="Category",
            y="Count",
            title="Patent Expiry Timeline",
            color="Count",
            color_continuous_scale="RdYlGn_r"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Table (NO download button)
    detailed_patents = data.get("detailed_patents", [])
    if detailed_patents:
        patent_df = pd.DataFrame([{
            "Patent #": p.get("patent_number", "N/A"),
            "Title": p.get("title", "N/A")[:60] + "..." if len(p.get("title", "")) > 60 else p.get("title", "N/A"),
            "Assignee": p.get("assignee", "N/A")[:30] if len(p.get("assignee", "")) > 30 else p.get("assignee", "N/A"),
            "Filing Date": p.get("filing_date", "N/A"),
            "Expiry Date": p.get("expiry_date", "N/A"),
            "Status": p.get("status", "Unknown")
        } for p in detailed_patents])
        
        st.dataframe(patent_df, use_container_width=True, height=400)
    
    st.markdown("---")


def render_exim_inline(data: Dict):
    """Inline trade data - charts + table (NO download button)"""
    
    drug_analyses = data.get("drug_analyses", [])
    
    if not drug_analyses:
        return
    
    st.markdown(f"####  Trade Analysis ({len(drug_analyses)} Drugs)")
    
    # Create DataFrame
    trade_df = pd.DataFrame([{
        "Drug": d.get("drug_name", "Unknown"),
        "Imports (USD)": d.get("trade_metrics", {}).get("total_import_value_usd", 0),
        "Exports (USD)": d.get("trade_metrics", {}).get("total_export_value_usd", 0),
        "Balance": d.get("trade_metrics", {}).get("trade_balance", "Unknown")
    } for d in drug_analyses])
    
    # Chart
    fig = go.Figure(data=[
        go.Bar(name="Imports", x=trade_df["Drug"], y=trade_df["Imports (USD)"], marker_color="#e74c3c"),
        go.Bar(name="Exports", x=trade_df["Drug"], y=trade_df["Exports (USD)"], marker_color="#2ecc71")
    ])
    
    fig.update_layout(
        title="Import vs Export",
        barmode="group",
        xaxis_title="Drug",
        yaxis_title="Trade Value (USD)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table (NO download button)
    st.dataframe(trade_df, use_container_width=True)
    
    st.markdown("---")


def render_literature_inline(data: Dict):
    """Inline literature data - table only (NO download button)"""
    
    total = data.get("total_publications_found", 0)
    if total == 0:
        return
    
    st.markdown(f"####  Scientific Literature ({total} Publications)")
    
    st.metric("Total Publications", total)
    
    # Table (NO download button)
    detailed_pubs = data.get("detailed_publications", [])
    if detailed_pubs:
        pub_df = pd.DataFrame([{
            "PMID": p.get("pmid", "N/A"),
            "Title": p.get("title", "N/A")[:60] + "..." if len(p.get("title", "")) > 60 else p.get("title", "N/A"),
            "Year": p.get("year", "N/A"),
            "Journal": p.get("journal", "N/A")[:40] if len(p.get("journal", "")) > 40 else p.get("journal", "N/A"),
            "URL": p.get("url", "N/A")
        } for p in detailed_pubs])
        
        st.dataframe(pub_df, use_container_width=True, height=400)
    
    st.markdown("---")

def generate_professional_pdf_report(response: Dict):
    """Generate PROFESSIONAL PDF report with enhanced charts - CLEAN VERSION"""
    with st.spinner(" Generating professional PDF report..."):
        try:
            # Import the ENHANCED report generator
            from agents.report_generator_agent import get_report_generator_agent
            
            # Initialize with verbose=FALSE for clean UI
            report_agent = get_report_generator_agent(verbose=False)
            
            # Generate the professional report
            result = report_agent.generate_report(
                query=response["query"],
                agent_responses=response["agent_responses"],
                synthesized_response=response["response"],
                report_format="pdf",
            )

            if result["success"]:
                # Read the generated PDF
                with open(result["filepath"], "rb") as f:
                    pdf_data = f.read()
                
                # Show simple success message
                st.success(" PDF Report Generated Successfully!")
                
                # Download button only
                st.download_button(
                    label=" Download Professional PDF Report",
                    data=pdf_data,
                    file_name=result["filename"],
                    mime="application/pdf",
                    use_container_width=True,
                )
                
            else:
                st.error(f" Error generating report: {result.get('error', 'Unknown error')}")
            
                with st.expander(" Detailed Error Information"):
                    st.json(result)
                
                with st.expander(" Installation Instructions"):
                    st.code("""
                            pip install reportlab matplotlib seaborn
                            # Or install all requirements:
                            pip install -r requirements.txt
                            """, language="bash")
        
        except Exception as e:
            st.error(f" Error generating report: {str(e)}")
            with st.expander(" Full Error Traceback"):
                import traceback
                st.code(traceback.format_exc())


def generate_excel_report(response: Dict):
    """Generate Excel report"""
    with st.spinner(" Generating Excel report..."):
        from agents.report_generator_agent import get_report_generator_agent

        report_agent = get_report_generator_agent(verbose=False)
        result = report_agent.generate_report(
            query=response["query"],
            agent_responses=response["agent_responses"],
            synthesized_response=response["response"],
            report_format="excel",
        )

        if result["success"]:
            with open(result["filepath"], "rb") as f:
                st.download_button(
                    label="⬇ Download Excel Report",
                    data=f,
                    file_name=result["filename"],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            st.success(f" Excel generated! Size: {result['size_mb']} MB")
        else:
            st.error(f" Error: {result.get('error', 'Unknown error')}")


def render_query_history():
    """Render query history in expandable section"""
    if not st.session_state.query_history:
        return

    st.markdown("---")
    st.markdown("###  Recent Queries")

    recent_queries = st.session_state.query_history[-5:][::-1]

    for i, item in enumerate(recent_queries):
        with st.expander(f" {item['timestamp']} - {item['query'][:60]}..."):
            st.write(f"**Full Query:** {item['query']}")
            st.write(
                f"**Intent:** {item['result']['parsed_query'].get('intent', 'N/A')}"
            )

            if st.button(f"View Full Response", key=f"view_response_{i}"):
                st.session_state.current_response = item["result"]
                st.rerun()


def main():
    """Main application function"""
    initialize_session_state()
    render_header()
    render_sidebar()
    render_query_interface()
    render_response()
    render_query_history()

    # Minimal Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #999; font-size: 0.85rem;">PharmAID - An Agentic AI solution | Powered by Ollama & CrewAI</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()