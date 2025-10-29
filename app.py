"""
Pharma Agentic AI - Enhanced Streamlit Web Interface
Main application file with improved visualizations and outputs
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

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #ffffff;
        color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    .response-box {
        background-color: #ffffff;
        color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
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
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
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
        '<h1 class="main-header">💊 Pharma Agentic AI</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #666;">Portfolio Planning Assistant - Drug Repurposing Intelligence</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


# Sidebar
def render_sidebar():
    """Render the sidebar with options"""
    st.sidebar.title("🔧 Configuration")

    # Database stats
    st.sidebar.subheader("📊 Database Info")
    mock_fetcher = st.session_state.mock_fetcher

    if hasattr(mock_fetcher, "drugs_db") and "drugs" in mock_fetcher.drugs_db:
        drug_count = len(mock_fetcher.drugs_db["drugs"])
        st.sidebar.info(f"**Total Drugs:** {drug_count}")

        # Count by therapeutic area
        therapeutic_areas = {}
        for drug in mock_fetcher.drugs_db["drugs"]:
            area = drug.get("therapeutic_area", "Unknown")
            therapeutic_areas[area] = therapeutic_areas.get(area, 0) + 1

        st.sidebar.write("**By Therapeutic Area:**")
        for area, count in sorted(therapeutic_areas.items()):
            st.sidebar.write(f"- {area}: {count}")
    else:
        st.sidebar.warning("Drug database not loaded")

    st.sidebar.markdown("---")

    # Sample queries
    st.sidebar.subheader("📝 Sample Queries")
    st.sidebar.write("Try these example queries:")

    selected_sample = st.sidebar.selectbox(
        "Choose a sample query:", [""] + TEST_QUERIES, key="sample_query_selector"
    )

    if selected_sample and st.sidebar.button("Use This Query"):
        st.session_state.selected_sample_query = selected_sample

    st.sidebar.markdown("---")

    # Query history
    st.sidebar.subheader("📜 Query History")
    if st.session_state.query_history:
        st.sidebar.write(f"Total queries: {len(st.session_state.query_history)}")
        if st.sidebar.button("Clear History"):
            st.session_state.query_history = []
            st.rerun()
    else:
        st.sidebar.write("No queries yet")

    st.sidebar.markdown("---")

    # System status
    st.sidebar.subheader("🚀 System Status")
    st.sidebar.success("✅ Ollama Connected")
    st.sidebar.success("✅ Mock Data Loaded")
    st.sidebar.info(f"Model: llama3.1:8b")


# Main query interface
def render_query_interface():
    """Render the main query interface"""
    st.subheader("🔍 Ask Your Strategic Question")

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
            "🚀 Submit Query", type="primary", use_container_width=True
        )

    if submit_button and user_query.strip():
        process_query(user_query)


def process_query(user_query: str):
    """Process the user query"""
    with st.spinner("🔄 Processing your query... This may take a moment."):
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

            st.success("✅ Query processed successfully!")

        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.exception(e)


def render_response():
    """Render the response from the agent with enhanced visualizations"""
    if st.session_state.current_response is None:
        st.info("👆 Enter a query above to get started!")

        # Show helpful information
        st.markdown("### 💡 What can I help you with?")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            **Market Analysis:**
            - Sales trends and CAGR
            - Competitive landscape
            - Prescription volumes
            - Market opportunities
            """
            )

            st.markdown(
                """
            **Clinical Intelligence:**
            - Ongoing clinical trials
            - Trial phase distribution
            - Sponsor information
            - Pipeline analysis
            """
            )

        with col2:
            st.markdown(
                """
            **Patent Intelligence:**
            - Patent expiry timelines
            - Freedom-to-operate analysis
            - Competitive filings
            - IP opportunities
            """
            )

            st.markdown(
                """
            **Trade Analysis:**
            - Import/export trends
            - Sourcing patterns
            - Price dynamics
            - Global trade insights
            """
            )

        return

    response = st.session_state.current_response

    # Display query
    st.markdown("### 📋 Your Query")
    st.markdown(
        f'<div class="query-box">{response["query"]}</div>', unsafe_allow_html=True
    )

    # Display parsed query info
    with st.expander("🔍 Query Analysis", expanded=False):
        parsed = response.get("parsed_query", {})

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Intent:**", parsed.get("intent", "N/A"))
            st.write(
                "**Entities:**",
                ", ".join(parsed.get("entities", {}).get("drugs", []))
                or "None detected",
            )
        with col2:
            st.write(
                "**Required Agents:**", ", ".join(parsed.get("required_agents", []))
            )
            st.write("**Keywords:**", ", ".join(parsed.get("keywords", [])[:5]))

    # ✨ NEW: ENHANCED VISUALIZATIONS SECTION
    render_visualizations(response)

    # Display response
    st.markdown("### 💬 Response")
    st.markdown(
        f'<div class="response-box">{response["response"]}</div>',
        unsafe_allow_html=True,
    )

    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Generate PDF Report"):
            generate_pdf_report(response)

    with col2:
        if st.button("📊 Generate Excel Data"):
            generate_excel_report(response)

    with col3:
        if st.button("🔄 New Query"):
            st.session_state.current_response = None
            st.rerun()


# ✨ NEW FUNCTION: Enhanced Visualizations
def render_visualizations(response: Dict):
    """Render enhanced visualizations for agent responses"""
    
    st.markdown("### 📊 Data Visualizations")
    
    agent_responses = response.get("agent_responses", [])
    
    # IQVIA Market Analysis Charts
    iqvia_response = next(
        (r for r in agent_responses if r.get("agent") == "iqvia" and r.get("success")),
        None
    )
    
    if iqvia_response:
        render_iqvia_charts(iqvia_response.get("data", {}))
    
    # Clinical Trials Phase Distribution
    ct_response = next(
        (r for r in agent_responses if r.get("agent") == "clinical_trials" and r.get("success")),
        None
    )
    
    if ct_response:
        render_clinical_trials_charts(ct_response.get("data", {}))
    
    # Patent Landscape Visualizations
    patent_response = next(
        (r for r in agent_responses if r.get("agent") == "patent" and r.get("success")),
        None
    )
    
    if patent_response:
        render_patent_charts(patent_response.get("data", {}))
    
    # EXIM Trade Analysis Charts
    exim_response = next(
        (r for r in agent_responses if r.get("agent") == "exim" and r.get("success")),
        None
    )
    
    if exim_response:
        render_exim_charts(exim_response.get("data", {}))


def render_iqvia_charts(data: Dict):
    """Render IQVIA market analysis charts"""
    
    with st.expander("💰 IQVIA Market Analysis", expanded=True):
        drug_analyses = data.get("drug_analyses", [])
        
        if not drug_analyses:
            st.info("No IQVIA market data available")
            return
        
        # Market Size Comparison
        st.markdown("#### Market Size Comparison")
        
        market_data = []
        for drug in drug_analyses:
            metrics = drug.get("market_metrics", {})
            market_data.append({
                "Drug": drug.get("drug_name", "Unknown"),
                "Sales (USD Million)": metrics.get("current_sales_usd_million", 0),
                "CAGR (%)": metrics.get("cagr_percent", 0),
                "Trend": metrics.get("market_trend", "N/A")
            })
        
        if market_data:
            df = pd.DataFrame(market_data)
            
            # Bar chart for sales
            fig_sales = px.bar(
                df,
                x="Drug",
                y="Sales (USD Million)",
                color="Trend",
                title="Current Sales by Drug",
                color_discrete_map={
                    "increasing": "#2ecc71",
                    "stable": "#f39c12",
                    "decreasing": "#e74c3c"
                }
            )
            st.plotly_chart(fig_sales, use_container_width=True)
            
            # CAGR comparison
            fig_cagr = px.bar(
                df,
                x="Drug",
                y="CAGR (%)",
                title="CAGR Comparison",
                color="CAGR (%)",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig_cagr, use_container_width=True)
            
            # Data table
            st.markdown("#### Detailed Market Metrics")
            st.dataframe(df, use_container_width=True)
        
        # Historical Trends
        for drug in drug_analyses[:3]:  # Show trends for first 3 drugs
            historical = drug.get("historical_data", [])
            if historical:
                st.markdown(f"#### {drug.get('drug_name')} - Historical Trend")
                
                trend_df = pd.DataFrame(historical)
                if not trend_df.empty and "year" in trend_df.columns:
                    fig_trend = px.line(
                        trend_df,
                        x="year",
                        y="sales_usd_million",
                        title=f"{drug.get('drug_name')} Sales Trend",
                        markers=True
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)


def render_clinical_trials_charts(data: Dict):
    """Render clinical trials visualizations"""
    
    with st.expander("🔬 Clinical Trials Analysis", expanded=True):
        
        # Phase Distribution Pie Chart
        phase_dist = data.get("phase_distribution", {}).get("distribution", {})
        
        if phase_dist:
            st.markdown("#### Trial Phase Distribution")
            
            phases = list(phase_dist.keys())
            counts = [phase_dist[p]["count"] for p in phases]
            
            fig_phase = go.Figure(data=[go.Pie(
                labels=phases,
                values=counts,
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            
            fig_phase.update_layout(title="Trials by Phase")
            st.plotly_chart(fig_phase, use_container_width=True)
        
        # Status Summary
        status_summary = data.get("status_summary", {}).get("distribution", {})
        
        if status_summary:
            st.markdown("#### Trial Status Distribution")
            
            statuses = list(status_summary.keys())
            status_counts = [status_summary[s]["count"] for s in statuses]
            
            fig_status = px.bar(
                x=statuses,
                y=status_counts,
                title="Trials by Status",
                labels={"x": "Status", "y": "Number of Trials"},
                color=status_counts,
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        # Sponsor Analysis
        sponsor_data = data.get("sponsor_analysis", {})
        top_sponsors = sponsor_data.get("top_sponsors", [])
        
        if top_sponsors:
            st.markdown("#### Top Sponsors")
            
            sponsor_df = pd.DataFrame(top_sponsors[:10])
            
            fig_sponsors = px.bar(
                sponsor_df,
                x="trial_count",
                y="sponsor",
                orientation="h",
                title="Top 10 Sponsors by Trial Count",
                color="trial_count",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_sponsors, use_container_width=True)
        
        # Metrics cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total = data.get("total_trials_found", 0)
            st.markdown(
                f'<div class="metric-card"><h3>{total}</h3><p>Total Trials</p></div>',
                unsafe_allow_html=True
            )
        
        with col2:
            active = data.get("status_summary", {}).get("active_trials", 0)
            st.markdown(
                f'<div class="metric-card"><h3>{active}</h3><p>Active Trials</p></div>',
                unsafe_allow_html=True
            )
        
        with col3:
            phase_dist = data.get("phase_distribution", {})
            advanced = phase_dist.get("advanced_trials_count", 0)
            st.markdown(
                f'<div class="metric-card"><h3>{advanced}</h3><p>Advanced Phase</p></div>',
                unsafe_allow_html=True
            )


def render_patent_charts(data: Dict):
    """Render patent landscape visualizations"""
    
    with st.expander("📜 Patent Landscape", expanded=True):
        
        # Patent Expiry Timeline
        expiry_timeline = data.get("expiry_timeline", {})
        
        if expiry_timeline:
            st.markdown("#### Patent Expiry Timeline")
            
            timeline_data = {
                "Category": [
                    "Expired",
                    "Expiring Soon (0-2 yrs)",
                    "Expiring Medium (2-5 yrs)",
                    "Expiring Long (5+ yrs)"
                ],
                "Count": [
                    expiry_timeline.get("expired_count", 0),
                    expiry_timeline.get("expiring_soon_count", 0),
                    expiry_timeline.get("expiring_medium_count", 0),
                    expiry_timeline.get("expiring_long_count", 0)
                ]
            }
            
            df_timeline = pd.DataFrame(timeline_data)
            
            fig_timeline = px.bar(
                df_timeline,
                x="Category",
                y="Count",
                title="Patent Expiry Distribution",
                color="Count",
                color_continuous_scale="RdYlGn_r"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Competitive Landscape - Top Assignees
        competitive = data.get("competitive_landscape", {})
        top_assignees = competitive.get("top_assignees", [])
        
        if top_assignees:
            st.markdown("#### Competitive Filing Heatmap")
            
            assignee_df = pd.DataFrame(top_assignees[:10])
            
            fig_assignees = px.bar(
                assignee_df,
                x="patent_count",
                y="assignee",
                orientation="h",
                title="Top Patent Holders",
                color="patent_count",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_assignees, use_container_width=True)
        
        # FTO Risk Assessment
        fto = data.get("fto_assessment", {})
        risk_level = fto.get("risk_level", "Unknown")
        
        st.markdown("#### Freedom to Operate Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_color = {
                "Low": "🟢",
                "Medium": "🟡",
                "High": "🔴"
            }.get(risk_level, "⚪")
            
            st.markdown(f"### {risk_color} {risk_level} Risk")
        
        with col2:
            active_count = fto.get("active_patents_count", 0)
            st.metric("Active Patents", active_count)
        
        with col3:
            holders = fto.get("unique_patent_holders", 0)
            st.metric("Patent Holders", holders)
        
        # Patent Status Distribution
        detailed_patents = data.get("detailed_patents", [])
        if detailed_patents:
            status_counts = {}
            for patent in detailed_patents:
                status = patent.get("status", "Unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                fig_status = go.Figure(data=[go.Pie(
                    labels=list(status_counts.keys()),
                    values=list(status_counts.values()),
                    hole=0.4
                )])
                
                fig_status.update_layout(title="Patent Status Distribution")
                st.plotly_chart(fig_status, use_container_width=True)


def render_exim_charts(data: Dict):
    """Render EXIM trade analysis charts"""
    
    with st.expander("🌍 EXIM Trade Analysis", expanded=True):
        
        drug_analyses = data.get("drug_analyses", [])
        
        if not drug_analyses:
            st.info("No EXIM trade data available")
            return
        
        # Import/Export Comparison
        st.markdown("#### Import vs Export Analysis")
        
        trade_data = []
        for drug in drug_analyses:
            metrics = drug.get("trade_metrics", {})
            trade_data.append({
                "Drug": drug.get("drug_name", "Unknown"),
                "Imports (USD)": metrics.get("total_import_value_usd", 0),
                "Exports (USD)": metrics.get("total_export_value_usd", 0),
                "Balance": metrics.get("trade_balance", "Unknown")
            })
        
        if trade_data:
            df_trade = pd.DataFrame(trade_data)
            
            # Grouped bar chart
            fig_trade = go.Figure(data=[
                go.Bar(name="Imports", x=df_trade["Drug"], y=df_trade["Imports (USD)"], marker_color="#e74c3c"),
                go.Bar(name="Exports", x=df_trade["Drug"], y=df_trade["Exports (USD)"], marker_color="#2ecc71")
            ])
            
            fig_trade.update_layout(
                title="Import vs Export Volume",
                barmode="group",
                xaxis_title="Drug",
                yaxis_title="Trade Value (USD)"
            )
            st.plotly_chart(fig_trade, use_container_width=True)
            
            # Trade Balance
            st.markdown("#### Trade Balance by Drug")
            
            balance_colors = df_trade["Balance"].map({
                "Net Exporter": "#2ecc71",
                "Net Importer": "#e74c3c",
                "Balanced": "#f39c12"
            }).fillna("#95a5a6")
            
            fig_balance = go.Figure(data=[go.Bar(
                x=df_trade["Drug"],
                y=df_trade["Exports (USD)"] - df_trade["Imports (USD)"],
                marker_color=balance_colors
            )])
            
            fig_balance.update_layout(
                title="Net Trade Position (Exports - Imports)",
                xaxis_title="Drug",
                yaxis_title="Net Balance (USD)"
            )
            st.plotly_chart(fig_balance, use_container_width=True)
        
        # Top Trading Partners
        for drug in drug_analyses[:3]:
            partners = drug.get("top_trading_partners", [])
            if partners:
                st.markdown(f"#### {drug.get('drug_name')} - Top Trading Partners")
                
                partner_df = pd.DataFrame(partners[:10])
                
                fig_partners = px.bar(
                    partner_df,
                    x="total_value_usd",
                    y="country",
                    orientation="h",
                    title=f"Top Partners for {drug.get('drug_name')}",
                    color="total_value_usd",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_partners, use_container_width=True)


def generate_pdf_report(response: Dict):
    """Generate PDF report with enhanced charts"""
    with st.spinner("Generating PDF report..."):
        from agents.report_generator_agent import get_report_generator_agent

        report_agent = get_report_generator_agent(verbose=False)
        result = report_agent.generate_report(
            query=response["query"],
            agent_responses=response["agent_responses"],
            synthesized_response=response["response"],
            report_format="pdf",
        )

        if result["success"]:
            with open(result["filepath"], "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=f,
                    file_name=result["filename"],
                    mime="application/pdf",
                )
            st.success(f"✅ PDF generated! Size: {result['size_mb']} MB")
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")


def generate_excel_report(response: Dict):
    """Generate Excel report"""
    with st.spinner("Generating Excel report..."):
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
                    label="⬇️ Download Excel",
                    data=f,
                    file_name=result["filename"],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            st.success(f"✅ Excel generated! Size: {result['size_mb']} MB")
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")


def render_query_history():
    """Render query history in expandable section"""
    if not st.session_state.query_history:
        return

    st.markdown("---")
    st.markdown("### 📜 Recent Queries")

    recent_queries = st.session_state.query_history[-5:][::-1]

    for i, item in enumerate(recent_queries):
        with st.expander(f"🕐 {item['timestamp']} - {item['query'][:60]}..."):
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

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.9rem;">'
        "Pharma Agentic AI v2.0 | Enhanced Visualizations | Powered by Ollama & CrewAI | "
        f'Last Updated: {datetime.now().strftime("%Y-%m-%d")}'
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()