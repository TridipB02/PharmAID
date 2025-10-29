"""
Pharma Agentic AI - Streamlit Web Interface
Main application file
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

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

    # Display response


def render_response():
    """Render the response from the agent"""
    if st.session_state.current_response is None:
        st.info("👆 Enter a query above to get started!")

        # Show some helpful information
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

        return  # IMPORTANT: Return here to prevent buttons from showing

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

    with st.expander("💊 Drug Database Context", expanded=False):
        agent_responses = response.get("agent_responses", [])
        db_response = next(
            (r for r in agent_responses if r.get("agent") == "drug_database"), None
        )

        if db_response and db_response.get("success"):
            drug_data_dict = db_response.get("data", {})

            if drug_data_dict:
                st.markdown("#### 🔬 Canonical Drug Information")

                for drug_query, drug_data in drug_data_dict.items():
                    if drug_data:
                        st.markdown(f"**Drug Queried:** {drug_query}")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Canonical Name", drug_data.get("canonical_name", "N/A")
                            )
                            st.write(
                                f"**Drug Class:** {drug_data.get('drug_class', 'N/A')}"
                            )

                        with col2:
                            st.metric(
                                "Synonyms Found", len(drug_data.get("synonyms", []))
                            )
                            st.metric(
                                "Brand Names", len(drug_data.get("brand_names", []))
                            )

                        with col3:
                            atc_codes = drug_data.get("atc_codes", [])
                            st.write(
                                f"**ATC Codes:** {', '.join(atc_codes) if atc_codes else 'N/A'}"
                            )
                            st.write(
                                f"**Therapeutic Area:** {drug_data.get('therapeutic_area', 'N/A')}"
                            )

                        if drug_data.get("mechanism"):
                            st.info(f"**Mechanism:** {drug_data['mechanism']}")

                        with st.expander(
                            f"📋 All Names for {drug_query}", expanded=False
                        ):
                            col_syn, col_brand = st.columns(2)

                            with col_syn:
                                st.write("**Top Synonyms:**")
                                synonyms = drug_data.get("synonyms", [])[:10]
                                if synonyms:
                                    for syn in synonyms:
                                        st.write(f"- {syn}")
                                else:
                                    st.write("No synonyms found")

                            with col_brand:
                                st.write("**Brand Names:**")
                                brands = drug_data.get("brand_names", [])[:10]
                                if brands:
                                    for brand in brands:
                                        st.write(f"- {brand}")
                                else:
                                    st.write("No brand names found")

                        st.markdown("---")
                    else:
                        st.warning(f"No data found for: {drug_query}")
            else:
                st.info("No drugs were queried in this search")
        else:
            st.info("Drug Database Agent did not run for this query")

    # Display tasks
    with st.expander("⚙️ Execution Plan", expanded=False):
        tasks = response.get("tasks", [])
        if tasks:
            for i, task in enumerate(tasks, 1):
                st.markdown(
                    f"""
                <div class="agent-info">
                <strong>Task {i}:</strong> {task['agent'].upper()} Agent<br>
                <strong>Action:</strong> {task['action']}<br>
                <strong>Parameters:</strong> {json.dumps(task.get('params', {}), indent=2)}
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.write("No tasks generated")

    # DEBUG: Show agent responses (what data was actually returned)
    with st.expander("🔍 Debug: Raw Agent Responses", expanded=False):
        agent_responses = response.get("agent_responses", [])
        if agent_responses:
            for i, agent_resp in enumerate(agent_responses, 1):
                agent_name = agent_resp.get("agent", "Unknown").upper()
                success = agent_resp.get("success", False)

                st.write(f"**{i}. {agent_name} Agent**")
                st.write(f"Status: {'✅ Success' if success else '❌ Failed'}")

                if success:
                    data = agent_resp.get("data", {})
                    # Show summary if available
                    if isinstance(data, dict) and "summary" in data:
                        st.write(f"**Summary:** {data['summary']}")

                    # Show key data points
                    st.json(data, expanded=False)
                else:
                    st.error(f"Error: {agent_resp.get('error', 'Unknown error')}")

                st.markdown("---")
        else:
            st.write("No agent responses available")

    # Display response
    st.markdown("### 💬 Response")
    st.markdown(
        f'<div class="response-box">{response["response"]}</div>',
        unsafe_allow_html=True,
    )

    # Action buttons - NOW PROPERLY INDENTED INSIDE THE CONDITIONAL BLOCK
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                from agents.report_generator_agent import get_report_generator_agent

                report_agent = get_report_generator_agent(verbose=False)
                result = report_agent.generate_report(
                    query=st.session_state.current_response["query"],
                    agent_responses=st.session_state.current_response[
                        "agent_responses"
                    ],
                    synthesized_response=st.session_state.current_response["response"],
                    report_format="pdf",
                )

                if result["success"]:
                    # Create download button
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

    with col2:
        if st.button("📊 Generate Excel Data"):
            with st.spinner("Generating Excel report..."):
                from agents.report_generator_agent import get_report_generator_agent

                report_agent = get_report_generator_agent(verbose=False)
                result = report_agent.generate_report(
                    query=st.session_state.current_response["query"],
                    agent_responses=st.session_state.current_response[
                        "agent_responses"
                    ],
                    synthesized_response=st.session_state.current_response["response"],
                    report_format="excel",
                )

                if result["success"]:
                    # Create download button
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

    with col3:
        if st.button("🔄 New Query"):
            st.session_state.current_response = None
            st.rerun()


# Display query history
def render_query_history():
    """Render query history in expandable section"""
    if not st.session_state.query_history:
        return

    st.markdown("---")
    st.markdown("### 📜 Recent Queries")

    # Show last 5 queries
    recent_queries = st.session_state.query_history[-5:][::-1]  # Last 5, reversed

    for i, item in enumerate(recent_queries):
        with st.expander(f"🕐 {item['timestamp']} - {item['query'][:60]}..."):
            st.write(f"**Full Query:** {item['query']}")
            st.write(
                f"**Intent:** {item['result']['parsed_query'].get('intent', 'N/A')}"
            )

            if st.button(f"View Full Response", key=f"view_response_{i}"):
                st.session_state.current_response = item["result"]
                st.rerun()


# Main app
def main():
    """Main application function"""
    # Initialize
    initialize_session_state()

    # Render components
    render_header()
    render_sidebar()
    render_query_interface()
    render_response()
    render_query_history()

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.9rem;">'
        "Pharma Agentic AI v1.0 | Powered by Ollama & CrewAI | "
        f'Last Updated: {datetime.now().strftime("%Y-%m-%d")}'
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
