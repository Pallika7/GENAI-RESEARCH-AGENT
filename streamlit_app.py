import streamlit as st
import json
from graph import research_app

# Page Configuration
st.set_page_config(
    page_title="Autonomous AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# Application Header
st.title("🔬 Autonomous Literature Synthesis Agent")
st.markdown("""
This multi-agent system automatically scouts academic databases, filters papers against your strict inclusion criteria, 
extracts key parameters, and synthesizes a comprehensive literature report using **LangGraph**.
""")

st.divider()

# User Input Form
with st.form("research_config_form"):
    st.subheader("📋 Research Configuration")
    
    query = st.text_input(
        "1. Research Topic / Search Query",
        placeholder="e.g., boron nitride graphene epoxy composites",
        help="The query passed to ArXiv to search for papers."
    )
    
    criteria = st.text_area(
        "2. Inclusion & Exclusion Criteria",
        placeholder="e.g., Must include physical experimental data on thermal conductivity. Exclude purely theoretical models.",
        help="Rules used by the LLM Triage Node to keep or discard papers."
    )
    
    parameters = st.text_area(
        "3. Parameters to Extract",
        placeholder="e.g., 1. Base Matrix\n2. Fillers\n3. Thermal Conductivity\n4. Key Findings",
        help="Specific details the Extractor Node will look for in each paper."
    )
    
    submit_button = st.form_submit_button("🚀 Run Autonomous Agent", use_container_width=True)

# Execution Logic
if submit_button:
    if not query.strip() or not criteria.strip() or not parameters.strip():
        st.error("⚠️ Please fill out all three configuration fields before running!")
    else:
        # Prepare graph input payload
        input_data = {
            "research_query": query,
            "inclusion_criteria": criteria,
            "parameters_to_extract": parameters,
            "scouted_papers": [],
            "relevant_papers": [],
            "final_report": ""
        }
        
        # Display progress spinner while LangGraph runs
        with st.spinner("🤖 Agent active: Searching ArXiv ➔ Triaging ➔ Extracting Parameters ➔ Synthesizing Report..."):
            try:
                # Invoke the LangGraph workflow compiled in graph.py
                final_state = research_app.invoke(input_data)
                
                st.success("🎉 Literature Synthesis Complete!")
                st.divider()
                
                # Render the final Markdown synthesis report
                st.markdown(final_state["final_report"])
                
                st.divider()
                
                # Expandable view to inspect extracted raw JSON data
                with st.expander("📊 View Analyzed Papers & Extracted Raw Data"):
                    if final_state["relevant_papers"]:
                        for idx, paper in enumerate(final_state["relevant_papers"], 1):
                            st.markdown(f"### Paper {idx}: [{paper.title}]({paper.url})")
                            st.json(paper.extracted_data)
                    else:
                        st.info("No papers passed the triage criteria for this run.")
                        
            except Exception as e:
                st.error(f"❌ An error occurred during pipeline execution: {e}")