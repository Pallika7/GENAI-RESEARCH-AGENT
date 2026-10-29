import json
import sys
from langgraph.graph import StateGraph, START, END

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Import shared state and chains from our custom modules
from schemas import ResearchState
from chains import (
    fetch_arxiv_papers, 
    triage_chain, 
    extractor_chain, 
    synthesizer_chain
)

# Define Node 1: Scout
def scout_node(state: ResearchState) -> dict:
    print(f"🔍 [Scout] Searching ArXiv for: '{state['research_query']}'...")
    papers = fetch_arxiv_papers(state['research_query'], max_results=5)
    print(f"   Fetched {len(papers)} raw papers.")
    return {"scouted_papers": papers}

# Define Node 2: Triage
def triage_node(state: ResearchState) -> dict:
    print("\n⚖️ [Triage] Filtering papers against criteria...")
    relevant_list = []
    
    
    for paper in state["scouted_papers"]:
        decision = triage_chain.invoke({
            "criteria": state["inclusion_criteria"],
            "title": paper.title,
            "abstract": paper.abstract
        })
        
        if decision.is_relevant:
            paper.is_relevant = True
            relevant_list.append(paper)
            print(f"   ✅ KEEP: {paper.title[:60]}...")
        else:
            print(f"   ❌ DISCARD: {paper.title[:60]}...")
            
    return {"relevant_papers": relevant_list}

# Define Node 3: Extractor
def extractor_node(state: ResearchState) -> dict:
    print("\n🧪 [Extractor] Extracting parameter details...")
    
    for paper in state["relevant_papers"]:
        try:
            extracted_data = extractor_chain.invoke({
                "parameters_to_extract": state["parameters_to_extract"],
                "title": paper.title,
                "abstract": paper.abstract
            })
            paper.extracted_data = (
                extracted_data.model_dump()
                if hasattr(extracted_data, "model_dump")
                else extracted_data.dict()
            )
            print(f"   Extracted details for: {paper.title[:60]}...")
        except Exception as e:
            print(f"   Extraction error on {paper.title[:60]}: {e}")
            
    return {"relevant_papers": state["relevant_papers"]}

# Define Node 4: Synthesizer
def synthesizer_node(state: ResearchState) -> dict:
    print("\n📝 [Synthesizer] Compiling final synthesis report...")
    
    paper_payloads = []
    for idx, paper in enumerate(state["relevant_papers"], 1):
        payload = f"PAPER {idx}:\nTitle: {paper.title}\nURL: {paper.url}\nExtracted Details: {json.dumps(paper.extracted_data, indent=2)}"
        paper_payloads.append(payload)

    full_text = "\n\n" + ("=" * 40) + "\n\n".join(paper_payloads)

    report_response = synthesizer_chain.invoke({
        "research_query": state["research_query"],
        "extracted_papers_text": full_text
    })

    return {"final_report": report_response.content}

# Assemble StateGraph
workflow = StateGraph(ResearchState)

workflow.add_node("scout", scout_node)
workflow.add_node("triage", triage_node)
workflow.add_node("extractor", extractor_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "scout")
workflow.add_edge("scout", "triage")
workflow.add_edge("triage", "extractor")
workflow.add_edge("extractor", "synthesizer")
workflow.add_edge("synthesizer", END)

# Compile the graph object
research_app = workflow.compile()

# Execution block for terminal testing
if __name__ == "__main__":
    print("🤖 Autonomous Research Agent")
    print("=" * 40)
    
    # Capture dynamic user inputs from terminal
    query = input("1. Enter your research topic/query:\n> ")
    criteria = input("\n2. Enter inclusion/exclusion criteria:\n> ")
    params = input("\n3. Enter specific parameters to extract:\n> ")
    
    dynamic_input = {
        "research_query": query,
        "inclusion_criteria": criteria,
        "parameters_to_extract": params,
        "scouted_papers": [],
        "relevant_papers": [],
        "final_report": ""
    }
    
    print("\n🚀 Executing Research Pipeline...\n")
    result = research_app.invoke(dynamic_input)
    
    print("\n" + "=" * 50)
    print("FINAL SYNTHESIS REPORT")
    print("=" * 50 + "\n")
    print(result["final_report"])