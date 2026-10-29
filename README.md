# 🔬 Autonomous AI Research Assistant

An autonomous, multi-agent literature review system powered by **LangGraph**, **LangChain**, **Groq (Llama-3.1-8b-instant)**, and **Streamlit**. 

This assistant automates the end-to-end literature review process—searching academic databases, filtering papers against strict custom criteria, extracting key engineering metrics, and synthesizing structured comparison reports.

---

## 🎯 Problem Solved

Reading through dozens of academic papers during a literature review is time-consuming. Traditional keyword searches return hundreds of results, many of which are irrelevant or off-topic.

This agent acts as an autonomous research pipeline:
1. **Scouts** academic papers via ArXiv.
2. **Triages** papers against user-defined inclusion/exclusion rules using LLM-based structured reasoning.
3. **Extracts** specific metrics, parameters, and findings using Pydantic schemas.
4. **Synthesizes** the filtered literature into an executive Markdown report with comparative tables.

---

## 🏗️ Architecture & Agentic Workflow

Data flows through a state-driven pipeline powered by **LangGraph**:

```text
[User Input] ➔ [Scout Node] ➔ [Triage Node] ➔ [Extractor Node] ➔ [Synthesizer Node] ➔ [Markdown Report]
Scout Node: Fetches raw papers matching the research query via ArXiv API.

Triage Node: Uses ChatGroq with structured outputs (Pydantic) to evaluate paper abstracts against inclusion rules, returning binary True/False decisions with reasoning.

Extractor Node: Dynamically extracts user-specified parameter key-value pairs (e.g., matrix materials, fillers, numerical results) from passing abstracts.

Synthesizer Node: Consolidates all extracted details into an executive summary, cross-paper trends analysis, and side-by-side Markdown comparison table.
📁 Repository Structure
├── schemas.py          # Data structures (Pydantic models & ResearchState)
├── chains.py           # ArXiv search tool & Groq LLM prompt chains
├── graph.py            # LangGraph StateGraph orchestration engine
├── streamlit_app.py    # Interactive Streamlit Web Interface
├── app.ipynb           # Jupyter Notebook for interactive visual execution
├── requirements.txt    # Project dependencies
└── .env                # API Keys (Groq API Key)
🚀 Getting Started
1. Prerequisites
Python 3.10+

Groq API Key (Get a free key from Groq Console)

2. Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/YOUR_USERNAME/genai-research-assistant.git](https://github.com/YOUR_USERNAME/genai-research-assistant.git)
cd genai-research-assistant
Create and activate a virtual environment:

Bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
Set up Environment Variables:
Create a .env file in the root directory and add your Groq API key:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
💻 How to Run
Option A: Streamlit Web UI (Recommended)
Launch the browser application:

Bash
streamlit run streamlit_app.py
Open http://localhost:8501 in your browser, enter your research prompt, and run the agent interactively.

Option B: Interactive Jupyter Notebook
Open app.ipynb in VS Code or Jupyter Lab, configure your inputs in Cell 2, and execute the cells to see formatted Markdown output rendered in real time.

Option C: Terminal Interface
Run the graph directly from your command prompt:
python graph.py
🛠️ Tech Stack
Orchestration: LangGraph, LangChain

LLM Provider: Groq API (llama-3.1-8b-instant)

Data Validation: Pydantic v2

API Integrations: ArXiv Python Client

Frontend UI: Streamlit, IPython Markdown