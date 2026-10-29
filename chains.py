import arxiv
import os
from dotenv import load_dotenv
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Import schema models from schemas.py
from schemas import (
    ResearchPaper, 
    TriageDecision, 
    GeneralExtraction
)

load_dotenv()

# 1. Initialize Groq LLM
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)


# 2. ArXiv Scout Tool Function
def fetch_arxiv_papers(query: str, max_results: int = 5) -> List[ResearchPaper]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for result in client.results(search):
        paper = ResearchPaper(
            title=result.title,
            authors=[author.name for author in result.authors],
            abstract=result.summary.replace('\n', ' '),
            url=result.pdf_url
        )
        papers.append(paper)
    return papers

# 3. Triage Chain
triage_prompt = PromptTemplate(
    input_variables=["criteria", "title", "abstract"],
    template="""You are an expert academic research reviewer.
    INCLUSION CRITERIA: {criteria}
    PAPER TITLE: {title}
    ABSTRACT: {abstract}
    Determine if this paper strictly meets the criteria.
    """
)
triage_chain = triage_prompt | llm.with_structured_output(TriageDecision)

# 4. Extractor Chain
extraction_prompt = PromptTemplate(
    input_variables=["parameters_to_extract", "title", "abstract"],
    template="""You are an expert data extractor.
    Extract ONLY the requested parameters below. If not found, set value to 'Not specified'.
    PARAMETERS TO EXTRACT:
    {parameters_to_extract}
    
    PAPER TITLE: {title}
    ABSTRACT: {abstract}
    """
)
extractor_chain = extraction_prompt | llm.with_structured_output(GeneralExtraction)

# 5. Synthesizer Chain
synthesizer_prompt = PromptTemplate(
    input_variables=["research_query", "extracted_papers_text"],
    template="""You are a lead research strategist.
    RESEARCH OBJECTIVE: {research_query}

    Synthesize the following extracted literature data into a Markdown report.
    MUST INCLUDE:
    1. ## Executive Summary
    2. ## Comparative Summary Table
    3. ## Cross-Paper Insights & Trends
    4. ## Actionable Next Steps

    EXTRACTED PAPERS DATA:
    {extracted_papers_text}
    """
)
synthesizer_chain = synthesizer_prompt | llm