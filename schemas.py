from typing import List, Dict, Any, TypedDict, Optional
from pydantic import BaseModel, Field

# 1. Single Paper Representation
class ResearchPaper(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    url: str
    is_relevant: bool = False
    extracted_data: Dict[str, Any] = {}

# 2. Shared Memory for LangGraph
class ResearchState(TypedDict):
    research_query: str                  
    inclusion_criteria: str              
    parameters_to_extract: str
    scouted_papers: List[ResearchPaper]  
    relevant_papers: List[ResearchPaper] 
    final_report: str                    

# 3. LLM Structured Output Models
class TriageDecision(BaseModel):
    is_relevant: bool = Field(description="True if paper meets inclusion criteria, False otherwise.")
    reasoning: str = Field(description="One sentence explanation.")

class ExtractedDetail(BaseModel):
    parameter_name: str = Field(description="Name of parameter requested.")
    value: str = Field(description="Extracted value or 'Not specified'.")

class GeneralExtraction(BaseModel):
    summary: str = Field(description="One-sentence main conclusion.")
    extracted_details: List[ExtractedDetail] = Field(description="Extracted parameter key-value pairs.")
    