from langgraph.graph import StateGraph, END
from backend.agents.retriever_agent import SymptomRetrieverAgent
from backend.agents.react_diagnosis_agent import ReactDiagnosisAgent
from typing import TypedDict, List, Tuple

class SymptomDiagnosisState(TypedDict):
    input: str
    matches: List[Tuple[str, str]]
    diagnosis: str

def run_retriever(state: SymptomDiagnosisState):
    user_input = state["input"]
    retriever = SymptomRetrieverAgent()
    matches = retriever.retrieve_symptoms(user_input)
    return {"matches": matches, "input": user_input}

def run_diagnosis(state: SymptomDiagnosisState):
    user_input = state["input"]
    diagnoser = ReactDiagnosisAgent()
    diagnosis = diagnoser.suggest_diagnosis(user_input)
    return {"diagnosis": diagnosis}

def build_graph():
    builder = StateGraph(SymptomDiagnosisState)
    builder.add_node("retriever", run_retriever)
    builder.add_node("diagnosis", run_diagnosis)
    builder.set_entry_point("retriever")
    builder.add_edge("retriever", "diagnosis")
    builder.add_edge("diagnosis", END)
    app = builder.compile()
    return app
