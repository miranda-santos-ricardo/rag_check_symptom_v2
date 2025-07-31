from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from backend.agents.retriever_tool import retrieve_symptoms_tool

def build_react_agent():
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    tools = [retrieve_symptoms_tool]
    agent = create_react_agent(model=model, tools=tools,
        prompt="You are a medical assistant AI. Use the tool when necessary to gather context.")
    return agent

class ReactDiagnosisAgent:
    def __init__(self):
        self.agent = build_react_agent()

    def suggest_diagnosis(self, user_input: str) -> str:
        state = {"messages": [{"role": "user", "content": user_input}]}
        result = self.agent.invoke(state)
        return result["messages"][-1].content
