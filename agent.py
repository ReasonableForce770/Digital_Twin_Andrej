import os
from typing import Dict, TypedDict, Any
from langchain_core.messages import BaseMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY_HERE"  

class AgentState(TypedDict):
    messages: list[BaseMessage]
    grounding_score: float
    retrieved_docs: list[str]
    temperature: float 


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def call_karpathy_twin(state: AgentState) -> Dict[str, Any]:
    user_query = state["messages"][-1].content
    dynamic_temp = state.get("temperature", 0.4)
    
   
    docs_with_scores = vector_db.similarity_search_with_score(user_query, k=2)
    
    context_text = ""
    retrieved_contents = []
    total_score = 0.0
    
    for doc, score in docs_with_scores:
        context_text += f"\n- {doc.page_content}\n"
        retrieved_contents.append(doc.page_content)
        total_score += score
        
    avg_score = total_score / len(docs_with_scores) if docs_with_scores else 1.0
    grounding_pct = max(0.0, min(100.0, (1.0 - (avg_score / 1.5)) * 100))
    
    
    with open("persona.md", "r", encoding="utf-8") as f:
        persona_blueprint = f.read()
        
    system_prompt = f"""{persona_blueprint}\n\nRETRIEVED CONTEXT FROM YOUR PAST LECTURES/BLOGS:\n{context_text}"""
    
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=dynamic_temp)
    
    payload = [AIMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(payload)
    
    return {
        "messages": [AIMessage(content=response.content)],
        "grounding_score": float(round(grounding_pct, 1)),
        "retrieved_docs": retrieved_contents
    }


workflow = StateGraph(AgentState)
workflow.add_node("twin_node", call_karpathy_twin)
workflow.add_edge(START, "twin_node")
workflow.add_edge("twin_node", END)

memory_checkpoint = MemorySaver()
compiled_agent = workflow.compile(checkpointer=memory_checkpoint)
