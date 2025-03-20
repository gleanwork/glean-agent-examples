from typing import List, TypedDict, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import os

from langchain_glean.retrievers import GleanSearchRetriever

load_dotenv()

glean_retriever = GleanSearchRetriever()

class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str
    tool_input: Optional[str]
    tool_output: Optional[str]

def glean_search(query: str) -> str:
    """Search for information in Glean."""
    try:
        docs = glean_retriever.invoke(query)
        
        if not docs:
            return "No relevant documents found in Glean."
        
        results_text = "\n\n".join([
            f"**{doc.metadata.get('title', 'Untitled')}**\n"
            f"{doc.page_content}\n"
            f"Source: {doc.metadata.get('url', 'Unknown source')}"
            for doc in docs
        ])
        
        return f"Found {len(docs)} relevant documents in Glean:\n\n{results_text}"
    except Exception as e:
        return f"Error searching Glean: {str(e)}"

def decide_next_step(state: AgentState) -> Dict[str, Any]:
    """Decide whether to use a tool or respond."""
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, HumanMessage):
        return {"messages": state["messages"], "next": END}
    
    query = last_message.content
    
    return {
        "messages": state["messages"],
        "next": "glean_search_tool", 
        "tool_input": query
    }

def create_response(state: AgentState) -> Dict[str, Any]:
    """Create a final response based on the tool output."""
    tool_output = state.get("tool_output", None)
    
    if not tool_output:
        response = AIMessage(content="I couldn't find any relevant information in the company knowledge base. Please try a different query.")
    else:
        response = AIMessage(content=f"Here's what I found in the company knowledge base: {tool_output}")
    
    new_messages = state["messages"] + [response]
    
    return {"messages": new_messages, "next": END}

def glean_search_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for the glean_search function that returns a dict for the state."""
    query = state.get("tool_input", "")
    
    if isinstance(query, str):
        result = glean_search(query)
    else:
        result = f"Error: Expected string query but got {type(query)}"
    
    return {"tool_output": result}

def create_agent_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("decide", decide_next_step)
    builder.add_node("glean_search_tool", glean_search_wrapper)
    builder.add_node("create_response", create_response)
    
    builder.add_edge("decide", "glean_search_tool")
    builder.add_edge("glean_search_tool", "create_response")
    
    builder.set_entry_point("decide")
    
    return builder.compile()

agent_graph = create_agent_graph()

app = FastAPI(title="LangGraph Agent Protocol Server with Glean Search")

class AgentInput(BaseModel):
    input: str
    conversation_id: str = None

class AgentResponse(BaseModel):
    output: str
    conversation_id: str = None

@app.post("/runs", response_model=AgentResponse)
async def create_run(agent_input: AgentInput):
    """Execute a LangGraph agent with the given input."""
    try:
        message = HumanMessage(content=agent_input.input)
        
        state = {
            "messages": [message],
            "tool_input": None,
            "tool_output": None
        }
        
        try:
            result = agent_graph.invoke(state)
            
            final_messages = result["messages"]
            ai_messages = [msg for msg in final_messages if isinstance(msg, AIMessage)]
            
            if ai_messages:
                output = ai_messages[-1].content
            else:
                tool_output = result.get("tool_output")
                if tool_output:
                    output = f"Here's what I found: {tool_output}"
                else:
                    output = "I processed your request but couldn't generate a proper response."
                    print("Warning: No AI message or tool output found in the result.")
        except Exception as inner_e:
            print(f"Inner error executing agent: {str(inner_e)}")
            output = f"I encountered an error processing your request: {str(inner_e)}"
        
        return AgentResponse(
            output=output,
            conversation_id=agent_input.conversation_id
        )
    except Exception as e:
        print(f"Outer error executing agent: {str(e)}")
        return AgentResponse(
            output=f"I encountered an error while processing your request: {str(e)}",
            conversation_id=agent_input.conversation_id
        )

if __name__ == "__main__":
    print("Starting LangGraph Agent server with Glean Search integration...")
    print(f"Glean Subdomain: {os.getenv('GLEAN_SUBDOMAIN')}")
    print(f"OpenAI API Key configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
