from typing import List, TypedDict, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import os

from langchain_glean.chat_models import ChatGlean

load_dotenv()

glean_chat = ChatGlean(
    agent_config={"agent": "DEFAULT", "mode": "DEFAULT"}
)

class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str
    tool_input: Optional[str]
    tool_output: Optional[str]

def ask_glean_chat(query: str) -> str:
    """Ask a question to Glean's chat model."""
    try:
        response = glean_chat.invoke([HumanMessage(content=query)])
        return response.content
    except Exception as e:
        return f"Error querying Glean Chat: {str(e)}"

def decide_next_step(state: AgentState) -> Dict[str, Any]:
    """Decide whether to use a tool or respond."""
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, HumanMessage):
        return {"messages": state["messages"], "next": END}
    
    query = last_message.content
    
    return {
        "messages": state["messages"],
        "next": "glean_chat_tool", 
        "tool_input": query
    }

def create_response(state: AgentState) -> Dict[str, Any]:
    """Create a final response based on the tool output."""
    tool_output = state.get("tool_output", None)
    
    if not tool_output:
        response = AIMessage(content="I couldn't get information from Glean. Please try again.")
    else:
        response = AIMessage(content=f"Based on Glean's knowledge: {tool_output}")
    
    new_messages = state["messages"] + [response]
    
    return {"messages": new_messages, "next": END}

def glean_chat_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper for the ask_glean_chat function that returns a dict for the state."""
    query = state.get("tool_input", "")
    
    if isinstance(query, str):
        result = ask_glean_chat(query)
    else:
        result = f"Error: Expected string query but got {type(query)}"
    
    return {"tool_output": result}

def create_agent_graph():
    builder = StateGraph(AgentState)
    
    builder.add_node("decide", decide_next_step)
    builder.add_node("glean_chat_tool", glean_chat_wrapper)
    builder.add_node("create_response", create_response)
    
    builder.add_edge("decide", "glean_chat_tool")
    builder.add_edge("glean_chat_tool", "create_response")
    
    builder.set_entry_point("decide")
    
    return builder.compile()

agent_graph = create_agent_graph()

# Create FastAPI app
app = FastAPI(title="LangGraph Agent Protocol Server with Glean Chat")

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
                    output = f"Based on Glean's knowledge: {tool_output}"
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
    print("Starting LangGraph Agent server with Glean Chat integration...")
    print(f"Glean Subdomain: {os.getenv('GLEAN_SUBDOMAIN')}")
    print(f"OpenAI API Key configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
