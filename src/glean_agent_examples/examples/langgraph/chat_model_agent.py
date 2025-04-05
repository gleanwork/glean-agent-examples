import os
from typing import List, TypedDict, Dict, Any, Optional, Type
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from langchain_glean.chat_models import ChatGlean
from glean_agent_examples.common import BaseExampleAgent


class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str
    tool_input: Optional[str]
    tool_output: Optional[str]


class AgentInput(BaseModel):
    input: str


class AgentResponse(BaseModel):
    output: str


class LangGraphChatAgent(BaseExampleAgent):
    """LangGraph example using Glean Chat Model."""

    def __init__(self):
        super().__init__(
            title="LangGraph Agent Protocol Server with Glean Chat",
            description="LangGraph agent that uses Glean's chat model to answer questions"
        )
        
        self.glean_chat = ChatGlean(
            agent_config={"agent": "DEFAULT", "mode": "DEFAULT"}
        )
        self.agent_graph = self._create_agent_graph()
    
    def ask_glean_chat(self, query: str) -> str:
        """Ask a question to Glean's chat model."""
        try:
            response = self.glean_chat.invoke([HumanMessage(content=query)])
            return response.content
        except Exception as e:
            return f"Error querying Glean Chat: {str(e)}"
    
    def _decide_next_step(self, state: AgentState) -> Dict[str, Any]:
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
    
    def _create_response(self, state: AgentState) -> Dict[str, Any]:
        """Create a final response based on the tool output."""
        tool_output = state.get("tool_output", None)
        
        if not tool_output:
            response = AIMessage(content="I couldn't get a response from the company knowledge base. Please try a different query.")
        else:
            response = AIMessage(content=tool_output)
        
        new_messages = state["messages"] + [response]
        
        return {"messages": new_messages, "next": END}
    
    def _glean_chat_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for the ask_glean_chat function that returns a dict for the state."""
        query = state.get("tool_input", "")
        
        if isinstance(query, str):
            result = self.ask_glean_chat(query)
        else:
            result = f"Error: Expected string query but got {type(query)}"
        
        return {"tool_output": result}
    
    def _create_agent_graph(self):
        """Create the LangGraph agent."""
        builder = StateGraph(AgentState)
        
        builder.add_node("decide", self._decide_next_step)
        builder.add_node("glean_chat_tool", self._glean_chat_wrapper)
        builder.add_node("create_response", self._create_response)
        
        builder.add_edge("decide", "glean_chat_tool")
        builder.add_edge("glean_chat_tool", "create_response")
        
        builder.set_entry_point("decide")
        
        return builder.compile()
    
    async def run_agent(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Run the agent with the given input."""
        try:
            message = HumanMessage(content=agent_input.input)
            
            state = {
                "messages": [message],
                "tool_input": None,
                "tool_output": None
            }
            
            try:
                result = self.agent_graph.invoke(state)
                
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
            
            return {"output": output}
        except Exception as e:
            print(f"Outer error executing agent: {str(e)}")
            return {"output": f"I encountered an error while processing your request: {str(e)}"}
    
    def get_input_model(self) -> Type[BaseModel]:
        """Get the Pydantic model for the agent input."""
        return AgentInput
    
    def get_response_model(self) -> Type[Dict[str, Any]]:
        """Get the type for the agent response."""
        return Dict[str, Any]
    
    def get_required_env_vars(self) -> list[str]:
        """Get a list of required environment variables."""
        return ["GLEAN_SUBDOMAIN", "GLEAN_API_TOKEN"]
