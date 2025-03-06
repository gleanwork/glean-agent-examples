from typing import List, TypedDict, Dict, Any, Optional, Type
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from langchain_glean.retrievers import GleanSearchRetriever
from glean_agent_examples.common import BaseExampleAgent
from glean_agent_examples.common.models import AgentInput
from glean_agent_examples.common.logger import logger


class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str
    tool_input: Optional[str]
    tool_output: Optional[str]

class LangGraphSearchAgent(BaseExampleAgent):
    """LangGraph example using Glean Search Retriever."""

    def __init__(self):
        super().__init__(
            title="LangGraph Agent Protocol Server with Glean Search",
            description="LangGraph agent that uses Glean's search retriever to find information"
        )
        
        self.retriever = GleanSearchRetriever()
        self.agent = self._create_agent_graph()
    
    def search_glean(self, query: str) -> str:
        """Search for information using Glean's search retriever."""
        try:
            docs = self.retriever.invoke(query)
            if not docs:
                return "No relevant documents found."
            
            results = []
            for doc in docs:
                results.append(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}")
            
            return "\n\n".join(results)
        except Exception as e:
            return f"Error searching Glean: {str(e)}"
    
    def _decide_next_step(self, state: AgentState) -> Dict[str, Any]:
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
    
    def _create_response(self, state: AgentState) -> Dict[str, Any]:
        """Create a final response based on the tool output."""
        tool_output = state.get("tool_output", None)
        
        if not tool_output:
            response = AIMessage(content="I couldn't find any relevant information in the company knowledge base. Please try a different query.")
        else:
            response = AIMessage(content=tool_output)
        
        new_messages = state["messages"] + [response]
        
        return {"messages": new_messages, "next": END}
    
    def _glean_search_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for the search_glean function that returns a dict for the state."""
        query = state.get("tool_input", "")
        
        if isinstance(query, str):
            result = self.search_glean(query)  # Call search_glean directly
        else:
            result = f"Error: Expected string query but got {type(query)}"
        
        return {"tool_output": result}
    
    def _create_agent_graph(self):
        """Create the LangGraph agent."""
        builder = StateGraph(AgentState)
        
        builder.add_node("decide", self._decide_next_step)
        builder.add_node("glean_search_tool", self._glean_search_wrapper)
        builder.add_node("create_response", self._create_response)
        
        builder.add_edge("decide", "glean_search_tool")
        builder.add_edge("glean_search_tool", "create_response")
        
        builder.set_entry_point("decide")
        
        return builder.compile()
    
    async def run_agent(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Run the agent with the given input."""
        try:
            logger.start_section("Starting LangGraph Search agent...")
            
            state = {
                "messages": [HumanMessage(content=agent_input.input)],
                "tool_input": None,
                "tool_output": None
            }
            
            try:
                result = self.agent.invoke(state)
                
                final_messages = result["messages"]
                ai_messages = [msg for msg in final_messages if isinstance(msg, AIMessage)]
                
                if ai_messages:
                    output = ai_messages[-1].content
                else:
                    tool_output = result.get("tool_output")
                    if tool_output:
                        output = tool_output
                    else:
                        output = "I processed your request but couldn't generate a proper response."
                        logger.error("No AI message or tool output found in the result")
            except Exception as inner_e:
                logger.error(f"Inner error executing agent: {str(inner_e)}")
                output = f"I encountered an error processing your request: {str(inner_e)}"
            
            logger.success("Workflow completed")
            return {"output": output}
        except Exception as e:
            logger.error(f"Outer error executing agent: {str(e)}")
            return {"output": f"I encountered an error while processing your request: {str(e)}"}
    
    def get_input_model(self) -> Type[BaseModel]:
        """Get the Pydantic model for the agent input."""
        return AgentInput
    
    def get_required_env_vars(self) -> list[str]:
        """Get a list of required environment variables."""
        return ["GLEAN_SUBDOMAIN", "GLEAN_API_TOKEN", "OPENAI_API_KEY"]
