import os
from typing import Dict, Any, Type
from pydantic import BaseModel
from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.convert_to_openai import convert_to_openai_function

from langchain_glean.retrievers import GleanSearchRetriever
from glean_agent_examples.common import BaseExampleAgent
from glean_agent_examples.examples.langchain.schemas import GleanSearchInput


class AgentInput(BaseModel):
    input: str


class AgentResponse(BaseModel):
    output: str


class LangchainSearchAgent(BaseExampleAgent):
    """LangChain example using Glean Search Retriever."""

    def __init__(self):
        super().__init__(
            title="Langchain Agent Protocol Server with Glean Search",
            description="Langchain agent that uses Glean's search retriever to find information"
        )
        
        self.retriever = GleanSearchRetriever()
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the LangChain agent."""
        tools = [
            Tool(
                name="glean_search",
                description="Search for information in your company's knowledge base",
                func=self.search_glean,
                args_schema=GleanSearchInput
            )
        ]
        
        llm = ChatOpenAI(model="gpt-4o-mini")
        llm_with_tools = llm.bind(
            functions=[convert_to_openai_function(t) for t in tools]
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can search your company's knowledge base using Glean's search retriever. When asked a question, use the Glean Search tool to find relevant information."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", []),
                "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=os.getenv("VERBOSE") == "true"
        )
    
    def search_glean(self, query: str) -> str:
        """Search for information using Glean's search retriever."""
        try:
            docs = self.retriever.get_relevant_documents(query)
            if not docs:
                return "No relevant documents found."
            
            results = []
            for doc in docs:
                results.append(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}")
            
            return "\n\n".join(results)
        except Exception as e:
            return f"Error searching Glean: {str(e)}"
    
    async def run_agent(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Run the agent with the given input."""
        try:
            result = self.agent_executor.invoke({"input": agent_input.input})
            return {"output": result["output"]}
        except Exception as e:
            print(f"Error executing agent: {str(e)}")
            return {"output": "I encountered an error while processing your request. Please try again or contact support if the issue persists."}
    
    def get_input_model(self) -> Type[BaseModel]:
        """Get the Pydantic model for the agent input."""
        return AgentInput
    
    def get_response_model(self) -> Type[Dict[str, Any]]:
        """Get the type for the agent response."""
        return Dict[str, Any]
    
    def get_required_env_vars(self) -> list[str]:
        """Get a list of required environment variables."""
        return ["GLEAN_SUBDOMAIN", "GLEAN_API_TOKEN", "OPENAI_API_KEY"]
