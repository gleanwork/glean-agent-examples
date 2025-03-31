import os
from typing import Dict, Any, Type
from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field

from langchain_glean.retrievers import GleanSearchRetriever
from glean_agent_examples.common import BaseExampleServer


class GleanSearchInput(BaseModel):
    query: str = Field(description="The search query to submit to Glean")


class AgentInput(BaseModel):
    input: str


class AgentResponse(BaseModel):
    output: str


class LangchainGleanSearchExample(BaseExampleServer):
    """
    LangChain example using Glean Search Retriever.
    """

    def __init__(self):
        """Initialize the example."""
        super().__init__(
            title="Langchain Agent Server with Glean Search",
            description="LangChain agent that uses Glean Search to retrieve information"
        )
        
        self.glean_retriever = GleanSearchRetriever()        
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the LangChain agent."""
        tools = [
            Tool(
                name="glean_search",
                description="Search for information in Glean's knowledge base",
                func=self.glean_search,
                args_schema=GleanSearchInput
            )
        ]
        
        llm = ChatOpenAI(model="gpt-4o-mini")
        llm_with_tools = llm.bind(
            functions=[convert_to_openai_function(t) for t in tools]
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can use tools to retrieve information from Glean. When asked a question, first search Glean for relevant information before providing an answer."),
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
    
    def glean_search(self, query: str) -> str:
        """Search for information in Glean."""
        docs = self.glean_retriever.invoke(query)
        
        if not docs:
            return "No relevant documents found in Glean."
        
        results_text = "\n\n".join([
            f"**{doc.metadata.get('title', 'Untitled')}**\n"
            f"{doc.page_content}\n"
            f"Source: {doc.metadata.get('url', 'Unknown source')}"
            for doc in docs
        ])
        
        return f"Found {len(docs)} relevant documents in Glean:\n\n{results_text}"
    
    async def run_agent(self, agent_input: AgentInput) -> Dict[str, Any]:
        """
        Run the agent with the given input.
        
        Args:
            agent_input: The input to the agent
            
        Returns:
            The agent's response
        """
        try:
            result = self.agent_executor.invoke({"input": agent_input.input})
            return AgentResponse(
                output=result["output"]
            )
        except Exception as e:
            print(f"Error executing agent: {str(e)}")
            return AgentResponse(
                output="I encountered an error while processing your request. Please try again or contact support if the issue persists."
            )
    
    def get_input_model(self) -> Type[BaseModel]:
        """
        Get the Pydantic model for the agent input.
        
        Returns:
            The Pydantic model class for the agent input
        """
        return AgentInput
    
    def get_response_model(self) -> Type[BaseModel]:
        """
        Get the Pydantic model for the agent response.
        
        Returns:
            The Pydantic model class for the agent response
        """
        return AgentResponse
    
    def get_required_env_vars(self) -> list[str]:
        """
        Get a list of required environment variables.
        
        Returns:
            A list of required environment variable names
        """
        return ["GLEAN_SUBDOMAIN", "GLEAN_API_TOKEN", "OPENAI_API_KEY"]


if __name__ == "__main__":
    example = LangchainGleanSearchExample()
    example.start_app()
