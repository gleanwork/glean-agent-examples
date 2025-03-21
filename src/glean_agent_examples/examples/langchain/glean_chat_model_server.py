import os
from typing import Dict, Any, Type
from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field

from langchain_glean.chat_models import ChatGlean
from glean_agent_examples.common import BaseExampleServer


class GleanChatInput(BaseModel):
    query: str = Field(description="The question to ask Glean Chat")


class AgentInput(BaseModel):
    input: str


class AgentResponse(BaseModel):
    output: str


class LangchainGleanChatExample(BaseExampleServer):
    """
    LangChain example using Glean Chat Model.
    """

    def __init__(self):
        """Initialize the example."""
        super().__init__(
            title="Langchain Agent Protocol Server with Glean Chat",
            description="LangChain agent that uses Glean's chat model to answer questions"
        )
        
        self.glean_chat = ChatGlean(
            agent_config={"agent": "DEFAULT", "mode": "DEFAULT"}
        )
        
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the LangChain agent."""
        tools = [
            Tool(
                name="glean_chat",
                description="Ask a question to Glean's chat model which has access to your company's knowledge base",
                func=self.ask_glean_chat,
                args_schema=GleanChatInput
            )
        ]
        
        llm = ChatOpenAI(model="gpt-4o-mini")
        llm_with_tools = llm.bind(
            functions=[convert_to_openai_function(t) for t in tools]
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can use Glean's chat model to answer questions about company information. When asked a question, use the Glean Chat tool to get accurate information from your company's knowledge base."),
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
    
    def ask_glean_chat(self, query: str) -> str:
        """Ask a question to Glean's chat model."""
        try:
            response = self.glean_chat.invoke([{"type": "human", "content": query}])
            return response.content
        except Exception as e:
            return f"Error querying Glean Chat: {str(e)}"
    
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
    example = LangchainGleanChatExample()
    example.start_app()
