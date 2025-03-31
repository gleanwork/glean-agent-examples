import os
from typing import Dict, Any, Type
from pydantic import BaseModel, Field
from agents import Agent, Runner, trace
from agents.mcp import MCPServerStdio

from glean_agent_examples.common import BaseExampleServer


class AgentInput(BaseModel):
    input: str = Field(description="The question to ask the agent")


class AgentOutput(BaseModel):
    output: str = Field(description="The agent's response")


class OpenAIGleanAgentExample(BaseExampleServer):
    """
    OpenAI Agent example using Glean MCP Server.
    """

    def __init__(self):
        """Initialize the example."""
        super().__init__(
            title="OpenAI Agent Protocol Server with Glean MCP",
            description="OpenAI agent that uses Glean's MCP server to access company knowledge"
        )
        
        self.agent = None
        self.mcp_server = None

        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the OpenAI agent with Glean MCP server."""
        self.mcp_server = MCPServerStdio(
            params={
                "command": "npx",
                "args": ["-y", "@gleanwork/mcp-server"],
                "env": {
                    "GLEAN_SUBDOMAIN": os.getenv("GLEAN_SUBDOMAIN"),
                    "GLEAN_API_TOKEN": os.getenv("GLEAN_API_TOKEN")
                }
            },
            cache_tools_list=True
        )
        
        self.agent = Agent(
            name="GleanAssistant",
            instructions="""You are a helpful assistant that can use Glean's tools to search and retrieve information 
            from the company's knowledge base. When asked a question, use the available Glean tools to get accurate 
            information.""",
            mcp_servers=[self.mcp_server]
        )
    
    async def run_agent(self, agent_input: AgentInput) -> Dict[str, Any]:
        """
        Run the agent with the given input.
        
        Args:
            agent_input: The input to the agent
            
        Returns:
            The agent's response
        """
        try:
            async with self.mcp_server as _:
                with trace(workflow_name="Glean OpenAI Agent Example"):
                    result = await Runner.run(
                        starting_agent=self.agent,
                        input=agent_input.input
                    )
                    return AgentOutput(
                        output=result.final_output
                    )
        except Exception as e:
            print(f"Error executing agent: {str(e)}")
            return AgentOutput(
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
        return AgentOutput
    
    def get_required_env_vars(self) -> list[str]:
        """
        Get a list of required environment variables.
        
        Returns:
            A list of required environment variable names
        """
        return ["GLEAN_SUBDOMAIN", "GLEAN_API_TOKEN", "OPENAI_API_KEY"]


if __name__ == "__main__":
    example = OpenAIGleanAgentExample()
    example.start_app() 