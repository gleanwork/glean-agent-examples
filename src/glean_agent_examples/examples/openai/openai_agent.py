import os
from typing import Dict, Any, Type
from pydantic import BaseModel
from agents import Agent, Runner, trace
from agents.mcp import MCPServerStdio

from glean_agent_examples.common import BaseExampleAgent
from glean_agent_examples.common.models import AgentInput
from glean_agent_examples.common.logger import logger

class OpenAIAgent(BaseExampleAgent):
    """OpenAI Agent example using Glean MCP Server."""

    def __init__(self):
        super().__init__(
            title="OpenAI Agent SDK with Glean MCP",
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
        """Run the agent with the given input."""
        try:
            logger.start_section("Starting OpenAI agent...")
            async with self.mcp_server as _:
                logger.info("Connected to MCP server")
                with trace(workflow_name="Glean OpenAI Agent Example"):
                    logger.info("Running agent workflow...")
                    result = await Runner.run(
                        starting_agent=self.agent,
                        input=agent_input.input
                    )
                    logger.success("Workflow completed")
                    return {"output": result.final_output}
        except Exception as e:
            logger.error(str(e))
            return {"output": "I encountered an error while processing your request. Please try again or contact support if the issue persists."}
    
    def get_input_model(self) -> Type[BaseModel]:
        """Get the Pydantic model for the agent input."""
        return AgentInput
    
    def get_required_env_vars(self) -> list[str]:
        """Get a list of required environment variables."""
        return ["GLEAN_SUBDOMAIN", "GLEAN_API_TOKEN", "OPENAI_API_KEY"] 