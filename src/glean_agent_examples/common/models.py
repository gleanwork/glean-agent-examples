from pydantic import BaseModel, Field

class AgentInput(BaseModel):
    """Common input model for all agents."""
    input: str = Field(description="The question to ask the agent")

class AgentOutput(BaseModel):
    """Common output model for all agents."""
    output: str = Field(description="The agent's response") 