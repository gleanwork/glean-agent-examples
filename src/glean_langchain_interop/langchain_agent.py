from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
import os

# Import our Glean retriever
from glean_langchain_interop.glean_retriever import GleanRetriever

# Load environment variables
load_dotenv()

# Initialize the retriever connecting to the real Glean API
glean_retriever = GleanRetriever(
    glean_api_url=os.getenv("GLEAN_API_URL"),
    api_key=os.getenv("GLEAN_API_KEY"),
    max_results=5
)

# Create a tool that uses the retriever
def glean_search(query: str) -> str:
    """Search for information in Glean."""
    docs = glean_retriever.get_relevant_documents(query)
    
    if not docs:
        return "No relevant documents found in Glean."
    
    # Format results for display
    results_text = "\n\n".join([
        f"**{doc.metadata.get('title', 'Untitled')}**\n"
        f"{doc.page_content}\n"
        f"Source: {doc.metadata.get('url', 'Unknown source')}"
        for doc in docs
    ])
    
    return f"Found {len(docs)} relevant documents in Glean:\n\n{results_text}"

# Define input schema for the tool
class GleanSearchInput(BaseModel):
    query: str = Field(description="The search query to submit to Glean")

# Create Tools
tools = [
    Tool(
        name="glean_search",
        description="Search for information in Glean's knowledge base",
        func=glean_search,
        args_schema=GleanSearchInput
    )
]

# Set up the agent
llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools to retrieve information from Glean."),
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

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Set up FastAPI server with LAP endpoints
app = FastAPI(title="Langchain Agent Protocol Server")

class AgentInput(BaseModel):
    input: str
    conversation_id: str = None

class AgentResponse(BaseModel):
    output: str
    conversation_id: str = None

@app.post("/runs", response_model=AgentResponse)
async def create_run(agent_input: AgentInput):
    result = agent_executor.invoke({"input": agent_input.input})
    return AgentResponse(
        output=result["output"],
        conversation_id=agent_input.conversation_id
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)