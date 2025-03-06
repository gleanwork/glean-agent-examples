from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
import os

from glean_langchain_interop.retrievers import GleanSearchRetriever

load_dotenv()

glean_retriever = GleanSearchRetriever(
    subdomain=os.getenv("GLEAN_SUBDOMAIN"),
    api_key=os.getenv("GLEAN_API_KEY"),
    max_results=5,
    act_as=os.getenv("GLEAN_ACT_AS")
)

def glean_search(query: str) -> str:
    """Search for information in Glean."""
    docs = glean_retriever.invoke(query)
    
    if not docs:
        return "No relevant documents found in Glean."
    
    results_text = "\n\n".join([
        f"**{doc.metadata.get('title', 'Untitled')}**\n"
        f"{doc.page_content}\n"
        f"Source: {doc.metadata.get('url', 'Unknown source')}"
        for doc in docs
    ])
    
    return f"Found {len(docs)} relevant documents in Glean:\n\n{results_text}"

class GleanSearchInput(BaseModel):
    query: str = Field(description="The search query to submit to Glean")

tools = [
    Tool(
        name="glean_search",
        description="Search for information in Glean's knowledge base",
        func=glean_search,
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

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=os.getenv("VERBOSE") == "true")

app = FastAPI(title="Langchain Agent Protocol Server")

class AgentInput(BaseModel):
    input: str
    conversation_id: str = None

class AgentResponse(BaseModel):
    output: str
    conversation_id: str = None

@app.post("/runs", response_model=AgentResponse)
async def create_run(agent_input: AgentInput):
    """Execute a LangChain agent with the given input."""
    try:
        result = agent_executor.invoke({"input": agent_input.input})
        return AgentResponse(
            output=result["output"],
            conversation_id=agent_input.conversation_id
        )
    except Exception as e:
        print(f"Error executing agent: {str(e)}")
        return AgentResponse(
            output="I encountered an error while processing your request. Please try again or contact support if the issue persists.",
            conversation_id=agent_input.conversation_id
        )

if __name__ == "__main__":
    print("Starting LangChain Agent server with Glean integration...")
    print(f"Glean Subdomain: {os.getenv('GLEAN_SUBDOMAIN')}")
    print(f"OpenAI API Key configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)