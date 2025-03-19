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

from langchain_glean.chat_models import ChatGlean

load_dotenv()

glean_chat = ChatGlean(
    agent_config={"agent": "DEFAULT", "mode": "DEFAULT"}
)

def ask_glean_chat(query: str) -> str:
    """Ask a question to Glean's chat model."""
    try:
        response = glean_chat.invoke([{"type": "human", "content": query}])
        return response.content
    except Exception as e:
        return f"Error querying Glean Chat: {str(e)}"

class GleanChatInput(BaseModel):
    query: str = Field(description="The question to ask Glean Chat")

tools = [
    Tool(
        name="glean_chat",
        description="Ask a question to Glean's chat model which has access to your company's knowledge base",
        func=ask_glean_chat,
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

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=os.getenv("VERBOSE") == "true")

app = FastAPI(title="Langchain Agent Protocol Server with Glean Chat")

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
    print("Starting LangChain Agent server with Glean Chat integration...")
    print(f"Glean Subdomain: {os.getenv('GLEAN_SUBDOMAIN')}")
    print(f"OpenAI API Key configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
