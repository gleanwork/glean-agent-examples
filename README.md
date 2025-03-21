# Glean Agent Examples

A collection of examples demonstrating how to integrate Glean with external Agent frameworks.

## Overview

This project demonstrates two key integration patterns using both LangChain and LangGraph frameworks:

1. **Glean Search Retriever**: Enabling agents to search and retrieve information from Glean's knowledge base.
2. **Glean Chat Model**: Allowing agents to interact with Glean's chat capabilities for more conversational responses.

## Architecture

Each example follows a consistent architecture with two main components:

### Server Components (`*_server.py`)

These files contain the core agent logic and expose a FastAPI server with REST endpoints. They:

- Define the agent's behavior and capabilities
- Set up the necessary tools and models
- Expose endpoints for interacting with the agent
- Handle incoming requests and return responses

### Runner Components (`*_runner.py`)

These files provide a convenient way to test and interact with the server components. They:

- Validate connectivity to required services (Glean, OpenAI, etc.)
- Provide a command-line interface for sending queries to the agent
- Handle error reporting and display results in a user-friendly format

This separation makes it easy to both run the agent as a service and to test it directly from the command line.

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for Python package management
- [Go-Task](https://taskfile.dev/) for running commands
- Glean API credentials
- OpenAI API key

## Setup

1. Clone this repository
2. Install uv if you haven't already:

   ```bash
   # macOS with Homebrew
   brew install uv
   
   # Linux/macOS with curl
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows with pip
   pip install uv
   ```

3. Install Go-Task if you haven't already:

   ```bash
   # macOS with Homebrew
   brew install go-task/tap/go-task
   
   # Linux/macOS with curl
   sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d
   
   # Windows with Scoop
   scoop install task
   ```

4. Install dependencies:

   ```bash
   task install
   ```

   This will run `uv pip install -e .`, which automatically creates a virtual environment and installs the project in development mode.

5. Configure your environment variables in `.env`:

   ```bash
   OPENAI_API_KEY=your_openai_api_key
   GLEAN_SUBDOMAIN=your-instance
   GLEAN_API_TOKEN=your_glean_api_token
   # Optional: Act as a specific user for testing
   GLEAN_ACT_AS=user@example.com
   ```

## Running Examples

### Available Examples

To see all available examples and how to use them:

```bash
task list:examples
```

This will show you all available examples, along with usage instructions. The examples are organized by framework (LangChain or LangGraph) and functionality (Search Retriever or Chat Model).

### Running an Example

There are two components used to interact with the examples:

#### 1. Start a Server

Start an example server to expose an API endpoint:

```bash
task serve:example EXAMPLE=langchain/glean_chat_model
```

This starts a FastAPI server on `localhost:8000` that you can interact with via HTTP requests.

#### 2. Run a Query Directly

Run a query against an example:

```bash
task run:example EXAMPLE=langgraph/glean_search_retriever -- "What information can you find about AI in Glean?"
```

This will:

1. First validate connectivity to the Glean API
2. Send your query to the server
3. Display the response

### Example Workflow

A typical workflow might look like this:

1. **Explore available examples**:

   ```bash
   task list:examples
   ```

2. **Start the associated server**:

   ```bash
   task serve:example EXAMPLE=langchain/glean_chat_model
   ```

3. **Run a test query**:

   ```bash
   task run:example EXAMPLE=langchain/glean_chat_model -- "What are the company holidays?"
   ```

## Components

### Framework Implementations

- **LangChain**: Uses the LangChain framework for building agents
  - `glean_search_retriever_server.py`: Implements a search retriever using LangChain
  - `glean_chat_model_server.py`: Implements a chat model using LangChain

- **LangGraph**: Uses the LangGraph framework for building agents with state management
  - `glean_search_retriever_server.py`: Implements a search retriever using LangGraph
  - `glean_chat_model_server.py`: Implements a chat model using LangGraph

### Core Components

- **BaseExampleServer**: Base class that handles common server functionality
- **BaseExampleRunner**: Base class that handles common runner functionality
- **GleanClient**: Client for interacting with the Glean API

## Glean API Integration

This project integrates with two key Glean APIs:

1. **Glean Search API**: Used by the search retriever examples

   - Sends search queries to Glean's search endpoint
   - Processes the results to extract content and metadata
   - Converts Glean search results into document objects for agent use

2. **Glean Chat API**: Used by the chat model examples

   - Sends chat queries to Glean's chat endpoint
   - Leverages Glean's knowledge base for accurate responses
   - Maintains conversation context for multi-turn interactions

## License

[MIT License](LICENSE)
