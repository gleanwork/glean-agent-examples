# Glean Agent Examples

A collection of examples demonstrating how to integrate Glean with external Agent frameworks.

## Overview

This project demonstrates two key integration patterns:

1. **LangChain → Glean**: Enabling LangChain agents to search and retrieve information from Glean's knowledge base.
2. **Glean → LangChain**: Allowing Glean users to invoke specialized LangChain agents from within the Glean interface.

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for Python package management
- [Go-Task](https://taskfile.dev/) for running commands
- [ngrok](https://ngrok.com/) for exposing your local server
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
   # Important: Glean API URL must include /api/v1 at the end
   GLEAN_API_URL=https://your-instance.glean.com/api/v1
   GLEAN_API_KEY=your_glean_api_key
   ```

   Note: The Glean API URL must follow the format `https://your-instance.glean.com/api/v1` with `/api/v1` at the end.

## Running Examples

### Available Examples

To see all available examples and how to use them:

```bash
task list:examples
```

This will show you all available example directories and files, along with usage instructions.

### Running an Example

Start an example server:

```bash
task example:serve EXAMPLE=langchain/glean_search_retriever
```

Or use the shorthand format if you want to use the default example:

```bash
task example:serve
# Defaults to langchain/agent
```

### Exposing the Server with ngrok

In a separate terminal, start ngrok to expose your local server:

```bash
task ngrok
```

Take note of the ngrok URL (e.g., `https://abc123.ngrok.io`).

### Running Queries Against the Example

There are several ways to interact with your running example:

#### Run a Query

Run a query against your example:

```bash
task example:run EXAMPLE=langchain/glean_search_retriever -- "What information can you find about AI in Glean?"
```

This will first test your Glean API connection directly before running the query.

#### Using curl

Test the example using curl:

```bash
task example:curl EXAMPLE=langchain/glean_search_retriever -- "What information can you find about AI in Glean?"
```

### Configuring Glean Actions

1. Update the `openapi.yaml` file with your ngrok URL
2. In Glean, create a new Action using the OpenAPI specification from `openapi.yaml`

### Running Both Services

To start both the example server and ngrok in separate terminals:

```bash
task example:start EXAMPLE=langchain/glean_chat_model
```

Or simply use the default example:

```bash
task example:start
```

## Components

- **GleanRetriever**: A LangChain retriever that connects to Glean's search API
- **LangChain Agent**: A FastAPI server implementing the LangChain Agent Protocol
- **OpenAPI Specification**: Defines the API that Glean can call

## Glean API Integration

This project uses the [Glean Search API](https://developers.glean.com/client/operation/search/) to retrieve documents. The integration:

1. Sends search queries to Glean's `/search` endpoint
2. Processes the results to extract content and metadata
3. Converts Glean search results into LangChain Document objects
4. Makes the documents available to LangChain agents

## License

[MIT License](LICENSE)
