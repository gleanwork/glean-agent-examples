# Glean-LangChain Interoperability

A proof of concept demonstrating bidirectional integration between Glean and LangChain.

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
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create a virtual environment and install dependencies:

   ```bash
   uv venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   task install
   ```

4. Configure your environment variables in `.env`:

   ```
   OPENAI_API_KEY=your_openai_api_key
   # Important: Glean API URL must include /api/v1 at the end
   GLEAN_API_URL=https://your-instance.glean.com/api/v1
   GLEAN_API_KEY=your_glean_api_key
   ```

   Note: The Glean API URL must follow the format `https://your-instance.glean.com/api/v1` with `/api/v1` at the end.

## Usage

### Running the LangChain Agent Server

Start the LangChain agent server:

```bash
task start
```

### Exposing the Server with ngrok

In a separate terminal, start ngrok to expose your local server:

```bash
task ngrok
```

Take note of the ngrok URL (e.g., `https://abc123.ngrok.io`).

### Testing the Agent

There are several ways to test the agent:

#### Single Query Test

Run a single test query:

```bash
task test -- "What information can you find about AI in Glean?"
```

This will first test your Glean API connection directly before testing the agent.

#### Interactive Test Session

Start an interactive test session:

```bash
task interactive
```

This opens a command-line interface where you can have a conversation with the agent.

#### Using curl

Test the agent using curl:

```bash
task curl -- "What information can you find about AI in Glean?"
```

### Configuring Glean Actions

1. Update the `openapi.yaml` file with your ngrok URL
2. In Glean, create a new Action using the OpenAPI specification from `openapi.yaml`

### Running Both Services

To start both the server and ngrok in separate terminals:

```bash
task dev
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
