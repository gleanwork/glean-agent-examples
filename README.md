# Glean Agent Examples

Example implementations showing how to integrate AI agent frameworks with Glean to access your company's knowledge base.

## Overview

This repository demonstrates how to integrate different AI agent frameworks with Glean. These examples show how to:

- Search and retrieve information from your Glean knowledge base
- Use Glean's chat capabilities for natural language interactions

## Available Integrations

### LangChain

Uses LangChain's built-in Glean integrations:

- `GleanSearchRetriever` for semantic search over your knowledge base
- `ChatGlean` for conversational interactions
- Example shows both standalone usage and as part of an agent

### LangGraph

Shows how to build stateful agents with Glean:

- Maintains conversation context across interactions
- Demonstrates workflow control with Glean's search and chat
- Handles complex multi-step queries

### OpenAI Assistants

Shows how to use Glean with OpenAI's Assistant framework:

- Uses Glean's MCP server to expose search and chat as Assistant tools
- Demonstrates function calling with Glean's capabilities
- Shows how to maintain context in conversations

## Getting Started

1. **Prerequisites**
   - Python 3.13+
   - Glean API credentials (subdomain and API token)
   - OpenAI API key (for OpenAI and LangChain examples)

2. **Installation**

   First, install [Go-Task](https://taskfile.dev/installation/):

   ```bash
   # macOS with Homebrew
   brew install go-task/tap/go-task
   
   # Linux/macOS with curl
   sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d
   
   # Windows with Scoop
   scoop install task
   ```

   Then install the package and set up your environment:

   ```bash
   # Install dependencies
   task install
   
   # Set up environment variables
   export GLEAN_SUBDOMAIN=your-instance
   export GLEAN_API_TOKEN=your_token
   export OPENAI_API_KEY=your_key  # If using OpenAI examples
   ```

3. **Running Examples**

   First, start the server:

   ```bash
   # Start the server
   task serve
   ```

   Then in a new terminal, you can run examples:

   ```bash
   # List available examples
   task list

   # Run a specific example
   task run -- langchain_search "What are our company policies?"
   ```

   The server will handle requests from multiple examples, so you only need to start it once.

## Architecture

Each example follows a consistent pattern:

1. Authenticates with Glean using your credentials
2. Sets up the agent framework with appropriate tools/capabilities
3. Handles queries by searching or chatting through Glean
4. Returns formatted responses

The examples are structured to be:

- Easy to understand and modify
- Production-ready with proper error handling
- Maintainable with clear separation of concerns

## API Documentation

- [Glean API Documentation](https://developers.glean.com/docs/)
- [LangChain Glean Integration](https://python.langchain.com/docs/integrations/retrievers/glean)
- [OpenAI Assistants API](https://platform.openai.com/docs/assistants/overview)
