#!/bin/bash

# Default query if none provided
QUERY=${1:-"What information can you find about machine learning in Glean?"}

# Escape quotes in the query for JSON
ESCAPED_QUERY=$(echo "$QUERY" | sed 's/"/\\"/g')

# Send request to the LangChain agent
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d "{\"input\": \"$ESCAPED_QUERY\", \"conversation_id\": \"curl-test\"}" \
  | jq 