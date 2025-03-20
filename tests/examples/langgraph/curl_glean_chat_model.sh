#!/bin/bash

# Get the query from command line arguments or use a default
QUERY="${@:-What are the company holidays this year?}"

# Make the API request
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d "{\"input\": \"$QUERY\", \"conversation_id\": \"test-conversation\"}" \
  | jq
