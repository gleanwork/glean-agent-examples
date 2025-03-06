#!/usr/bin/env python
import requests
import json

def query_agent(query, conversation_id=None):
    """Send a query to the LangChain agent."""
    url = "http://localhost:8000/runs"
    
    payload = {
        "input": query,
        "conversation_id": conversation_id
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def main():
    """Interactive test client for the LangChain agent."""
    print("=== LangChain Agent Interactive Test Client ===")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'new' to start a new conversation.")
    
    conversation_id = "interactive-session-" + str(hash(str(requests.get("http://worldtimeapi.org/api/ip").json()["datetime"])))
    
    while True:
        print("\n> ", end="")
        user_input = input().strip()
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        if user_input.lower() == "new":
            conversation_id = "interactive-session-" + str(hash(str(requests.get("http://worldtimeapi.org/api/ip").json()["datetime"])))
            print("Started a new conversation.")
            continue
        
        if not user_input:
            continue
        
        result = query_agent(user_input, conversation_id)
        
        if result:
            print("\nAgent: " + result["output"])
            conversation_id = result.get("conversation_id", conversation_id)

if __name__ == "__main__":
    main() 