#!/usr/bin/env python
import requests
import json
import sys
import os
from dotenv import load_dotenv

from glean_agent_examples.client import GleanAuth, GleanSession

def test_agent(query):
    """Test the LangGraph agent with a query."""
    url = "http://localhost:8000/runs"

    payload = {
        "input": query,
        "conversation_id": "test-conversation"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("\n=== Sending Request ===")
    print(f"Query: {query}")
    print(f"Endpoint: {url}")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        print("\n=== Agent Response ===")
        print(result["output"])
        return result
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the server.")
        print("   Make sure the server is running with 'task example:serve EXAMPLE=langgraph/glean_search_retriever'.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        print(f"   Status Code: {e.response.status_code}")
        try:
            error_data = e.response.json()
            print(f"   Error Details: {json.dumps(error_data, indent=2)}")
        except:
            print(f"   Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}")
        return None

def test_glean_search_directly():
    """Test the Glean Search API directly."""
    load_dotenv()
    
    subdomain = os.getenv("GLEAN_SUBDOMAIN")
    api_key = os.getenv("GLEAN_API_TOKEN")
    act_as = os.getenv("GLEAN_ACT_AS")
    
    if not subdomain or not api_key:
        print("\n❌ Error: Missing Glean API credentials in .env file.")
        return False
    
    try:
        print("\n=== Ensuring direct Glean API connection ===")
        print(f"Glean Subdomain: {subdomain}")
        print(f"Acting as: {act_as}")

        auth = GleanAuth(
            api_token=api_key,
            subdomain=subdomain,
            act_as=act_as
        )
        client = GleanSession(auth=auth)
        
        # Perform a simple search
        query = "company holidays"
        print(f"Making search request for: '{query}'...")
        
        response = client.get("search", params={"q": query, "maxResults": 3})
        
        if "results" in response and len(response.get("results", [])) > 0:
            print("\n✅ Glean Search API connection successful!")
            print(f"\nFound {len(response['results'])} results.")
            
            # Print a preview of the first result
            if response["results"]:
                first_result = response["results"][0]
                title = first_result.get("title", "Untitled")
                snippet = first_result.get("snippet", "No snippet available")
                print(f"\nFirst result: {title}")
                print(f"Snippet: {snippet[:200]}...")
            
            return True
        else:
            print("\n⚠️ Glean Search API connected but no results were returned.")
            print(f"Response: {json.dumps(response, indent=2)}")
            return False
        
    except Exception as e:
        print(f"\n❌ Error connecting to Glean Search API: {str(e)}")
        return False

if __name__ == "__main__":
    # First test Glean Search API directly
    glean_ok = test_glean_search_directly()
    
    if not glean_ok:
        print("\n⚠️ Warning: Glean Search API test failed. The agent may not work correctly.")
        proceed = input("Do you want to proceed with testing the agent anyway? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(1)
    
    # Get query from command line argument or use a default
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the company holidays this year?"
    
    # Test the agent
    test_agent(query)
