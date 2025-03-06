#!/usr/bin/env python
import requests
import json
import sys
import os
from dotenv import load_dotenv

# Import the Glean client for direct testing
from glean_langchain_interop.client import GleanAuth, GleanSession

def test_agent(query):
    """Test the LangChain agent with a query."""
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
        print("   Make sure the server is running with 'task start'.")
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

def test_glean_directly():
    """Test the Glean API directly using the Glean client."""
    load_dotenv()
    
    subdomain = os.getenv("GLEAN_SUBDOMAIN")
    api_key = os.getenv("GLEAN_API_KEY")
    act_as = os.getenv("GLEAN_ACT_AS", "steve.calvert@glean.com")
    
    if not subdomain or not api_key:
        print("\n❌ Error: Missing Glean API credentials in .env file.")
        return False
    
    try:
        print("\n=== Testing Glean Client Directly ===")
        print(f"Glean Subdomain: {subdomain}")
        print(f"Acting as: {act_as}")
        
        # Initialize the Glean client with the subdomain directly
        auth = GleanAuth(
            api_token=api_key,
            subdomain=subdomain,
            act_as=act_as
        )
        client = GleanSession(auth=auth)
        
        # Make a simple search request
        payload = {
            "query": "test",
            "pageSize": 1
        }
        
        print("Making search request...")
        results = client.post("search", json=payload)
        
        # Check if we got results
        if "results" in results:
            result_count = len(results.get("results", []))
            print(f"✅ Glean client connection successful! Got {result_count} results.")
            return True
        else:
            print("⚠️ Glean client connected but no results were returned.")
            print(f"Response: {json.dumps(results, indent=2)}")
            return True  # Still consider this a success since we connected
        
    except Exception as e:
        print(f"\n❌ Error connecting to Glean API: {str(e)}")
        return False

if __name__ == "__main__":
    # First test Glean API directly
    glean_ok = test_glean_directly()
    
    if not glean_ok:
        print("\n⚠️ Warning: Glean API test failed. The agent may not work correctly.")
        proceed = input("Do you want to proceed with testing the agent anyway? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(1)
    
    # Get query from command line argument or use a default
    query = sys.argv[1] if len(sys.argv) > 1 else "What information can you find about machine learning in Glean?"
    
    # Test the agent
    test_agent(query) 