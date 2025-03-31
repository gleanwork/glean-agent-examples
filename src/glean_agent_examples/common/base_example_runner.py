import sys
import asyncio
import os
import requests
import json

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from .base_example import BaseExample
from .base_example_server import BaseExampleServer, IconType
from .glean_client import GleanClient, GleanAuth


class BaseExampleRunner(BaseExample, ABC):
    """
    Base runner for Glean agent examples.
    
    This abstract class provides common functionality for running examples and testing them.
    Subclasses must implement the validate_connectivity and test_agent methods.
    """
    
    def __init__(
        self,
        example: BaseExampleServer
    ):
        """
        Initialize the runner.
        
        Args:
            example: The example to run
        """
        self.example = example
        
    def run(self, query: str = "What information can you find about machine learning?", check_env: bool = True, validate_connectivity: bool = True) -> None:
        """
        Run tests against a running example server.
        
        This method assumes the server is already running. It does not start the server.
        
        Args:
            query: The query to test with
            check_env: Whether to check environment variables before running
            validate_connectivity: Whether to validate connectivity to external services
        """
        if check_env:
            missing_vars = self.example.check_environment()
            if missing_vars:
                print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
                print("Please set these variables in your .env file or environment.")
                sys.exit(1)
        
        if validate_connectivity:
            connectivity_ok = self.validate_connectivity()
            if not connectivity_ok:
                print("\n⚠️ Warning: Connectivity validation failed. The example may not work correctly.")
                proceed = input("Do you want to proceed with testing the agent anyway? (y/n): ")
                if proceed.lower() != 'y':
                    sys.exit(1)

        self.run_test_suite(query)
    
    def validate_connectivity(self) -> bool:
        """
        Validate connectivity to external services.
        Template method that handles common validation logic.
        
        This method checks that the example can connect to all required
        external services (e.g., Glean API).
        
        Returns:
            True if all connectivity checks pass, False otherwise
        """

        load_dotenv()
        
        # Check env vars (common to all)
        subdomain = os.getenv("GLEAN_SUBDOMAIN")
        api_key = os.getenv("GLEAN_API_TOKEN")
        act_as = os.getenv("GLEAN_ACT_AS")
        
        if not subdomain or not api_key:
            self.print_message("Error: Missing Glean API credentials in .env file.", IconType.ERROR)
            return False
        
        try:
            self.print_title("Ensuring direct Glean API connection", IconType.API)
            self.print_message(f"Glean Subdomain: {subdomain}")
            self.print_message(f"Acting as: {act_as or 'Not required'}")
            
            auth = GleanAuth(
                api_token=api_key,
                subdomain=subdomain,
                act_as=act_as
            )
            client = GleanClient(auth=auth)
            
            # Call the subclass-specific validation method
            return self._validate_glean_connection(client)
            
        except Exception as e:
            self.print_message(f"Error connecting to Glean API: {str(e)}", IconType.ERROR)
            return False
    
    @abstractmethod
    def _validate_glean_connection(self, client) -> bool:
        """
        Validate connection to Glean using the provided client.
        Subclasses must implement this to perform their specific validation.
        
        Args:
            client: Authenticated GleanClient instance
            
        Returns:
            True if validation succeeds, False otherwise
        """
    
    def test_agent(self, query: str, server_url: str = None) -> Dict[str, Any]:
        """
        Test the agent with a query by making an HTTP request to the server.
        
        Args:
            query: The query to send to the agent
            server_url: The URL of the server to test (optional, uses example.server_url if not provided)
            
        Returns:
            The agent's response, or None if an error occurred
        """
        
        server_url = server_url or self.example.server_url
        url = f"{server_url}/runs"

        payload = {
            "input": query
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        self.print_message(f"Query: {query}")
        self.print_message(f"Server URL: {url}")
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            self.print_title("Agent Response", IconType.AGENT)
            print(result["output"])
            return result
        except requests.exceptions.ConnectionError:
            self.print_message("Error: Could not connect to the server.", IconType.ERROR)
            self.print_message(f"   Make sure the server is running with 'task serve:example EXAMPLE={self._get_example_path()}'")
            return None
        except requests.exceptions.HTTPError as e:
            self.print_message(f"HTTP Error: {e}", IconType.ERROR)
            self.print_message(f"   Status Code: {e.response.status_code}")
            try:
                error_data = e.response.json()
                self.print_message(f"   Error Details: {json.dumps(error_data, indent=2)}")
            except:
                self.print_message(f"   Response: {e.response.text}")
            return None
        except Exception as e:
            self.print_message(f"Unexpected Error: {str(e)}", IconType.ERROR)
            return None
            
    def _get_example_path(self) -> str:
        """
        Get the example path for use in error messages.
        
        Returns:
            A string in the format 'framework/example_name'
        """
        module_path = self.example.__class__.__module__
        parts = module_path.split('.')
        
        if len(parts) >= 4 and parts[0] == 'glean_agent_examples' and parts[1] == 'examples':
            framework = parts[2]
            server_name = parts[3]
            
            if server_name.endswith('_server'):
                example_name = server_name[:-7]
            else:
                example_name = server_name
                
            return f"{framework}/{example_name}"
        
        return "your_framework/your_example"
    
    
    def test(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test the example with the given input directly (without HTTP).
        
        This method is useful for testing examples without starting a server.
        
        Args:
            input_data: The input data for the example
            
        Returns:
            The example's response
        """
        input_model = self.example.get_input_model()(**input_data)

        result = asyncio.run(self.example.run_agent(input_model))
        
        return result
        
    def read_query(self) -> str:
        """
        Read a query from command line arguments.
        
        Returns:
            The query string from sys.argv[1]
            
        Raises:
            ValueError: If no query is provided in command line arguments
        """
        if len(sys.argv) <= 1:
            raise ValueError("No query provided. Please provide a query as a command line argument.")
        return sys.argv[1]

    def run_test_suite(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Run a test suite against a running server.
        
        Args:
            query: The query to test with
            
        Returns:
            The test result, or None if the test failed
        """
        self.print_title("Sending Request to Agent", IconType.AGENT)

        result = self.test_agent(query, self.example.server_url)

        if result:
            self.print_message("Test suite completed successfully!", IconType.SUCCESS)
            return result
        

        self.print_message("Test suite failed.", IconType.ERROR)
        return None
