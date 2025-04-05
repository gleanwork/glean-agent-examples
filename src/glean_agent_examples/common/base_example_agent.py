import os
import uvicorn
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from .base_example import BaseExample, IconType


class BaseExampleAgent(BaseExample, ABC):
    """
    Base class for all Glean agent examples.
    
    This class provides common functionality for setting up and running examples
    that integrate with Glean's API.
    """
    
    def __init__(
        self,
        title: str,
        description: str = "",
        version: str = "0.1.0",
        host: str = "0.0.0.0",
        port: int = 8000,
        load_env: bool = True
    ):
        """
        Initialize the example.
        
        Args:
            title: The title of the example
            description: Optional description of the example
            version: API version
            host: Host to run the server on
            port: Port to run the server on
            load_env: Whether to load environment variables from .env file
        """
        if load_env:
            load_dotenv()
            
        self.title = title
        self.description = description
        self.version = version
        self.host = host
        self.port = port
        
        # Load environment variables if needed
        if load_env:
            self._check_environment()
    
    def _check_environment(self) -> None:
        """Check that all required environment variables are set."""
        missing_vars = self.check_environment()
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    @abstractmethod
    async def run_agent(self, agent_input: Any) -> Dict[str, Any]:
        """
        Run the agent with the given input.
        
        Args:
            agent_input: The input to the agent
            
        Returns:
            The agent's response
        """
        pass
    
    @abstractmethod
    def get_input_model(self) -> Type[BaseModel]:
        """
        Get the Pydantic model for the agent input.
        
        Returns:
            The Pydantic model class for the agent input
        """
        pass
    
    @abstractmethod
    def get_response_model(self) -> Type[BaseModel]:
        """
        Get the Pydantic model for the agent response.
        
        Returns:
            The Pydantic model class for the agent response
        """
        pass
    
    def check_environment(self) -> List[str]:
        """
        Check that all required environment variables are set.
        
        Returns:
            A list of missing environment variables
        """
        required_vars = self.get_required_env_vars()
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        return missing_vars
    
    def get_required_env_vars(self) -> List[str]:
        """
        Get a list of required environment variables.
        
        Returns:
            A list of required environment variable names
        """
        return ["GLEAN_SUBDOMAIN", "GLEAN_API_TOKEN"]
    
    def get_app(self) -> FastAPI:
        """
        Get the FastAPI app.
        
        Returns:
            The FastAPI app
        """
        return self.app
        
    def start_app(self) -> None:
        """
        Start the FastAPI app using uvicorn.
        
        This method starts the server and blocks until it's stopped.
        
        Args:
        """
        
        self.print_startup_info()
        
        uvicorn.run(self.get_app(), host=self.host, port=self.port)
    
    def print_startup_info(self) -> None:
        """
        Print information about the example at startup.
        """
        self.print_title(f"Starting {self.title}", IconType.START)
        self.print_message(f"Server URL: {self.server_url}")
        self.print_message(f"Glean Subdomain: {os.getenv('GLEAN_SUBDOMAIN')}")
        
        if "OPENAI_API_KEY" in self.get_required_env_vars():
            self.print_message(f"OpenAI API Key configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    
    @property
    def server_url(self) -> str:
        """
        Get the full server URL.
        
        Returns:
            The full server URL (e.g., http://0.0.0.0:8000)
        """
        return f"http://{self.host}:{self.port}"
