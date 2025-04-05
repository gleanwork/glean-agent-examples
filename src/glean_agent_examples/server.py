import os
import importlib
import inspect
import pkgutil
from typing import Dict, Type
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.padding import Padding

from glean_agent_examples.common.base_example_agent import BaseExampleAgent

console = Console()

class AgentRequest(BaseModel):
    agent: str
    input: str

class AgentResponse(BaseModel):
    output: str

class AgentServer:
    """Main server that handles all agent requests and auto-discovers available agents."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Glean Agent Examples Server",
            description="Server that runs various Glean agent examples"
        )
        self.agents: Dict[str, Type[BaseExampleAgent]] = {}
        self.agent_instances: Dict[str, BaseExampleAgent] = {}
        
        self._discover_agents()
        self._setup_routes()
    
    def _discover_agents(self) -> None:
        """Auto-discover all available agents in the examples directory."""
        examples_dir = os.path.join(os.path.dirname(__file__), "examples")
        
        for _, name, _ in pkgutil.iter_modules([examples_dir]):
            framework_dir = os.path.join(examples_dir, name)
            if os.path.isdir(framework_dir):
                for _, module_name, _ in pkgutil.iter_modules([framework_dir]):
                    if "_agent" in module_name:
                        try:
                            module = importlib.import_module(f"glean_agent_examples.examples.{name}.{module_name}")
                            
                            for class_name, cls in inspect.getmembers(module, inspect.isclass):
                                if (issubclass(cls, BaseExampleAgent) and 
                                    cls != BaseExampleAgent):
                                    
                                    agent_name = self._class_to_agent_name(class_name)
                                    self.agents[agent_name] = cls
                        except Exception as e:
                            print(f"Error loading module {module_name}: {str(e)}")
    
    def _class_to_agent_name(self, class_name: str) -> str:
        """Convert a class name to an agent name."""
        if class_name.endswith("Agent"):
            class_name = class_name[:-5]
        
        common_terms = ["openai", "langchain", "langgraph", "glean"]
        
        lower_name = class_name.lower()
        for term in common_terms:
            idx = lower_name.find(term)
            if idx != -1:
                class_name = (
                    class_name[:idx] + 
                    term + 
                    class_name[idx + len(term):]
                )
        
        agent_name = ""
        for i, char in enumerate(class_name):
            if i > 0 and char.isupper():
                prev_is_lower = class_name[i-1].islower()
                is_in_common_term = False
                for term in common_terms:
                    if class_name[max(0, i-len(term)):i+1].lower() in term:
                        is_in_common_term = True
                        break
                
                if prev_is_lower and not is_in_common_term:
                    agent_name += "_"
            agent_name += char.lower()
            
        return agent_name
    
    def _setup_routes(self) -> None:
        """Set up the FastAPI routes."""
        
        @self.app.get("/agents")
        async def list_agents():
            return {
                "agents": list(self.agents.keys())
            }
        
        @self.app.post("/agent")
        async def run_agent(request: AgentRequest):
            request_info = f"""[bold]Agent:[/bold] {request.agent}
[bold]Query:[/bold] {request.input}"""
            console.print("\n")
            console.print(Panel(request_info, title="New Request", expand=False))
            
            if request.agent not in self.agents:
                raise HTTPException(status_code=404, detail=f"Agent '{request.agent}' not found")
            
            if request.agent not in self.agent_instances:
                console.print(Padding("[bold blue]→[/bold blue] Initializing agent...", (0, 0, 0, 2)))
                agent_class = self.agents[request.agent]
                self.agent_instances[request.agent] = agent_class()
            
            agent = self.agent_instances[request.agent]
            console.print(Padding("[bold blue]→[/bold blue] Processing request...", (0, 0, 0, 2)))
            
            input_model = agent.get_input_model()(input=request.input)
            
            try:
                result = await agent.run_agent(input_model)
                console.print(Padding("[bold green]✓[/bold green] Request completed successfully", (0, 0, 1, 2)))
                return AgentResponse(output=result["output"])
            except Exception as e:
                error_msg = f"[bold red]✗[/bold red] Error: {str(e)}"
                console.print(Padding(error_msg, (0, 0, 1, 2)))
                raise HTTPException(status_code=500, detail=str(e))
    
    def start(self) -> None:
        """Start the server."""
        import uvicorn
        
        agent_list = "\n".join([f"• {agent}" for agent in sorted(self.agents.keys())])
        
        server_info = f"""[bold]Glean Agent Server[/bold]

Available Agents:
{agent_list}

Server URL: http://{self.host}:{self.port}"""
        
        console.print("\n")
        console.print(Panel(server_info, title="Server Information", expand=False))
        console.print("\n[bold blue]Server Status:[/bold blue] Starting...\n")
        
        log_config = uvicorn.config.LOGGING_CONFIG
        log_config["formatters"]["access"]["fmt"] = "%(message)s"
        log_config["formatters"]["default"]["fmt"] = "%(message)s"
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_config=log_config,
            log_level="warning"
        )

if __name__ == "__main__":
    server = AgentServer()
    server.start() 