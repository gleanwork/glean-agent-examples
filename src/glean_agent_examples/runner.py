#!/usr/bin/env python3

import sys
import argparse
import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.padding import Padding

def run_agent(agent: str, query: str) -> None:
    """
    Run an agent with the given query.
    
    Args:
        agent: Name of the agent to run
        query: Query to send to the agent
    """
    console = Console()
    
    # Print request panel
    request_info = f"""[bold]Agent:[/bold] {agent}
[bold]Query:[/bold] {query}"""
    
    console.print("\n")
    console.print(Panel(request_info, title="Agent Request", expand=False, subtitle_align="left"))
    
    try:
        # Make request to server
        console.print(Padding("[bold blue]→[/bold blue] Sending request to server...", (1, 0, 0, 2)))
        response = requests.post(
            "http://localhost:8000/agent",
            json={
                "agent": agent,
                "input": query
            }
        )
        response.raise_for_status()
        
        # Parse and display response
        result = response.json()
        response_panel = Panel(
            Markdown(result["output"]),
            title="Agent Response",
            expand=False
        )
        console.print("\n")
        console.print(response_panel)
        console.print("\n")
        
    except requests.exceptions.ConnectionError:
        error_msg = """[bold red]Error:[/bold red] Could not connect to server
Make sure the server is running with: task serve"""
        console.print(Padding(error_msg, (1, 0, 1, 2)))
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        try:
            error_data = e.response.json()
            error_msg = f"""[bold red]Error:[/bold red] HTTP {e.response.status_code}
Details: {error_data['detail']}"""
        except:
            error_msg = f"""[bold red]Error:[/bold red] HTTP {e.response.status_code}
Response: {e.response.text}"""
        console.print(Padding(error_msg, (1, 0, 1, 2)))
        sys.exit(1)
    except Exception as e:
        console.print(Padding(f"[bold red]Error:[/bold red] {str(e)}", (1, 0, 1, 2)))
        sys.exit(1)

def list_agents() -> None:
    """
    List all available agents.
    
    Args:
        server_url: URL of the server (default: http://localhost:8000)
    """
    console = Console()
    
    try:
        # Get list of agents from server
        response = requests.get("http://localhost:8000/agents")
        response.raise_for_status()
        
        agents = response.json()["agents"]
        
        if not agents:
            console.print("[yellow]No agents available[/yellow]")
            return
        
        console.print("\n[bold blue]Available Agents:[/bold blue]")
        for agent in agents:
            console.print(f"- {agent}")
        console.print()
        
    except requests.exceptions.ConnectionError:
        console.print("[bold red]Error:[/bold red] Could not connect to server")
        console.print("Make sure the server is running with: python -m glean_agent_examples.server")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Glean agent examples")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    subparsers.add_parser("list", help="List available agents")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run an agent")
    run_parser.add_argument("agent", help="Name of the agent to run")
    run_parser.add_argument("query", help="Query to send to the agent")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_agents()
    elif args.command == "run":
        run_agent(args.agent, args.query)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 