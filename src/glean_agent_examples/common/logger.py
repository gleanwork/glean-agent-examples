from rich.console import Console
from rich.padding import Padding

console = Console()

class Logger:
    """Common logger for all Glean agent examples."""
    
    @staticmethod
    def info(message: str) -> None:
        """Log an info message with blue arrow formatting."""
        console.print(Padding(f"[bold blue]→[/bold blue] {message}", (0, 0, 0, 2)))
    
    @staticmethod
    def success(message: str) -> None:
        """Log a success message with green checkmark formatting."""
        console.print(Padding(f"[bold green]✓[/bold green] {message}", (0, 0, 0, 2)))
    
    @staticmethod
    def error(message: str) -> None:
        """Log an error message with red X formatting."""
        console.print(Padding(f"[bold red]✗[/bold red] Error: {message}", (0, 0, 0, 2)))
    
    @staticmethod
    def start_section(message: str) -> None:
        """Log a section start with extra padding above."""
        console.print(Padding(f"[bold blue]→[/bold blue] {message}", (1, 0, 0, 2)))

# Create a singleton instance for easy import
logger = Logger() 