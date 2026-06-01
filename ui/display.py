from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

THEME = Theme(
    {
        "pokemon.name": "bold cyan",
        "pokemon.type": "bold yellow",
        "pokemon.stat": "green",
        "pokemon.label": "dim white",
        "agent": "bold magenta",
        "user": "bold yellow",
        "verifier": "bold cyan",
        "error": "bold red",
    }
)

console = Console(theme=THEME)


def print_welcome():
    console.print(
        Panel(
            "[bold cyan]  Pokédex AI  [/bold cyan]\n"
            "[dim]Fetcher → Verifier dual-agent | Powered by Ollama + LangGraph[/dim]\n\n"
            "[dim]Commands: 'exit' to quit · 'gen <number>' to set generation · 'verbose' to toggle verifier output[/dim]",
            border_style="cyan",
            expand=False,
        )
    )


def print_user(text: str):
    console.print(f"\n[user]You:[/user] {text}")


def print_agent(text: str):
    console.print(f"\n[agent]Pokédex:[/agent] {text}\n")


def print_verifier(text: str):
    console.print(f"\n[verifier dim][ Verifier ][/verifier dim] {text}\n")


def print_error(text: str):
    console.print(f"[error]Error:[/error] {text}")


def print_status(text: str):
    console.print(f"[dim]  {text}...[/dim]")