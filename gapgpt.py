#!/usr/bin/env python3
"""GapGPT Chatbot CLI"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text
from typing import Optional, List, Dict
import json
from pathlib import Path
from datetime import datetime
import os
import platform
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.history import FileHistory
import re
import webbrowser


def clear_screen():
    """Clear terminal screen (cross-platform)"""
    if platform.system() == "Windows":
        os.system("cls")
    else:  # Linux, Mac, etc.
        os.system("clear")


def show_banner():
    """Display cool ASCII art banner"""
    banner = """
    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║    ██████╗  █████╗ ██████╗  ██████╗ ██████╗ ████████╗
    ║    ██╔════╝ ██╔══██╗██╔══██╗██╔════╝ ██╔══██╗╚══██╔══╝
    ║    ██║  ███╗███████║██████╔╝██║  ███╗██████╔╝   ██║   
    ║    ██║   ██║██╔══██║██╔═══╝ ██║   ██║██╔═══╝    ██║   
    ║    ╚██████╔╝██║  ██║██║     ╚██████╔╝██║        ██║   
    ║     ╚═════╝ ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝        ╚═╝   
    ║                                                  ║
    ║           AI Chatbot for Terminal                ║
    ╚══════════════════════════════════════════════════╝
    """
    console = Console()
    console.print(banner, style="bold green")


"""Import openai"""
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI not installed. run pip install openai")
    exit(1)

app = typer.Typer(help="GapGPT CLI")
console = Console()

# Define key bindings
bindings = KeyBindings()


@bindings.add("c-c")  # Ctrl+C
def exit_binding(event):
    """Exit on Ctrl+C"""
    print("\n👋 Goodbye!\n")
    event.app.exit()


@bindings.add("c-l")  # Ctrl+L
def clear_screen_binding(event):
    """Clear screen on Ctrl+L"""
    clear_screen()
    event.app.exit()


class App:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.model = model
        self.api_key = api_key

        # Get API key from various sources if not provided
        if not self.api_key:
            self.api_key = self._load_api_key()

        # Check API Key
        if not self.api_key:
            console.print("[red]❌ No API key found![/red]")
            console.print("Set it with:")
            console.print("  gapgpt config --set-key 'your-api-key'  # Save globally")
            console.print("  gapgpt chat --api-key 'your-api-key'    # One time use")
            raise typer.Exit(1)

        # Initialize client with GapGPT base URL
        self.client = OpenAI(
            base_url="https://api.gapgpt.app/v1", api_key=str(self.api_key)
        )
        self.conversation_history: List[Dict[str, str]] = []
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def get_config_path():
        """Get the global config file path (cross-platform)"""
        # Use standard config directory
        if platform.system() == "Windows":
            config_dir = (
                Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
                / "gapgpt"
            )
        else:  # Linux, Mac
            config_dir = Path.home() / ".config" / "gapgpt"

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "config.json"

    @staticmethod
    def load_global_config():
        """Load global configuration"""
        config_file = App.get_config_path()

        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def save_global_config(config: dict):
        """Save global configuration"""
        config_file = App.get_config_path()
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        console.print(f"[green]✅ Configuration saved to {config_file}[/green]")

    def _load_api_key_from_global_config(self):
        """Load API key from global config"""
        config = self.load_global_config()
        return config.get("api_key")

    def _load_api_key(self):
        """Load API key from multiple sources (priority order)"""

        # Priority 1: Global config
        global_key = self._load_api_key_from_global_config()
        if global_key:
            return global_key

        # Priority 2: Environment variable
        env_key = os.getenv("GAPGPT_API_KEY")
        if env_key:
            return env_key

        return None

    def _add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def _get_response(self):
        """Get response from GapGPT and yield chunks as they arrive"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model, messages=self.conversation_history, stream=True
            )

            for chunk in stream:
                try:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        yield content
                except (AttributeError, IndexError):
                    # skips chunk without content
                    continue

        except Exception as e:
            yield f"❌ Error: {str(e)}"

    def _clear_history(self):
        """Reset conversation"""
        self.conversation_history = []
        console.print("[yellow]🧹 Conversation history cleared[/yellow]")

    def save_conversation(self):
        """Save conversation to a json file"""
        filename = f"chat_{self.conversation_id}.json"
        with open(filename, "w") as f:
            json.dump(self.conversation_history, f, indent=2)
        console.print(f"[green]💾 Saved to {filename}[/green]")

    def load_conversation(self, filename: str):
        """Load conversation from json file"""
        try:
            with open(filename, "r") as f:
                self.conversation_history = json.load(f)
            console.print(
                f"[green]📂 Loaded {len(self.conversation_history)} messages[/green]"
            )
        except FileNotFoundError:
            console.print(f"[red]File {filename} not found[/red]")

    def chat(self):
        """Run interactive chat session with keyboard shortcuts"""
        # Setup history file
        history_file = Path.home() / ".gapgpt_history"

        # Create prompt session with bindings and history
        session = PromptSession(
            key_bindings=bindings, history=FileHistory(str(history_file))
        )

        console.print(
            Panel.fit(
                f"[bold cyan]🤖 GapGPT Chatbot[/bold cyan]\n"
                f"[dim]Model: {self.model}[/dim]\n"
                f"[dim]Commands: /clear, /save, /load, /exit[/dim]\n"
                f"[dim]Shortcuts: Ctrl+C (exit), Ctrl+L (clear screen), ↑↓ (history)[/dim]",
                border_style="cyan",
            )
        )

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")

                # Handle commands
                if user_input.startswith("/"):
                    cmd = user_input.lower()
                    if cmd == "/exit" or cmd == "/quit":
                        console.print("[cyan]Goodbye! 👋[/cyan]")
                        break
                    elif cmd == "/clear":
                        self._clear_history()
                        continue
                    elif cmd == "/save":
                        self.save_conversation()
                        continue
                    elif cmd.startswith("/load "):
                        filename = cmd.split(" ", 1)[1]
                        self.load_conversation(filename)
                        continue
                    else:
                        console.print(
                            "[red]Unknown command. Try: /clear, /save, /load filename, /exit[/red]"
                        )
                        continue

                # Handle @src file reference
                elif user_input.startswith("@src("):
                    # Pattern to match @src(path/to/file) rest of question
                    pattern = r"^@src\(([^)]+)\)\s*(.*)"
                    match = re.match(pattern, user_input)

                    if match:
                        file_path = match.group(1)  # file path
                        user_text = match.group(2)  # user's question

                        # Check if file exists
                        path = Path(file_path)
                        if not path.exists():
                            console.print(f"[red]❌ File not found: {file_path}[/red]")
                            continue

                        # Read file content
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                file_content = f.read()

                            # Get file extension for syntax highlighting
                            ext = path.suffix[1:] if path.suffix else "text"

                            # Format with code block
                            user_input = f"""```{ext}
                                {file_content}
                                ```
                                Question: {user_text}"""
                        except Exception as e:
                            console.print(f"[red]❌ Error reading file: {e}[/red]")
                            continue
                    else:
                        console.print(
                            "[red]❌ Invalid @src format. Use: @src(path/to/file) your question[/red]"
                        )
                        continue

                # Add user message
                self._add_message("user", user_input)

                console.print()

                full_response = ""

                console.print("[dim]🤖: [/dim]", end="")

                # Show thinking indicator
                with console.status("[bold green]🤔 Thinking..."):
                    response_text = Text()

                    for chunk in self._get_response():
                        full_response += chunk
                        console.print(chunk, end="", style="bright_green")

                    console.print()

                # Add assistant response
                self._add_message("assistant", full_response)

                # Display response
                console.print(
                    Panel(
                        Markdown(full_response),
                        title="🤖 GapGPT",
                        border_style="green",
                        padding=(1, 2),
                    )
                )

            except KeyboardInterrupt:
                console.print(
                    "\n[yellow]Press Ctrl+C again or type /exit to quit[/yellow]"
                )
            except EOFError:
                console.print("\n[cyan]Goodbye! 👋[/cyan]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


@app.command()
def chat(
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        help="GapGPT API key (overrides saved config)",
        show_default=False,
    ),
    model: str = typer.Option(
        "gpt-4o",
        "--model",
        "-m",
        help="GapGPT model to use. (see https://gapgpt.app/platform-v2/pricing?provider=OpenAI for more models)",
    ),
    no_clear: bool = typer.Option(
        False, "--no-clear", help="Don't clear screen on startup"
    ),
):
    """Start an interactive chat session with GapGPT"""

    # If no API key provided, try global config
    if not api_key:
        config = App.load_global_config()
        api_key = config.get("api_key")
        if api_key:
            console.print("[dim]Using API key from global config[/dim]")

    # Clear screen and show banner (unless disabled)
    if not no_clear:
        clear_screen()
        show_banner()

    bot = App(api_key=api_key, model=model)
    bot.chat()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="GapGPT API key (overrides saved config)"
    ),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model to use"),
):
    """Ask a single question (doesn't clear screen)"""

    # If no API key provided, try global config
    if not api_key:
        config = App.load_global_config()
        api_key = config.get("api_key")

    bot = App(api_key=api_key, model=model)
    bot._add_message("user", question)

    with console.status("[bold cyan]Getting answer..."):
        response = bot._get_response()

    console.print(Panel(response, title="🤖 Answer", border_style="green"))


@app.command()
def config(
    set_key: Optional[str] = typer.Option(
        None, "--set-key", "-s", help="Set API key globally"
    ),
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    clear: bool = typer.Option(
        False, "--clear", help="Clear API key from global config"
    ),
    path: bool = typer.Option(False, "--path", help="Show config file path"),
    new_key: bool = typer.Option(
        False, "--new-key", help="Open browser to get new API key"
    ),
):
    """Manage global configuration (API key stored in ~/.config/gapgpt/config.json)"""

    if path:
        config_path = App.get_config_path()
        console.print(f"[cyan]Config file location:[/cyan] {config_path}")
        return

    if show:
        config = App.load_global_config()
        if config and config.get("api_key"):
            console.print("[bold green]Current configuration:[/bold green]")
            api_key = config.get("api_key", "Not set")
            # Show only first/last few characters for security
            if len(api_key) > 20:
                masked = api_key[:15] + "..." + api_key[-5:]
            else:
                masked = "*" * len(api_key)
            console.print(f"  API Key: {masked}")
            console.print(f"  Default Model: {config.get('default_model', 'gpt-4o')}")
        else:
            console.print(
                "[yellow]No configuration found. Use 'gapgpt config --set-key YOUR_KEY' to set your API key[/yellow]"
            )
        return

    if clear:
        config = App.load_global_config()
        if "api_key" in config:
            del config["api_key"]
            App.save_global_config(config)
            console.print("[green]✅ API key cleared from global config[/green]")
        else:
            console.print("[yellow]No API key found in config[/yellow]")
        return

    if set_key:
        config = App.load_global_config()
        config["api_key"] = set_key
        if "default_model" not in config:
            config["default_model"] = "gpt-4o"
        App.save_global_config(config)
        console.print("[green]✅ API key saved globally![/green]")
        console.print("[dim]You can now run 'gapgpt chat' without --api-key[/dim]")
        return
    if new_key:
        url = "https://gapgpt.app/platform-v2/tokens"
        console.print(f"[dim]🌐 Opening browser to: {url}[/dim]")
        webbrowser.open(url)
        console.print(
            "[green]✅ Browser opened! Get your API key and then run:[/green]"
        )
        console.print("[dim]  config --set-key YOUR_KEY[/dim]")
        return

    # If no options, show help
    console.print("[bold cyan]GapGPT Configuration Manager[/bold cyan]")
    console.print("\n[bold]Commands:[/bold]")
    console.print(
        "  [green]gapgpt config --set-key YOUR_KEY[/green]   # Save API key globally"
    )
    console.print(
        "  [green]gapgpt config --show[/green]               # Show current config"
    )
    console.print(
        "  [green]gapgpt config --clear[/green]              # Remove API key"
    )
    console.print(
        "  [green]gapgpt config --path[/green]               # Show config file location"
    )
    console.print("\n[bold]Examples:[/bold]")
    console.print("  gapgpt config --set-key sk-abc123...")
    console.print("  gapgpt chat  # Uses saved key")


@app.command()
def help():
    """Show detailed help for GapGPT CLI"""
    console.print("\n[bold green]GapGPT CLI v1.0.0[/bold green]\n")

    console.print("[bold]Configuration Commands:[/bold]")
    console.print("  [cyan]config --set-key KEY[/cyan]     - Set API key globally")
    console.print("  [cyan]config --show[/cyan]            - Show current config")
    console.print("  [cyan]config --clear[/cyan]           - Remove API key globally")
    console.print("  [cyan]config --path[/cyan]            - Show config file location")

    console.print("\n[bold]Chat Commands:[/bold]")
    console.print("  [cyan]chat[/cyan]                     - Start interactive chat")
    console.print("  [cyan]chat --api-key KEY[/cyan]       - Use specific API key")
    console.print("  [cyan]chat --no-clear[/cyan]          - Don't clear terminal")
    console.print(
        "  [cyan]chat --model MODEL[/cyan]       - Set model (default: gpt-4o)"
    )

    console.print("\n[bold]Question Commands:[/bold]")
    console.print("  [cyan]ask QUESTION[/cyan]             - Ask a single question")
    console.print("  [cyan]ask --api-key KEY[/cyan]        - Use specific API key")
    console.print(
        "  [cyan]ask --model MODEL[/cyan]        - Set model for this question"
    )

    console.print("\n[bold]File Reference:[/bold]")
    console.print(
        "  [cyan]@src(path/to/file)[/cyan]       - Include file content in chat"
    )

    console.print("\n[bold]Chat Shortcuts:[/bold]")
    console.print(
        "  [cyan]/clear[/cyan]                   - Clear conversation history"
    )
    console.print("  [cyan]/save[/cyan]                    - Save conversation to JSON")
    console.print("  [cyan]/load filename[/cyan]           - Load saved conversation")
    console.print("  [cyan]/exit[/cyan]                    - Exit chat")

    console.print("\n[bold]Keyboard Shortcuts:[/bold]")
    console.print("  [cyan]Ctrl+C[/cyan]                   - Exit chat")
    console.print("  [cyan]Ctrl+L[/cyan]                   - Clear screen")
    console.print("  [cyan]↑/↓[/cyan]                      - Navigate command history")


if __name__ == "__main__":
    app()
