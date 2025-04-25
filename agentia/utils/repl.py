import asyncio
from pathlib import Path
from agentia.agent import Agent
from agentia.run import Run, MessageStream
from agentia.message import Event, ToolCallEvent, CommunicationEvent
from agentia.message import Message, UserMessage
from agentia.utils.config import load_agent_from_config
import rich, rich.panel
import dotenv

dotenv.load_dotenv()


async def __dump(agent: Agent, run: Run[MessageStream | Event]):
    def print_name_and_icon(name: str | None, icon: str | None, end: str = "\n"):
        name = name or "Agent"
        name_and_icon = f"[{icon} {name}]" if icon else f"[{name}]"
        rich.print(f"[bold blue]{name_and_icon}[/bold blue]", end=end, flush=True)

    async for msg in run:
        if isinstance(msg, Message):
            print_name_and_icon(agent.name, agent.icon)
            print(msg.content)
        elif isinstance(msg, MessageStream):
            name_printed = False
            outputed = False
            async for delta in msg:
                if delta == "":
                    continue
                if not name_printed:
                    print_name_and_icon(agent.name, agent.icon)
                    name_printed = True
                outputed = True
                print(delta, end="", flush=True)
            if outputed:
                print()
        elif isinstance(msg, ToolCallEvent | CommunicationEvent):
            if (
                isinstance(msg, ToolCallEvent)
                and msg.result is None
                and msg.name != "_communicate"
            ):
                rich.print(
                    f"[magenta][[bold]✨ TOOL:[/bold] {msg.display_name}][/magenta]"
                )
            elif isinstance(msg, CommunicationEvent):
                c = agent.colleagues[msg.child].name
                direction = "->" if msg.response is None else "<-"
                rich.print(
                    f"[magenta][[bold]✨ COMMUNICATE:[/bold] {agent.name} {direction} {c}][/magenta] [dim]{msg.message}[/dim]"
                )


async def __run_async(agent: Agent):
    await agent.init()
    assert agent.config_path
    config_path = agent.config_path.relative_to(Path.cwd())
    header = f"[bold blue]RUNNING:[/bold blue] [blue]{agent.id}[/blue] [dim italic]{config_path}[/dim italic]"
    rich.print(rich.panel.Panel.fit(header))
    while True:
        try:
            console = rich.console.Console()
            prompt = console.input("[bold green]>[/bold green] ").strip()
            if prompt == "exit" or prompt == "quit":
                break
        except EOFError:
            break
        run = agent.run(prompt, stream=True, events=True)
        await __dump(agent, run)


def run(agent: Agent | str):
    if isinstance(agent, str):
        agent = load_agent_from_config(agent, persist=False, session_id=None)

    asyncio.run(__run_async(agent))
