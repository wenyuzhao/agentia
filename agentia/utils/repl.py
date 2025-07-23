import asyncio
from agentia.agent import Agent
from agentia.run import Run, MessageStream
from agentia.message import Event, ToolCallEvent
from agentia.message import Message
from agentia.utils.config import load_agent_from_config
import rich, rich.panel
import dotenv

dotenv.load_dotenv()


async def __dump(agent: Agent, run: Run[MessageStream | Event]):
    def print_name_and_icon(name: str | None, icon: str | None, end: str = "\n"):
        name = name or "Agent"
        name_and_icon = f"[{icon} {name}]" if icon else f"[{name}]"
        rich.print(f"[bold blue]{name_and_icon}[/bold blue]", end=end, flush=True)

    if config := agent.context.get("config"):
        name = config.agent.name or agent.id
        icon = config.agent.icon
    else:
        name = agent.id
        icon = None

    async for msg in run:
        if isinstance(msg, Message):
            print_name_and_icon(name, icon)
            print(msg.content)
        elif isinstance(msg, MessageStream):
            name_printed = False
            outputed = False
            async for delta in msg:
                if delta == "":
                    continue
                if not name_printed:
                    print_name_and_icon(name, icon)
                    name_printed = True
                outputed = True
                print(delta, end="", flush=True)
            if outputed:
                print()
        elif isinstance(msg, ToolCallEvent):
            if (
                isinstance(msg, ToolCallEvent)
                and msg.result is None
                and msg.name != "_communicate"
            ):
                rich.print(
                    f"[magenta][[bold]✨ TOOL:[/bold] {msg.display_name}][/magenta]"
                )


async def __run_async(agent: Agent):
    await agent.init()
    header = f"[bold blue]RUNNING:[/bold blue] [blue]{agent.id}[/blue]"
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
        agent = load_agent_from_config(agent)

    asyncio.run(__run_async(agent))
