import asyncio
from agentia.agent import (
    Agent,
    ChatCompletion,
    Event,
    ToolCallEvent,
    CommunicationEvent,
    MessageStream,
)
from agentia.message import Message, UserMessage
from agentia.utils.config import load_agent_from_config
import rich
import dotenv

dotenv.load_dotenv()


async def __dump(agent: Agent, completion: ChatCompletion[MessageStream | Event]):
    def print_name_and_icon(name: str | None, icon: str | None, end: str = "\n"):
        name = name or "Agent"
        name_and_icon = f"[{icon} {name}]" if icon else f"[{name}]"
        rich.print(f"[bold blue]{name_and_icon}[/bold blue]", end=end, flush=True)

    async for msg in completion:
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
                and msg.tool.name != "_communicate"
            ):
                rich.print(
                    f"[magenta][[bold]✨ TOOL:[/bold] {msg.tool.display_name}][/magenta]"
                )
            elif isinstance(msg, CommunicationEvent):
                p, c = msg.parent, msg.child
                direction = "->" if msg.response is None else "<-"
                rich.print(
                    f"[magenta][[bold]✨ COMMUNICATE:[/bold] {p.name} {direction} {c.name}][/magenta] [dim]{msg.message}[/dim]"
                )


async def __run_async(agent: Agent):
    await agent.init()
    while True:
        try:
            console = rich.console.Console()
            prompt = console.input("[bold green]>[/bold green] ").strip()
            if prompt == "exit" or prompt == "quit":
                break
        except EOFError:
            break
        response = agent.chat_completion(
            [UserMessage(prompt)], stream=True, events=True
        )
        await __dump(agent, response)


def run(agent: Agent | str):
    if isinstance(agent, str):
        agent = load_agent_from_config(agent)

    asyncio.run(__run_async(agent))
