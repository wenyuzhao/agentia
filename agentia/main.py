import os
import typer
import agentia.utils
import asyncio
from pathlib import Path
from agentia import Agent
import dotenv

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings=dict(help_option_names=["-h", "--help"]),
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
)


def __check_group():
    try:
        import streamlit
    except ImportError:
        raise RuntimeError(
            "Agentia REPL is not supported. You may need to reinstall agentia with all dependencies: `pipx install agentia[all]`"
        )


@app.command(help="Start the command line REPL")
def repl(agent: str):
    dotenv.load_dotenv()
    agentia.utils.repl.run(agent)


@app.command(help="Start the web app server")
def serve(port: int = 8501, dev: bool = False):
    __check_group()

    dotenv.load_dotenv()

    os.environ["AGENTIA_SERVER"] = "1"
    # Streamlit options
    if "SERVER_PORT" in os.environ:
        port = int(os.environ["SERVER_PORT"])

    import streamlit.web.bootstrap

    entry = Path(__file__).parent / "_app" / "1_💬_Chat.py"

    streamlit.web.bootstrap.load_config_options(
        flag_options={
            "server.port": port,
            "server.fileWatcherType": "auto" if dev else "none",
            "runner.magicEnabled": False,
        }
    )

    streamlit.web.bootstrap.run(str(entry), False, [], {})


@app.command(help="Setup plugins for all agents")
def setup():
    dotenv.load_dotenv()
    ALL_AGENTS = Agent.get_all_agents()
    for agent_info in ALL_AGENTS:
        agent = Agent.load_from_config(agent_info.id)
        asyncio.run(agent._Agent__init_plugins())  # type: ignore


@app.callback()
def callback():
    pass


def main():
    app()
