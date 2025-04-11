import os
from typing import Annotated
import typer
import agentia.utils
import asyncio
from pathlib import Path
from agentia import Agent
import dotenv

import agentia.utils.config as config
import agentia.utils.session as session
from agentia.utils.config import DEFAULT_AGENT_CONFIG_PATH

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings=dict(help_option_names=["-h", "--help"]),
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
)


@app.command(help="Start the command line REPL")
def repl(
    agent: str,
    config_dir: Annotated[
        Path | None,
        typer.Option(
            help=f"Path to the agent configuration directory. DEFAULT: {DEFAULT_AGENT_CONFIG_PATH}",
        ),
    ] = None,
    user_plugin_dir: Annotated[
        Path | None,
        typer.Option(
            help=f"Path to the user-defined plugins directory. DEFAULT: {DEFAULT_AGENT_CONFIG_PATH}",
        ),
    ] = None,
):
    dotenv.load_dotenv()
    os.environ["AGENTIA_CLI"] = "1"
    if config_dir is not None:
        os.environ["AGENTIA_CONFIG_DIR"] = str(config_dir)
    if user_plugin_dir is not None:
        os.environ["AGENTIA_USER_PLUGIN_DIR"] = str(user_plugin_dir)
    config.prepare_user_plugins()
    agentia.utils.repl.run(agent)


@app.command(help="Setup plugins for all agents")
def setup():
    dotenv.load_dotenv()
    os.environ["AGENTIA_CLI"] = "1"
    ALL_AGENTS = config.get_all_agents()
    for agent_info in ALL_AGENTS:
        agent = Agent.load_from_config(agent_info.id)
        asyncio.run(agent._Agent__init_plugins())  # type: ignore


def __check_and_setup_server(log_level: str, port: int, config_dir: Path | None):
    try:
        import streamlit
    except ImportError:
        raise RuntimeError(
            "Agentia REPL is not supported. You may need to reinstall agentia with all dependencies: `pipx install agentia[all]`"
        )
    dotenv.load_dotenv()

    os.environ["AGENTIA_SERVER"] = "1"
    if "LOG_LEVEL" not in os.environ:
        os.environ["LOG_LEVEL"] = log_level
    if config_dir is not None:
        os.environ["AGENTIA_CONFIG_DIR"] = str(config_dir)

    # Streamlit options
    if "SERVER_PORT" in os.environ:
        port = int(os.environ["SERVER_PORT"])
    return port


serve = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings=dict(help_option_names=["-h", "--help"]),
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
)
app.add_typer(serve, name="serve")


@serve.command(name="app", help="Start the web app server")
def serve_app(
    port: int = 8501,
    dev: bool = False,
    config_dir: Annotated[
        Path | None,
        typer.Option(
            help=f"Path to the agent configuration directory. DEFAULT: {DEFAULT_AGENT_CONFIG_PATH}",
        ),
    ] = None,
    log_level: str = "DEBUG",
):
    port = __check_and_setup_server(log_level, port, config_dir)
    agentia.utils._setup_logging()
    config.prepare_user_plugins()
    session.cleanup_cache()

    import streamlit.web.bootstrap

    entry = Path(__file__).parent / "_app" / "app.py"

    streamlit.web.bootstrap.load_config_options(
        flag_options={
            "server.port": port,
            "server.fileWatcherType": "auto" if dev else "none",
            "runner.magicEnabled": False,
        }
    )

    streamlit.web.bootstrap.run(str(entry), False, [], {})


@serve.command(name="api", help="Start the API server")
def serve_api(
    port: int = 8000,
    dev: bool = False,
    config_dir: Annotated[
        Path | None,
        typer.Option(
            help=f"Path to the agent configuration directory. DEFAULT: {DEFAULT_AGENT_CONFIG_PATH}",
        ),
    ] = None,
    log_level: str = "DEBUG",
):
    port = __check_and_setup_server(log_level, port, config_dir)
    agentia.utils._setup_logging()
    config.prepare_user_plugins()
    session.cleanup_cache()

    import uvicorn
    from agentia._api import app

    uvicorn.run(app, host="localhost", port=port)


@app.callback()
def callback():
    pass


def main():
    app()
