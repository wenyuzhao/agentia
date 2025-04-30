import os
from typing import Annotated
import typer
import agentia.utils
from pathlib import Path
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


def __setup_server(log_level: str, port: int, config_dir: Path | None):
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
app.add_typer(serve, name="start")


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
    port = __setup_server(log_level, port, config_dir)
    agentia.utils._setup_logging()
    config.prepare_user_plugins()
    session.cleanup_cache()

    import streamlit.web.bootstrap

    entry = Path(__file__).parent / "web" / "app.py"

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
    port = __setup_server(log_level, port, config_dir)
    agentia.utils._setup_logging()
    config.prepare_user_plugins()
    session.cleanup_cache()

    import uvicorn
    from agentia_app.api import app

    uvicorn.run(app, host="localhost", port=port)


@app.callback()
def callback():
    pass


def main():
    app()
