import os
from typing import Annotated
import typer
import agentia.utils
from pathlib import Path
import dotenv

import agentia.utils.config as config
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


def main():
    app()
