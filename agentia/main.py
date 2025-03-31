import typer
import agentia.utils

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
    __check_group()
    agentia.utils.repl.run(agent)


@app.command(help="Start the web app server")
def serve():
    __check_group()
    import streamlit.web.bootstrap
    from pathlib import Path

    entry = Path(__file__).parent / "utils" / "app" / "app.py"

    streamlit.web.bootstrap.run(str(entry), False, [], {})


@app.callback()
def callback():
    pass


def main():
    app()
