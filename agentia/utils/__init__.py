from pathlib import Path
import tomllib
import logging.config

from . import config, voice, repl

__all__ = ["voice", "config", "repl"]


def _setup_logging():
    if (Path.cwd() / "logging.toml").exists():
        config = tomllib.loads((Path.cwd() / "logging.toml").read_text())
        logging.config.dictConfig(config)
