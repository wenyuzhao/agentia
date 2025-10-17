from pathlib import Path
import tomllib
import logging.config

from . import config, voice

__all__ = ["voice", "config"]


def _setup_logging():
    if (Path.cwd() / "logging.toml").exists():
        config = tomllib.loads((Path.cwd() / "logging.toml").read_text())
        logging.config.dictConfig(config)
