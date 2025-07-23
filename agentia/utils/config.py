from typing import TYPE_CHECKING, Annotated, Any
import tomllib
from pathlib import Path
import tomlkit
import os
import importlib.util

from agentia.agent import Agent
from pydantic import AfterValidator, BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from agentia.plugins import Plugin

DEFAULT_AGENT_CONFIG_PATH = Path.cwd() / "agents"
DEFAULT_AGENT_USER_PLUGIN_PATH = Path.cwd() / "plugins"


class AgentConfig(BaseModel):
    name: str
    icon: str | None = None
    description: str | None = None
    instructions: str | None = None
    model: str | None = None
    user: str | None = None
    plugins: list[str] | None = None


PluginConfigs = dict[str, dict[str, Any]]


def check_plugins(configs: PluginConfigs) -> PluginConfigs:
    from agentia.plugins import ALL_PLUGINS

    for name, config in configs.items():
        if config is False or config is None:
            continue
        if name not in ALL_PLUGINS:
            raise ValueError(
                f"Unknown plugin: {name}. Available plugins: {', '.join(ALL_PLUGINS.keys())}"
            )
        config = config if isinstance(config, dict) else {}
        ALL_PLUGINS[name].validate_config(config)
    return configs


class Config(BaseModel):
    agent: AgentConfig
    plugins: Annotated[PluginConfigs, AfterValidator(check_plugins)] = Field(
        default_factory=dict
    )

    def get_enabled_plugins(self) -> list[str]:
        if self.agent.plugins is not None:
            return sorted(self.agent.plugins)
        else:
            return sorted(self.plugins.keys())


class AgentInfo(BaseModel):
    id: str
    config_path: Path
    config: "Config"


def __create_tools(config: Config) -> tuple[list["Plugin"], dict[str, Any]]:
    from agentia.plugins import ALL_PLUGINS

    tools: list["Plugin"] = []
    tool_configs = {}
    enabled_plugins = (
        config.agent.plugins
        if config.agent.plugins is not None
        else list(config.plugins.keys())
    )
    for name in enabled_plugins:
        c = config.plugins.get(name, {})
        if name not in ALL_PLUGINS:
            raise ValueError(f"Unknown tool: {name}")
        PluginCls = ALL_PLUGINS[name]
        if not (c is None or isinstance(c, dict)):
            raise ValueError(f"Invalid config for tool {name}: must be a dict or null")
        c = c if isinstance(c, dict) else {}
        tool_configs[name] = c
        tools.append(PluginCls.instantiate(config=c))
    return tools, tool_configs


def __load_agent_from_config(file: Path) -> Agent:
    """Load a bot from a configuration file"""

    # Load the configuration file
    assert file.exists()
    file = file.resolve()
    try:
        config = Config(**tomllib.loads(file.read_text()))
    except ValidationError as e:
        raise ValueError(f"Invalid config file: {file}\n{repr(e)}") from e
    # Create tools
    tools, tool_configs = __create_tools(config)
    # Create agent
    agent_id = file.stem
    agent = Agent(
        id=agent_id,
        model=config.agent.model,
        tools=tools,
        instructions=config.agent.instructions,
    )
    agent.context["config"] = config
    return agent


def get_config_dir() -> Path:
    if path := os.environ.get("AGENTIA_CONFIG_DIR"):
        config_dir = Path(path)
    else:
        config_dir = DEFAULT_AGENT_CONFIG_PATH
    config_dir = config_dir.absolute()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


__user_plugins_loaded = False


def prepare_user_plugins():
    from agentia import LOGGER

    global __user_plugins_loaded
    if __user_plugins_loaded:
        return
    __user_plugins_loaded = True
    if path := os.environ.get("AGENTIA_USER_PLUGIN_DIR"):
        plugins_dir = Path(path)
    else:
        plugins_dir = DEFAULT_AGENT_USER_PLUGIN_PATH
    if not plugins_dir.is_dir():
        return
    LOGGER.info(f"Loading user plugins from {plugins_dir}")
    # Install requirements.txt
    if (plugins_dir / "requirements.txt").is_file():
        ret = os.system(f"uv pip install -r {plugins_dir / "requirements.txt"}")
        if ret != 0:
            raise RuntimeError(
                f"Failed to install requirements.txt from {plugins_dir}: {ret}"
            )
    # Load plugins
    for file in plugins_dir.glob("*.py"):
        if not file.is_file() or file.stem.startswith(("_", ".", "-")):
            continue
        name = file.stem
        LOGGER.info(f" - {name} ({file})")
        spec = importlib.util.spec_from_file_location(name, file)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def load_agent_from_config(config_path: str | Path) -> Agent:
    """Load a bot from a configuration file"""
    config_path = Path(config_path) if isinstance(config_path, str) else config_path
    assert config_path.suffix == ".toml", "Agent config file must be a .toml file"
    config_path = config_path.resolve()
    return __load_agent_from_config(config_path)
