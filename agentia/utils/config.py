from typing import TYPE_CHECKING, Annotated, Any, Literal
import tomllib
from pathlib import Path
import os
import importlib.util

from agentia.agent import Agent
from pydantic import AfterValidator, BaseModel, Field, ValidationError

from agentia.mcp import MCPServer

if TYPE_CHECKING:
    from agentia.tools import Tool
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
    mcp: list[str] | None = None

    def build_instructions(self) -> str | None:
        if self.name and not self.description:
            return f"YOUR ARE {self.name}\n\n{self.instructions or ''}"
        elif self.description and not self.name:
            return f"YOUR DESCRIPTION: {self.description}\n\n{self.instructions or ''}"
        elif self.description and self.name:
            return (
                f"YOUR ARE {self.name}, {self.description}\n\n{self.instructions or ''}"
            )
        if self.user:
            return f"{self.instructions or ''}\n\nYOU ARE TALKING TO THE USER.\nMY INFO (THE USER): {self.user}"
        else:
            return self.instructions


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


class MCPLocalConfig(BaseModel):
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    cwd: str | None = None
    type: Literal["local"] = "local"


class MCPHTTPConfig(BaseModel):
    url: str
    headers: dict[str, str] | None = None
    type: Literal["http"] = "http"


class MCPSSEConfig(BaseModel):
    url: str
    headers: dict[str, str] | None = None
    type: Literal["sse"] = "sse"


class MCPWebSocketConfig(BaseModel):
    name: str
    url: str
    type: Literal["websocket"] = "websocket"


MCPConfig = Annotated[
    MCPLocalConfig | MCPHTTPConfig | MCPSSEConfig | MCPWebSocketConfig,
    Field(union_mode="left_to_right"),
]


class Config(BaseModel):
    agent: AgentConfig
    plugins: Annotated[PluginConfigs, AfterValidator(check_plugins)] = Field(
        default_factory=dict
    )
    mcp: dict[str, MCPConfig] = Field(default_factory=dict)

    def get_enabled_plugins(self) -> list[str]:
        if self.agent.plugins is not None:
            return sorted(self.agent.plugins)
        else:
            return sorted(self.plugins.keys())

    def get_enabled_mcp_servers(self) -> dict[str, MCPConfig]:
        if self.agent.mcp is not None:
            return {name: self.mcp[name] for name in self.agent.mcp if name in self.mcp}
        else:
            return self.mcp


class AgentInfo(BaseModel):
    id: str
    config_path: Path
    config: "Config"


def __create_plugins(config: Config) -> tuple[list["Plugin"], dict[str, Any]]:
    from agentia.plugins import ALL_PLUGINS

    plugins: list["Plugin"] = []
    plugin_configs = {}
    enabled_plugins = config.get_enabled_plugins()
    for name in enabled_plugins:
        c = config.plugins.get(name, {})
        if name not in ALL_PLUGINS:
            raise ValueError(f"Unknown plugin: {name}")
        PluginCls = ALL_PLUGINS[name]
        if not (c is None or isinstance(c, dict)):
            raise ValueError(
                f"Invalid config for plugin {name}: must be a dict or null"
            )
        c = c if isinstance(c, dict) else {}
        plugin_configs[name] = c
        plugins.append(PluginCls.instantiate(config=c))
    return plugins, plugin_configs


def __create_mcp_servers(config: Config) -> dict[str, "MCPServer"]:
    mcps: dict[str, MCPServer] = {}
    for name, cfg in config.mcp.items():
        match cfg.type:
            case "local":
                mcps[name] = MCPServer(
                    name=name,
                    command=cfg.command,
                    args=cfg.args,
                    env=cfg.env,
                    cwd=cfg.cwd,
                )
            case "http":
                mcps[name] = MCPServer(
                    name=name,
                    type="http",
                    url=cfg.url,
                    headers=cfg.headers,
                )
            case "sse":
                mcps[name] = MCPServer(
                    name=name,
                    type="sse",
                    url=cfg.url,
                    headers=cfg.headers,
                )
            case "websocket":
                mcps[name] = MCPServer(
                    name=name,
                    type="websocket",
                    url=cfg.url,
                )
            case _:
                raise ValueError(f"Unknown MCP server type: {cfg.type}")
    return mcps


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
    plugins, _ = __create_plugins(config)
    mcp_servers = __create_mcp_servers(config)
    tools: list["Tool"] = [*plugins, *mcp_servers.values()]
    # Create agent
    agent_id = file.stem
    agent = Agent(
        id=agent_id,
        model=config.agent.model,
        tools=tools,
        instructions=config.agent.build_instructions(),
    )
    agent.context["config"] = config
    return agent


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
