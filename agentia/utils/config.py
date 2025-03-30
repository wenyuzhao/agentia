from typing import Annotated, Any
import tomllib
from pathlib import Path

from agentia.agent import Agent
from agentia.plugins import ALL_PLUGINS, Plugin
from pydantic import AfterValidator, BaseModel, Field

AGENTS_SEARCH_PATHS = [
    Path.cwd(),
    Path.cwd() / "agents",
    Path.cwd() / ".agents",
    Path.home() / ".config" / "agentia" / "agents",
]


class AgentConfig(BaseModel):
    name: str
    icon: str | None = None
    instructions: str | None = None
    description: str | None = None
    language: str | None = None
    model: str | None = None
    icon: str | None = None
    knowledge_base: str | bool = False
    colleagues: list[str] = Field(default_factory=list)
    user: str | None = None


PluginsConfig = dict[str, bool | dict[str, Any]]


def check_plugins(configs: PluginsConfig) -> PluginsConfig:
    for name, config in configs.items():
        if config is False or config is None:
            continue
        if name not in ALL_PLUGINS:
            raise ValueError(f"Unknown plugin: {name}")
        config = config if isinstance(config, dict) else {}
        ALL_PLUGINS[name].validate_config(config)
    return configs


class Config(BaseModel):
    agent: AgentConfig
    plugins: Annotated[PluginsConfig, AfterValidator(check_plugins)] = Field(
        default_factory=dict
    )


def __get_config_path(cwd: Path, id: str):
    id = id.strip()
    possible_paths = []
    if id.endswith(".toml"):
        possible_paths.append(id)
    else:
        possible_paths.append(f"{id}.toml")
    for s in possible_paths:
        p = Path(s)
        if not p.is_absolute():
            p = cwd / p
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"Agent config not found: {id}")


def __create_tools(
    config: Config, parent_plugins_config: PluginsConfig
) -> tuple[list[Plugin], dict[str, Any]]:
    tools: list[Plugin] = []
    tool_configs = {}
    for name, c in config.plugins.items():
        if c == False or c is None:
            continue
        if name not in ALL_PLUGINS:
            raise ValueError(f"Unknown tool: {name}")
        PluginCls = ALL_PLUGINS[name]
        if not (c is None or isinstance(c, dict)):
            raise ValueError(f"Invalid config for tool {name}: must be a dict or null")
        if c is None and name in parent_plugins_config:
            c = parent_plugins_config[name]
        c = c if isinstance(c, dict) else {}
        tool_configs[name] = c
        tools.append(PluginCls(config=c))
    return tools, tool_configs


def __load_agent_from_config(
    file: Path,
    pending: set[Path],
    agents: dict[Path, Agent],
    parent_tool_configs: dict[str, Any],
):
    """Load a bot from a configuration file"""
    # Load the configuration file
    assert file.exists()
    file = file.resolve()
    config = Config(**tomllib.loads(file.read_text()))
    # Already loaded?
    if file in pending:
        raise ValueError(f"Circular dependency detected: {file.stem}")
    pending.add(file)
    if file in agents:
        return agents[file]
    # Create tools
    tools, tool_configs = __create_tools(config, parent_tool_configs)
    # Load colleagues
    colleagues: list[Agent] = []
    for child_id in config.agent.colleagues:
        child_path = __get_config_path(file.parent, child_id)
        colleague = __load_agent_from_config(child_path, pending, agents, tool_configs)
        colleagues.append(colleague)
    # Create agent

    knowledge_base: str | bool = config.agent.knowledge_base
    agent_id = file.stem
    agent = Agent(
        id=agent_id,
        name=config.agent.name,
        icon=config.agent.icon,
        description=config.agent.description,
        model=config.agent.model,
        tools=tools,
        instructions=config.agent.instructions,
        colleagues=colleagues,
        knowledge_base=(
            Path(knowledge_base) if isinstance(knowledge_base, str) else knowledge_base
        ),
        user=config.agent.user,
    )
    agent.original_config = config
    pending.remove(file)
    agents[file] = agent
    return agent


def load_agent_from_config(name: str | Path) -> Agent:
    """Load a bot from a configuration file"""
    if isinstance(name, Path):
        if not name.exists():
            raise FileNotFoundError(f"Agent config not found: {name}")
        config_path = name.resolve()
    elif name.endswith(".toml"):
        # name is also a path
        config_path = Path(name)
        if not config_path.exists() or not config_path.is_file():
            raise FileNotFoundError(f"Agent config not found: {name}")
        config_path = config_path.resolve()
    elif (s := Path(name).suffix) and s != ".toml":
        raise ValueError(f"Invalid agent path: {name}")
    else:
        # If the name is a string, we need to find the configuration file from a list of search paths
        config_path = None
        for dir in AGENTS_SEARCH_PATHS:
            if (file := dir / f"{name}.toml").exists():
                config_path = file
                break
        if config_path is None:
            raise FileNotFoundError(f"Agent config not found: {name}")
    return __load_agent_from_config(config_path, set(), {}, {})
