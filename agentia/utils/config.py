from typing import Annotated, Any
import tomllib
from pathlib import Path
import shelve
import tomlkit
import os
import importlib.util

from agentia.agent import Agent, AgentInfo, _get_global_cache_dir
from agentia.plugins import ALL_PLUGINS, Plugin
from pydantic import AfterValidator, BaseModel, Field, ValidationError

DEFAULT_AGENT_CONFIG_PATH = Path.cwd() / "agents"
DEFAULT_AGENT_USER_PLUGIN_PATH = Path.cwd() / "plugins"


class AgentConfig(BaseModel):
    name: str
    icon: str | None = None
    description: str | None = None
    instructions: str | None = None
    model: str | None = None
    knowledge_base: str | bool = False
    colleagues: list[str] = Field(default_factory=list)
    user: str | None = None
    plugins: list[str] | None = None


PluginConfigs = dict[str, dict[str, Any]]


def check_plugins(configs: PluginConfigs) -> PluginConfigs:
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
    plugins: Annotated[PluginConfigs, AfterValidator(check_plugins)] = Field(
        default_factory=dict
    )

    def get_enabled_plugins(self) -> list[str]:
        if self.agent.plugins is not None:
            return sorted(self.agent.plugins)
        else:
            return sorted(self.plugins.keys())


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


def __create_tools(config: Config) -> tuple[list[Plugin], dict[str, Any]]:
    tools: list[Plugin] = []
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
        tools.append(PluginCls(config=c))
    return tools, tool_configs


def __load_agent_from_config(
    file: Path,
    pending: set[Path],
    agents: dict[Path, Agent],
    persist: bool,
    session_id: str | None,
    log_level: str | int | None = None,
):
    """Load a bot from a configuration file"""
    # Load the configuration file
    assert file.exists()
    file = file.resolve()
    try:
        config = Config(**tomllib.loads(file.read_text()))
    except ValidationError as e:
        raise ValueError(f"Invalid config file: {file}\n{repr(e)}") from e
    # Already loaded?
    if file in pending:
        raise ValueError(f"Circular dependency detected: {file.stem}")
    pending.add(file)
    if file in agents:
        return agents[file]
    # Create tools
    tools, tool_configs = __create_tools(config)
    # Load colleagues
    colleagues: list[Agent] = []
    colleague_session_ids: dict[str, str] = {}
    if persist and session_id:
        history = _get_global_cache_dir() / "sessions" / session_id / "history"
        history.parent.mkdir(parents=True, exist_ok=True)
        with shelve.open(history) as db:
            colleague_session_ids: dict[str, str] = db.get("colleagues", {})
    for child_id in config.agent.colleagues:
        child_path = __get_config_path(file.parent, child_id)
        colleague = __load_agent_from_config(
            child_path,
            pending,
            agents,
            persist,
            colleague_session_ids.get(child_id) if persist else None,
        )
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
        persist=persist,
        session_id=session_id,
        log_level=log_level,
    )
    agent.config = config
    agent.config_path = file.resolve()
    # Load history
    if persist and session_id:
        agent.load()
    pending.remove(file)
    agents[file] = agent
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


def load_agent_from_config(
    name: str | Path,
    persist: bool,
    session_id: str | None,
    log_level: str | int | None = None,
) -> Agent:
    config_dir = get_config_dir()
    """Load a bot from a configuration file"""
    if isinstance(name, Path):
        if not name.exists():
            raise FileNotFoundError(f"Agent config file not found: {name}")
        config_path = name.resolve()
    elif name.endswith(".toml"):
        # name is also a path
        config_path = Path(name)
        if not config_path.exists() or not config_path.is_file():
            raise FileNotFoundError(f"Agent config file not found: {name}")
        config_path = config_path.resolve()
    elif (s := Path(name).suffix) and s != ".toml":
        raise ValueError(f"Invalid agent path: {name}")
    else:
        # If the name is a string, we need to find the configuration file from a list of search paths
        if (file := config_dir / f"{name}.toml").exists():
            config_path = file
        else:
            raise FileNotFoundError(f"Agent config not found: {name}")
    if config_path.stem.startswith(("_", ".", "-")):
        raise ValueError(f"Invalid agent file name: {config_path.stem}")
    return __load_agent_from_config(
        config_path, set(), {}, persist, session_id, log_level
    )


def find_all_agents() -> list[AgentInfo]:
    """Find all agents in the search paths"""
    agents: dict[str, AgentInfo] = {}
    config_dir = get_config_dir()
    for file in config_dir.glob("*.toml"):
        if file.stem in agents:
            continue
        if file.stem.startswith(("_", ".", "-")):
            continue
        doc = tomllib.loads(file.read_text())
        if not isinstance(doc, dict) or "agent" not in doc:
            continue
        try:
            config = Config(**doc)
        except ValidationError as e:
            raise ValueError(f"Invalid config file: {file}\n{repr(e)}") from e
        agent_info = AgentInfo(
            id=file.stem,
            config=config,
            config_path=file.resolve().relative_to(Path.cwd()),
        )
        agents[file.stem] = agent_info
    agents_list = list(agents.values())
    agents_list.sort(key=lambda x: x.config.agent.name)
    return agents_list


def save(path: Path, doc: tomlkit.TOMLDocument):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w+") as f:
        tomlkit.dump(doc, f)


def load(path: Path) -> tomlkit.TOMLDocument:
    if not path.exists():
        raise FileNotFoundError(f"Agent config file not found: {path}")
    with path.open("r") as f:
        doc = tomlkit.load(f)
    return doc


def set_session_tags(
    session_id: str,
    tags: list[str],
):
    """Set the tags for a session"""
    for t in tags:
        if " " in t or "," in t:
            raise ValueError(f"Invalid tag: {t}")
    session_dir = _get_global_cache_dir() / "sessions" / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    with open(session_dir / "tags", "w+") as f:
        f.write(",".join(tags))


def get_session_tags(session_id: str) -> list[str]:
    """Get the tags for a session"""
    session_dir = _get_global_cache_dir() / "sessions" / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    tags_file = session_dir / "tags"
    if not tags_file.exists():
        return []
    with open(tags_file, "r") as f:
        tags = f.read().strip().split(",")
    return [t.strip() for t in tags if t.strip()]


ALL_RECOMMENDED_MODELS = [
    "deepseek/deepseek-chat-v3-0324",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/o1",
    "openai/o3-mini",
    "openai/o3-mini-high",
]
