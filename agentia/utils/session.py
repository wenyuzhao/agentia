from pathlib import Path
import shelve
import shutil
from typing import Optional
import weakref
from filelock import BaseFileLock, FileLock
from pydantic import BaseModel, Field
import os

from agentia.agent import Agent

_global_cache_dir = None


def get_global_cache_dir() -> Path:
    global _global_cache_dir
    if _global_cache_dir is not None:
        return _global_cache_dir
    # Check if the environment variable is set
    env_cache_dir = os.getenv("AGENTIA_CACHE_DIR")
    if env_cache_dir:
        _global_cache_dir = Path(env_cache_dir)
    else:
        _global_cache_dir = Path.cwd() / ".cache"
    return _global_cache_dir


class SessionInfo(BaseModel):
    id: str
    agent: str
    title: str | None = None
    tags: list[str] = Field(default_factory=list)

    @staticmethod
    def from_agent(agent: Agent) -> "SessionInfo":
        """Create a SessionInfo object from an agent"""
        return SessionInfo(
            id=agent.session_id,
            agent=agent.id,
            title=agent.history.summary,
            tags=get_session_tags(agent.session_id),
        )


def load_session_info(session: str) -> SessionInfo | None:
    """Get the session info"""
    session_dir = get_global_cache_dir() / "sessions" / session
    if not session_dir.exists():
        return None
    config_file = session_dir / "config"
    if not config_file.exists():
        return None
    with shelve.open(config_file) as db:
        sess_title = db.get("title", None)
        sess_agent = db["agent"]
    tags = get_session_tags(session)
    return SessionInfo(id=session, agent=sess_agent, title=sess_title, tags=tags)


def load_session(session: str) -> Agent | None:
    """Get the session data"""
    session_dir = get_global_cache_dir() / "sessions" / session
    if not session_dir.exists():
        return None
    config_file = session_dir / "config"
    if not config_file.exists():
        return None
    with shelve.open(config_file) as db:
        sess_data = db.get("data", None)
    return sess_data


def get_all_sessions(agent: str) -> list[SessionInfo]:
    sessions_dir = get_global_cache_dir() / "sessions"
    if not sessions_dir.exists():
        return []
    sessions: list[SessionInfo] = []
    for entry in sessions_dir.iterdir():
        if entry.is_dir():
            session_id = entry.stem
            if session := load_session_info(session_id):
                if not agent or session.agent == agent:
                    sessions.append(session)
    # Sort in descending order
    sessions.sort(reverse=True, key=lambda x: x.id)
    return sessions


def delete_session(id: str):
    """Delete a session"""
    session_dir = get_global_cache_dir() / "sessions" / id
    if not session_dir.exists():
        return
    shutil.rmtree(session_dir)


def delete_agent(id: str):
    """Delete the agent"""
    from agentia.utils.config import get_config_dir

    # delete all sessions
    sessions = get_all_sessions(id)
    for session in sessions:
        delete_session(session.id)
    # delete the agent
    agent_dir = get_global_cache_dir() / "agents" / id
    if not agent_dir.exists():
        print(f"Agent {id} not found")
        return
    shutil.rmtree(agent_dir)
    agent_config_file = get_config_dir() / f"{id}.toml"
    if agent_config_file.exists():
        os.remove(agent_config_file)


def cleanup_cache():
    """Delete all stale sessions and agents"""
    import agentia.utils.config as config

    all_agents = config.get_all_agents()
    agent_ids = set([a.id for a in all_agents])
    # delete all stale agents
    agents_dir = get_global_cache_dir() / "agents"
    if agents_dir.exists():
        for entry in agents_dir.iterdir():
            if entry.is_dir():
                agent_id = entry.stem
                if agent_id not in agent_ids:
                    shutil.rmtree(entry)
    # delete all stale sessions
    sessions_dir = get_global_cache_dir() / "sessions"
    if sessions_dir.exists():
        for entry in sessions_dir.iterdir():
            if entry.is_dir():
                session_id = entry.stem
                session = load_session_info(session_id)
                if session and session.agent not in agent_ids:
                    shutil.rmtree(entry)


def set_session_tags(
    session_id: str,
    tags: list[str],
):
    """Set the tags for a session"""
    for t in tags:
        if " " in t or "," in t:
            raise ValueError(f"Invalid tag: {t}")
    session_dir = get_global_cache_dir() / "sessions" / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    with open(session_dir / "tags", "w+") as f:
        f.write(",".join(tags))


def get_session_tags(session_id: str) -> list[str]:
    """Get the tags for a session"""
    session_dir = get_global_cache_dir() / "sessions" / session_id
    if not session_dir.exists():
        return []
    tags_file = session_dir / "tags"
    if not tags_file.exists():
        return []
    with open(tags_file, "r") as f:
        tags = f.read().strip().split(",")
    return [t.strip() for t in tags if t.strip()]


class SessionLock:
    """Session lock to prevent concurrent access to the session data"""

    def __init__(self, agent: Agent):
        self.__agent = agent
        self.__lock = None
        self.lock()
        weakref.finalize(self, SessionLock.__finalize, self.__lock)

    @staticmethod
    def __finalize(lock: Optional[BaseFileLock]):
        if lock is not None and lock.is_locked:
            lock.release()

    def lock(self):
        """Lock the session"""
        if self.__agent.persist:
            self.__agent.session_data_folder.mkdir(parents=True, exist_ok=True)
            self.__lock = FileLock(self.__agent.session_data_folder / "lock")
            self.__lock.acquire()

    def unlock(self):
        """Unlock the session"""
        if self.__agent.persist and self.__lock is not None:
            self.__lock.release()
            self.__lock = None


def load_history(agent: Agent, subagents=True):
    """Load the history for a session"""
    if not agent.persist:
        return
    history_file = agent.session_data_folder / "history"
    if not history_file.exists():
        return
    agent.history._load(history_file)
    if subagents:
        with shelve.open(history_file) as db:
            for k, v in db.get("subagents", {}).items():
                subagent = agent.subagents.get(k)
                if subagent is None:
                    continue
                subagent.session_id = v
                load_history(subagent, False)
