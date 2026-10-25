"""Skills: folders with a SKILL.md (frontmatter `name`, `description`, plus instructions).

The `Skills` plugin lists the available skills in the system prompt and gives the model
a tool to load one. Where skills live is up to a `SkillSource`: `LocalSkillSource`
scans local folders; hosts whose skills live elsewhere (a sandbox, a database)
implement `list_skills()` and `read_skill()` themselves.
"""

import abc
import posixpath
from collections.abc import Sequence
from pathlib import Path
from typing import Any, override

from pydantic import BaseModel, Field

from agentia.tools.tools import BaseTool, ModelTool, ToolCall
from agentia.utils.markdown import load_markdown

from . import Plugin

INSTRUCTIONS = """\
# Skills
Skills are folders with a SKILL.md. When a task matches a skill, load it with the `{tool}` tool first, then follow it.
{skills}"""


class Skill(BaseModel):
    name: str
    description: str | None = None
    path: str
    """The skill's SKILL.md file."""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def folder(self) -> str:
        return posixpath.dirname(self.path) if "/" in self.path else "."

    @property
    def disable_model_invocation(self) -> bool:
        return bool(self.metadata.get("disable-model-invocation", False))

    @property
    def user_invocable(self) -> bool:
        return bool(self.metadata.get("user-invocable", True))

    @staticmethod
    def load(path: Path) -> "Skill":
        """Read a skill's metadata from a folder (or its SKILL.md)."""
        md = path if path.name == "SKILL.md" else path / "SKILL.md"
        assert md.is_file(), f"Skill metadata file '{md}' does not exist"
        doc = load_markdown(str(md))
        if not ((name := doc.metadata.get("name")) and isinstance(name, str)):
            raise ValueError(f"Skill name must be a string in '{md}'")
        desc = doc.metadata.get("description", None)
        if desc is not None and not isinstance(desc, str):
            raise ValueError(f"Skill description must be a string in '{md}'")
        return Skill(name=name, description=desc, path=str(md), metadata=doc.metadata)


class SkillSource(abc.ABC):
    @abc.abstractmethod
    async def list_skills(self) -> Sequence[Skill]: ...

    @abc.abstractmethod
    async def read_skill(self, skill: Skill, args: list[str] | None = None) -> str:
        """The skill's instructions (SKILL.md, processed)."""


class LocalSkillSource(SkillSource):
    """Skills in local folders, searched recursively. Later paths override earlier
    ones with the same skill name."""

    def __init__(self, search_paths: Sequence[Path | str] | None = None) -> None:
        if not search_paths:
            search_paths = [
                Path.cwd() / ".skills",
                Path.cwd() / ".agentia" / "skills",
                Path.home() / ".config" / "agentia" / "skills",
            ]
        self.search_paths = [Path(p) for p in search_paths]

    def scan(self) -> list[Skill]:
        skills: dict[str, Skill] = {}
        for path in self.search_paths:
            if path.is_dir():
                for s in self._discover(path):
                    skills[s.name] = s
        return list(skills.values())

    def _discover(self, path: Path) -> list[Skill]:
        if (path / "SKILL.md").is_file():
            return [Skill.load(path)]
        out: list[Skill] = []
        for child in sorted(path.iterdir()):
            if child.is_dir() and not child.name.startswith((".", "_")):
                out += self._discover(child)
        return out

    @override
    async def list_skills(self) -> list[Skill]:
        return self.scan()

    @override
    async def read_skill(self, skill: Skill, args: list[str] | None = None) -> str:
        return load_markdown(
            skill.path, args=args or [], substitutions=["args", "bash"]
        ).content


class _SkillArgs(BaseModel):
    name: str = Field(description="Skill name (no leading slash)")


class Skills(Plugin):
    def __init__(
        self,
        search_paths: Sequence[Path | str] | None = None,
        *,
        source: SkillSource | None = None,
        tool_name: str = "Skill",
    ) -> None:
        super().__init__()
        if source is not None and search_paths is not None:
            raise ValueError("pass either search_paths or source")
        self.source = source or LocalSkillSource(search_paths)
        self.tool_name = tool_name
        self._skills: list[Skill] | None = None

    async def skills(self, refresh: bool = False) -> list[Skill]:
        """All skills (cached; `refresh=True` re-reads the source)."""
        if self._skills is None or refresh:
            self._skills = list(await self.source.list_skills())
        return self._skills

    async def find(self, name: str) -> Skill | None:
        return next((s for s in await self.skills() if s.name == name), None)

    async def agent_skills(self) -> list[Skill]:
        """Skills the model may load."""
        return [s for s in await self.skills() if not s.disable_model_invocation]

    async def user_invocable_skills(self) -> dict[str, Skill]:
        """Skills the user may invoke as `/<name>`."""
        return {s.name: s for s in await self.skills() if s.user_invocable}

    async def execute(self, skill: Skill, args: list[str] | None = None) -> str:
        """The skill's instructions, headed with its name and folder (relative file
        references in a skill resolve against its folder)."""
        body = await self.source.read_skill(skill, args)
        return f"# Skill {skill.name} ({skill.folder})\n\n{body}"

    @override
    async def init(self):
        await self.skills(refresh=True)

    @override
    async def get_instructions(self) -> str | None:
        skills = await self.agent_skills()
        if not skills:
            return None
        lines = "\n".join(
            f"- {s.name}: {s.description or ''} ({s.path})" for s in skills
        )
        return INSTRUCTIONS.format(tool=self.tool_name, skills=lines)

    @override
    def tools(self) -> list[BaseTool]:
        async def run(_ctx: ToolCall, a: _SkillArgs) -> str:
            skill = await self.find(a.name.removeprefix("/"))
            if not skill or skill.disable_model_invocation:
                names = ", ".join(s.name for s in await self.agent_skills())
                raise ValueError(
                    f"unknown skill '{a.name}'. Available: {names or 'none'}"
                )
            return await self.execute(skill)

        return [
            ModelTool(
                self.tool_name,
                "Load a skill's instructions by name (see the Skills list in your instructions), then follow them.",
                _SkillArgs,
                run,
            )
        ]


__all__ = ["LocalSkillSource", "Skill", "SkillSource", "Skills"]
