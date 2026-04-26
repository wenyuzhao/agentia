from . import Plugin, tool
from pathlib import Path
from typing import Any, Sequence, override
from pydantic import BaseModel
from agentia.utils.markdown import MarkdownDoc, load_markdown

INSTRUCTIONS_TEMPLATE = """\
# SKILLS

Skills provide specialized capabilities and domain knowledge.

When users ask you to perform tasks, check if any of the available skills match.

Use the `Skill` tool to execute a skill. Load only the necessary skills to complete the tasks.

## AVAILABLE SKILLS
{skills}
"""


class Skill(BaseModel):
    path: Path
    name: str
    description: str | None = None
    metadata: dict[str, Any]
    doc: MarkdownDoc | None = None

    @property
    def loaded(self) -> bool:
        return self.doc is not None

    @staticmethod
    def load(path: Path) -> "Skill":
        md = path / "SKILL.md"
        assert md.is_file(), f"Skill metadata file '{md}' does not exist"
        doc = load_markdown(str(md))
        if not ((name := doc.metadata.get("name")) and isinstance(name, str)):
            raise ValueError(f"Skill name must be a string in '{md}'")
        desc = doc.metadata.get("description", None)
        if desc is not None and not isinstance(desc, str):
            raise ValueError(f"Skill description must be a string in '{md}'")
        return Skill(
            path=path,
            name=name,
            description=desc,
            metadata=doc.metadata,
            doc=doc,
        )

    @property
    def disable_model_invocation(self) -> bool:
        return self.metadata.get("disable-model-invocation", False)

    @property
    def user_invocable(self) -> bool:
        return self.metadata.get("user-invocable", True)

    def execute(self, args: list[str] | None = None) -> str:
        if self.doc:
            return self.doc.content
        self.doc = load_markdown(
            str(self.path / "SKILL.md"), args=args, substitutions=["args", "bash"]
        )
        return self.doc.content


class Skills(Plugin):
    def __init__(self, search_paths: Sequence[Path | str] | None = None) -> None:
        if not search_paths:
            # default search path
            search_paths = [
                Path.cwd() / ".skills",
                Path.cwd() / ".agentia" / "skills",
                Path.home() / ".config" / "agentia" / "skills",
            ]
        self.search_paths = [Path(p) for p in search_paths]
        self.user_invocable_skills: dict[str, Skill] = {}
        self.agent_skills: dict[str, Skill] = {}

    @override
    async def init(self):
        all_skills = self.load_all_skills(self.search_paths)
        for skill in all_skills.values():
            if skill.user_invocable:
                self.user_invocable_skills[skill.name] = skill
            if not skill.disable_model_invocation:
                self.agent_skills[skill.name] = skill

    def load_all_skills(self, search_paths: list[Path]) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        for path in search_paths:
            if not path.is_dir():
                continue
            for skill in self.discover_skills(path):
                skills[skill.name] = skill
        return skills

    def is_skill(self, path: Path) -> bool:
        # Skip hidden directories and files
        if path.name.startswith(".") or path.name.startswith("_"):
            return False
        # Skip directories without SKILL.md
        if not (path / "SKILL.md").is_file():
            return False
        return True

    def ignore_skill(self, skill: Skill) -> bool:
        return False

    def discover_skills(self, path: Path) -> list[Skill]:
        """Recursively discover skills from the given path."""
        assert path.is_dir(), f"'{path}' is not a directory"
        if self.is_skill(path):
            skill = Skill.load(path)
            if self.ignore_skill(skill):
                return []
            return [skill]
        skills = []
        for child in path.iterdir():
            if child.is_dir():
                skills.extend(self.discover_skills(child))
        return skills

    @override
    def get_instructions(self) -> str:
        s = ""
        for skill in self.agent_skills.values():
            if skill.disable_model_invocation:
                continue
            s += f"- **{skill.name}**:\n    * path: {str(skill.path)}\n    * description: {skill.description}\n\n"
        return INSTRUCTIONS_TEMPLATE.format(skills=s)

    @tool(name="Skill")
    def execute_skill(self, skill_name: str):
        """
        Execute a skill within the main conversation. All available skills are listed in system instructions.

        How to invoke:
            - Invoke the tool with `skill_name` set to the exact name of an available skill (no leading slash).

        Returns the content of the skill's SKILL.md after processing any directives with the provided arguments.
            - For file-references (`@path/to/file`, either absolute path or relative to the skill path), use the Read tool to access the file content.
            - Always use this tool to execute skills instead of loading SKILL.md directly.
        """
        skill = self.agent_skills.get(skill_name)
        if not skill:
            raise ValueError(f"Skill '{skill_name}' not found")
        if skill.disable_model_invocation:
            raise ValueError(f"Skill '{skill_name}' is disabled for model invocation")
        result_content = skill.execute([])
        return result_content
