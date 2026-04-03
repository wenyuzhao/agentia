from . import Plugin
from pathlib import Path
from typing import Sequence, override
import frontmatter
from pydantic import BaseModel

INSTRUCTIONS_TEMPLATE = """\
You have access to a set of skills to extend your ability.
To use a skill, read its SKILL.md file carefully and follow the instructions.
Load only the necessary skills to complete the tasks.

# AVAILABLE SKILLS
{skills}
"""


class Skill(BaseModel):
    path: Path
    name: str
    description: str
    loaded: bool = False


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

    def load_all_skills(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        for path in self.search_paths:
            if not path.is_dir():
                continue
            for skill in self.discover_skills(path):
                skills[skill.name] = skill
        return skills

    def is_skill(self, path: Path) -> bool:
        if path.name.startswith(".") or path.name.startswith("_"):
            return False
        if not (path / "SKILL.md").is_file():
            return False
        return True

    def discover_skills(self, path: Path) -> list[Skill]:
        """Recursively discover skills from the given path."""
        assert path.is_dir(), f"'{path}' is not a directory"
        if self.is_skill(path):
            return [self.load_skill(path)]
        skills = []
        for child in path.iterdir():
            if child.is_dir():
                skills.extend(self.discover_skills(child))
        return skills

    def load_skill(self, path: Path) -> Skill:
        path = path.resolve()
        md = path / "SKILL.md"
        assert md.is_file(), f"Skill metadata file '{md}' does not exist"
        doc = frontmatter.loads(md.read_text(encoding="utf-8"))
        name = doc.metadata.get("name", path.stem)
        assert isinstance(name, str), f"Skill name must be a string in '{md}'"
        description = doc.metadata.get("description", "")
        assert isinstance(
            description, str
        ), f"Skill description must be a string in '{md}'"
        return Skill(path=path, name=name, description=description)

    @override
    def get_instructions(self) -> str:
        skills = self.load_all_skills()
        s = ""
        for skill in skills.values():
            s += f"- **{skill.name}**:\n    * path: {str(skill.path/"SKILL.md")}\n    * description: {skill.description}\n\n"
        return INSTRUCTIONS_TEMPLATE.format(skills=s)
