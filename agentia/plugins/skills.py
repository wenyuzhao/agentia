from . import Plugin, tool
from pathlib import Path
from typing import Any, Sequence, override
import frontmatter
from pydantic import BaseModel
import subprocess
import os

INSTRUCTIONS_TEMPLATE = """\
You have access to a set of skills, each with its own instructions, capabilities, resources, and scripts for completing tasks within its spcific domain.

# AVAILABLE SKILLS
{skills}

# SKILL USAGE GUIDELINES
When a task falls within the domain of a skill,
1. Use `load_skill` to get the skill instructions and other detailed information.
2. Follow the skill instructions carefully to complete the task.
3. You can also use the resources and scripts provided by the skill to help you complete the task.

IMPORTANT: Load only the necessary skills to complete the task.
"""


class Skill(BaseModel):
    path: Path
    name: str
    description: str
    instructions: str
    resource_paths: list[str] = []
    script_paths: list[str] = []
    loaded: bool = False

    def load_resource(self, rel_path: str) -> str:
        if rel_path not in self.resource_paths:
            raise ValueError(f"Resource '{rel_path}' not found in skill '{self.name}'")
        resource_path = self.path / rel_path
        return resource_path.read_text(encoding="utf-8")

    def run_script(self, rel_path: str, args: list[str]) -> tuple[int, str, str]:
        if rel_path not in self.script_paths:
            raise ValueError(f"Script '{rel_path}' not found in skill '{self.name}'")
        script_path = self.path / rel_path
        # run the script with the given arguments and return the output and stderr
        result = subprocess.run(
            ["python", str(script_path), *args],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        return result.returncode, result.stdout, result.stderr


class Skills(Plugin):
    def __init__(self, search_paths: Sequence[Path | str] | None = None) -> None:
        if not search_paths:
            # default search path
            search_paths = [
                Path.cwd() / ".skills",
                Path.cwd() / ".agentia" / "skills",
                Path.home() / ".local" / "share" / "agentia" / "skills",
            ]
        self.__search_paths = [Path(p) for p in search_paths]
        # Load skills metadata from search paths
        self.skills: dict[str, Skill] = {}
        for path in self.__search_paths:
            if not path.is_dir():
                continue
            for skill in self.__load_skill_recursive(path, no_ignore=True):
                # assert skill.name not in self.skills, f"Duplicate skill name '{skill.name}' found in '{skill.path}'"
                self.skills[skill.name] = skill

    def __load_skill_recursive(self, path: Path, no_ignore=False) -> list[Skill]:
        assert path.is_dir(), f"'{path}' is not a directory"
        if not no_ignore and (path.name.startswith(".") or path.name.startswith("_")):
            return []
        if (path / "SKILL.md").is_file():
            return [self.__load_skill(path)]
        skills = []
        for child in path.iterdir():
            if child.is_dir():
                skills.extend(self.__load_skill_recursive(child))
        return skills

    def __load_skill(self, path: Path) -> Skill:
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
        instructions = doc.content
        # resources
        resources: list[str] = []
        supported_extensions = [".md", ".json", ".yaml", ".yml", ".csv", ".xml", ".txt"]
        excluded_files = {"SKILL.md"}
        for ext in supported_extensions:
            for p in (path / "resources").rglob(f"*{ext}"):
                if p.is_file() and p.name not in excluded_files:
                    try:
                        p = p.resolve()
                        rel_path = p.relative_to(path)
                        resources.append(rel_path.as_posix())
                    except Exception as e:
                        ...
        # scripts
        scripts: list[str] = []
        for p in (path / "scripts").rglob(f"*.py"):
            if p.is_file() and p.name != "__init__.py":
                try:
                    p = p.resolve()
                    rel_path = p.relative_to(path)
                    scripts.append(rel_path.as_posix())
                except Exception as e:
                    ...
        return Skill(
            path=path,
            name=name,
            description=description,
            instructions=instructions,
            resource_paths=resources,
            script_paths=scripts,
        )

    @override
    def get_instructions(self) -> str:
        s = ""
        for skill in self.skills.values():
            s += f"- **{skill.name}**:\n{skill.description}\n\n"
        return INSTRUCTIONS_TEMPLATE.format(skills=s)

    # @tool(name="list_skills")
    def list_skills(self, skill_name: str) -> Any:
        """
        Get the list of available skills, their descriptions, and whether they are loaded.
        Use this tool to discover what skills are available and their descriptions, to refresh your memory.
        """
        skills = list(self.skills.values())
        skill_docs = []
        for skill in skills:
            skill_docs.append(
                {
                    "name": skill.name,
                    "description": skill.description,
                    "loaded": skill.loaded,
                }
            )
        return skill_docs

    @tool(name="load_skill")
    def load_skill(self, skill_name: str) -> Any:
        """
        Load the instructions and capabilities of a skill by its name.

        Also lists the resources and scripts provided by the skill, if any, so that you can use them later.

        Call this tool (only once) when you need to perform a task within the skill's domain.
        """
        if skill_name not in self.skills:
            return {"error": f"Skill '{skill_name}' not found"}
        skill = self.skills[skill_name]
        doc = skill.model_dump()
        del doc["path"]
        return doc

    @tool(name="load_skill_resource")
    def load_skill_resource(self, skill_name: str, resource_path: str) -> Any:
        """
        Load a resource provided by a skill.

        `resource_path` is the path of the resource within the skill directory.

        Call this tool when you need to use a specific resource provided by a skill.
        """
        if skill_name not in self.skills:
            return {"error": f"Skill '{skill_name}' not found"}
        skill = self.skills[skill_name]
        try:
            content = skill.load_resource(resource_path)
            return {"content": content}
        except Exception as e:
            return {"error": str(e)}

    @tool(name="run_skill_script")
    def run_skill_script(
        self, skill_name: str, script_path: str, args: list[str]
    ) -> Any:
        """
        Run a script provided by a skill with the given arguments.

        `script_path` is the path of the script within the skill's `scripts` directory.

        Call this tool when you need to execute a specific script provided by a skill to complete a task.
        """
        if skill_name not in self.skills:
            return {"error": f"Skill '{skill_name}' not found"}
        skill = self.skills[skill_name]
        try:
            code, stdout, stderr = skill.run_script(script_path, args)
            if code != 0:
                return {
                    "error": f"Script exited with code {code}",
                    "stdout": stdout,
                    "stderr": stderr,
                }
            else:
                doc = {"returncode": code, "stdout": stdout}
                if stderr:
                    doc["stderr"] = stderr
                return doc
        except Exception as e:
            return {"error": str(e)}
