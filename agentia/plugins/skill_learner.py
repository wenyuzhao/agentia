from pathlib import Path
from agentia.utils.skill_learner import SkillInfo
from . import Plugin, tool
from typing import Annotated


class SkillLearner(Plugin):
    def __init__(self, save_dir: Path | str | None = None) -> None:
        if save_dir is None:
            save_dir = Path.cwd() / ".skills"
        self.save_dir = Path(save_dir)

    @tool(name="generate_skill")
    async def generate_skill(
        self,
        name: Annotated[
            str,
            "The name of the skill. lowercase, no space, hyphen separated. Must be unique among all skills.",
        ],
        skill: SkillInfo,
    ):
        """
        Create/generate a skill based on your prior knowledge. The skills can then be reused either by you or other LLMs to solve problems or offer guidance within the scope of the skill.
        """
        from agentia.plugins.skills import Skills

        assert self.llm and self.llm._active_tools
        existing_skills: set[str] = set()
        skills: Skills | None = self.llm._active_tools.get_plugin(Skills)
        if skills:
            existing_skills = set(skills.skills.keys())

        s = skill.save(name=name, out=self.save_dir / name)

        if name in existing_skills:
            raise ValueError(f"A skill with the name '{name}' already exists")

        if skills:
            skills.add_skill(s)

        return {"name": s.name, "description": s.description, "loaded": False}

    @tool(name="learn_skill_from_documents")
    async def learn_skill_from_documents(
        self,
        name: Annotated[
            str,
            "The name of the skill. lowercase, no space, hyphen separated. Must be unique among all skills.",
        ],
        docs: Annotated[
            list[str],
            "A list of documents to learn the skill from. Each document can be a pdf/txt/markdown file path or a URL.",
        ],
        prompt: Annotated[
            str,
            "Additional instructions or information to guide the learning of the skill. e.g. what skill to learn or focus on, etc.",
        ],
    ):
        """Learn or acquire a skill from a list of documents"""
        from agentia.utils.skill_learner import learn_skill_from_documents
        from agentia.plugins.skills import Skills

        assert self.llm and self.llm._active_tools

        for d in docs:
            if not d.startswith(("http://", "https://")):
                if not Path(d).is_file():
                    raise ValueError(f"Document '{d}' does not exist")

        existing_skills: set[str] = set()
        skills: Skills | None = self.llm._active_tools.get_plugin(Skills)
        if skills:
            existing_skills = set(skills.skills.keys())

        if name in existing_skills:
            raise ValueError(f"A skill with the name '{name}' already exists")

        path = self.save_dir / name
        path.mkdir(parents=True, exist_ok=True)

        skill = await learn_skill_from_documents(
            docs=docs, name=name, prompt=prompt, out=path, model=self.llm.model
        )

        if skills:
            skills.add_skill(skill)

        return {"name": skill.name, "description": skill.description, "loaded": False}
