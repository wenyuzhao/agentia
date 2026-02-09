import os
from pathlib import Path
from typing import Sequence
from agentia import (
    Agent,
    MessagePartText,
    UserMessage,
    UserMessagePart,
    MessagePartFile,
)
from agentia.plugins.skills import Skill
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient
from dotenv import load_dotenv
import frontmatter

load_dotenv()


class SkillScript(BaseModel):
    name: str = Field(
        description="The name of the script (without the path), e.g. 'data_processing.py'"
    )
    content: str = Field(
        description="The content of the script as a string. Must be valid Python code. Inputs are from CLI arguments. Outputs are through stdout (using print) either in JSON or raw text format."
    )


class SkillReference(BaseModel):
    name: str = Field(description="The name of the reference document, with .md suffix")
    description: str = Field(description="A brief description of the skill")
    content: str = Field(
        description="The markdown content of the reference document as a string."
    )


class SkillInfo(BaseModel):
    description: str = Field(description="A brief description of the skill")
    readme: str = Field(
        description="""
        A detailed description of the skill, including its detailed descriptions, capabilities, and usage instructions (or experiences).
        This should be in markdown format that can be reused by other LLM agents to follow/apply the skill.

        # Scripts

        Use scripts to provide executable code that can be used to perform tasks.

        When providing scripts:
            In this README, **add a section** that lists each script path and how to use it, so that other agents can use them.
            In this README, please specify the expected CLI arguments, expected output, and provide examples.

        Script paths should be `scripts/<name>.py`

        # Reference documents

        Use reference docs to provide more detailed information about the skill, such as case studies, examples, detailed explanations, etc.
        This can help other agents better understand the skill and apply it more effectively.

        When providing reference docs:
            In this README, **add a section** that lists each reference doc path and their short description, so that other agents can load it when needed to strengthen the skill.

        Please provide a brief summary of the key points in the document and how it can be used to strengthen the skill.

        Reference document paths should be `references/<name>.md`
        """
    )
    scripts: list[SkillScript] = Field(
        default_factory=list,
        description="(Optional, only when applicable) A list of scripts provided by the skill, if any. Scripts are placed in the 'scripts' folder.",
    )
    reference_docs: list[SkillReference] = Field(
        default_factory=list,
        description="(Optional, only when applicable) A list of reference documents for the skill (as appendix or supplementary materials), if any.",
    )

    def save(self, name: str, out: Path) -> Skill:
        out.mkdir(parents=True, exist_ok=True)

        if any(out.iterdir()):
            raise ValueError(f"Output directory '{out}' is not empty")

        assert out.is_dir(), f"Output path '{out}' must be a directory"
        skills_md = frontmatter.dumps(
            frontmatter.Post(
                content=self.readme, name=name, description=self.description
            )
        )
        (out / "SKILL.md").write_text(skills_md, encoding="utf-8")
        if self.scripts:
            (out / "scripts").mkdir(exist_ok=True)
        for f in self.scripts:
            if not f.name.endswith(".py"):
                f.name += ".py"
            (out / "scripts" / f.name).write_text(f.content, encoding="utf-8")
        if self.reference_docs:
            (out / "references").mkdir(exist_ok=True)
        for r in self.reference_docs:
            if not r.name.endswith(".md"):
                r.name += ".md"
            doc = frontmatter.dumps(
                frontmatter.Post(
                    content=r.content, name=r.name, description=r.description
                )
            )
            (out / "references" / r.name).write_text(doc, encoding="utf-8")

        skill = Skill(
            path=out,
            name=name,
            description=self.description,
            instructions=self.readme,
            script_paths=[f.name for f in self.scripts],
            resource_paths=[f.name for f in self.reference_docs],
        )
        return skill


async def learn_skill_from_documents(
    docs: Sequence[Path | str], name: str, prompt: str, out: Path, model: str
) -> Skill:
    instructions = "You are a skill generator/learner. Your task is to learn a skill based on the given documents. Output the skill in JSON format\n\n"
    instructions += f"Skill Name: {name}\n\n"
    instructions += f"{prompt}"

    agent = Agent(model=model, instructions=instructions)

    parts: list[UserMessagePart] = []

    urls: list[str] = []
    files: list[Path] = []
    allowed_suffixes = [".pdf", ".md", ".markdown", ".txt"]
    for x in docs:
        if isinstance(x, str) and x.startswith(("http://", "https://")):
            urls.append(x)
        else:
            p = Path(x)
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in allowed_suffixes:
                        files.append(f)

    # retrieve content from URLs using tavily
    if urls:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required to fetch web content")
        tavily = AsyncTavilyClient(api_key=tavily_api_key)
        try:
            result = await tavily.extract(urls=urls, extract_depth="advanced")
            results = result.get("results", [])
            for result in results:
                content = result["raw_content"]
                url = result["url"]
                text = "# Web Content: " + url + "\n\n" + content
                parts.append(MessagePartText(text=text))
        except Exception as e:
            raise ValueError(f"Failed to fetch web content: {e}")

    # retrieve content from files
    for doc in files:
        content = doc.read_text()
        media_type = "text/plain"
        suffix = doc.suffix.lower()
        if suffix in [".pdf"]:
            media_type = "application/pdf"
            parts.append(
                MessagePartFile(filename=doc.name, data=content, media_type=media_type)
            )
        elif suffix in [".md", ".markdown", ".txt"]:
            text = "# Document: " + doc.name + "\n\n" + content
            parts.append(MessagePartText(text=text))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    if not parts:
        raise ValueError("No valid documents provided to learn the skill")

    skill_info = await agent.generate_object(UserMessage(parts), SkillInfo)

    skill = skill_info.save(name=name, out=out)

    return skill
