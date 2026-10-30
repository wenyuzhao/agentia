from agentia.plugins.skills import Skill
from agentia import Agent
import pytest
import dotenv
from agentia.plugins import System

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_skills():
    agent = Agent(
        model="anthropic/claude-sonnet-4.6", skills=["./examples"], tools=[System()]
    )
    run = agent.run("What is the weather and temperature like in boston?")
    all_assistant_content: str = ""
    async for msg in run:
        if msg.role == "assistant":
            all_assistant_content += msg.text
        print(msg)
    assert "27" in all_assistant_content


def test_skill_execute_uses_the_arguments_of_each_call(tmp_path):
    skill_dir = tmp_path / "greet"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: greet\ndescription: Greet\n---\n\nHello $0 from !`printf shell`"
    )
    skill = Skill.load(skill_dir)
    assert "Hello Sydney from shell" in skill.execute(["Sydney"])
    assert "Hello Paris from shell" in skill.execute(["Paris"])
