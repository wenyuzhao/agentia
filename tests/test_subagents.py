from agentia import Agent, CommunicationEvent
import pytest
import dotenv

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_subagents():
    coder = Agent(
        model="openai/gpt-4o-mini",
        name="Coder",
        description="programmar",
    )
    reviewer = Agent(
        model="openai/gpt-4o-mini",
        name="Code Reviewer",
        description="code reviewer",
    )
    leader = Agent(
        model="openai/gpt-4o-mini",
        name="Leader",
        description="Leader agent",
        subagents=[
            coder,
            reviewer,
        ],
    )

    run = leader.run(
        "ask your subagents to 1. write a quick sort algorithm in python; 2. review it once; and 3. improve it once and return the final version",
        events=True,
    )

    leader_to_coder = False
    leader_to_reviewer = False
    async for e in run:
        print(e)
        if isinstance(e, CommunicationEvent):
            assert e.parent == leader.id
            assert e.child in [coder.id, reviewer.id]
            if e.child == coder.id:
                leader_to_coder = True
            else:
                leader_to_reviewer = True
    assert leader_to_coder and leader_to_reviewer
    # print(message)
