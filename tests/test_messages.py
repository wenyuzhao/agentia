import base64
import json

import dotenv
import pytest
from pydantic import TypeAdapter

from agentia import Agent
from agentia.history import History
from agentia.llm.providers._openai_api import (
    parse_output_item,
    to_responses_input,
)
from agentia.models import (
    AssistantMessage,
    File,
    Message,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolMessage,
    ToolResultPart,
    UrlCitation,
    UserMessage,
)

dotenv.load_dotenv()


def test_user_message_coerces_content():
    m = UserMessage("hello")
    assert m.content == [TextPart("hello")]
    assert m.text == "hello"

    f = File("https://example.com/cat.png")
    m = UserMessage(["What is this?", f])
    assert m.content == [TextPart("What is this?"), f]
    assert m.files == [f]


def test_message_ids_are_unique():
    a, b = UserMessage("x"), UserMessage("x")
    assert a.id and b.id and a.id != b.id


def test_message_roundtrip():
    msgs: list[Message] = [
        UserMessage(["hi", File("https://example.com/a.pdf")]),
        AssistantMessage(
            [
                ReasoningPart(
                    "think", provider_data={"type": "reasoning", "id": "rs_1"}
                ),
                TextPart(
                    "hello",
                    annotations=[UrlCitation(start=0, end=5, title="t", url="u")],
                ),
                ToolCallPart(tool_call_id="c1", tool_name="f", input={"x": 1}),
            ]
        ),
        ToolMessage(
            [ToolResultPart(tool_call_id="c1", tool_name="f", input={"x": 1}, output=2)]
        ),
    ]
    adapter = TypeAdapter(list[Message])
    assert adapter.validate_json(adapter.dump_json(msgs)) == msgs


def test_file_infers_media_type():
    assert File("photo.JPG").media_type == "image/jpeg"
    assert File("https://example.com/a.pdf?x=1").media_type == "application/pdf"
    assert File("data:audio/wav;base64,AAAA").media_type == "audio/wav"
    assert File(b"abc", media_type="text/plain").media_type == "text/plain"
    with pytest.raises(ValueError):
        File(b"abc")


def test_audio_file_is_base64_encoded(tmp_path):
    raw = b"RIFF\x00\x01binary"
    expected = base64.b64encode(raw).decode()
    path = tmp_path / "a.wav"
    path.write_bytes(raw)
    for f in [
        File(raw, media_type="audio/wav"),
        File(str(path)),
        File(f"data:audio/wav;base64,{expected}"),
    ]:
        [item] = to_responses_input([UserMessage(f)])
        assert item["content"] == [
            {"type": "input_audio", "input_audio": {"data": expected, "format": "wav"}}
        ]


def test_user_message_to_responses_input():
    m = UserMessage(
        [
            "look",
            File("data:image/png;base64,AAAA"),
            File("https://example.com/a.pdf"),
            File(b"%PDF", media_type="application/pdf"),
            File("file-abc", media_type="application/pdf"),
        ]
    )
    assert to_responses_input([m]) == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "auto",
                },
                {"type": "input_file", "file_url": "https://example.com/a.pdf"},
                {
                    "type": "input_file",
                    "file_data": "data:application/pdf;base64,JVBERg==",
                    "filename": "file.pdf",
                },
                {"type": "input_file", "file_id": "file-abc"},
            ],
        }
    ]


def test_assistant_message_to_responses_input():
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "enc",
    }
    m = AssistantMessage(
        [
            ReasoningPart("hmm", provider_data=reasoning_item),
            TextPart("a"),
            ToolCallPart(tool_call_id="c", tool_name="f", input={"x": 1}),
            File("data:image/png;base64,AAAA"),
        ]
    )
    assert to_responses_input([m]) == [
        reasoning_item,
        {"role": "assistant", "content": "a"},
        {"type": "function_call", "call_id": "c", "name": "f", "arguments": '{"x": 1}'},
    ]
    # Reasoning without the provider's item cannot be replayed
    m = AssistantMessage([ReasoningPart("hmm"), "answer"])
    assert to_responses_input([m]) == [{"role": "assistant", "content": "answer"}]
    assert m.reasoning == "hmm"
    assert AssistantMessage("x").reasoning is None


def test_annotations_are_kept():
    c = UrlCitation(start=0, end=1, title="t", url="https://example.com")
    m = AssistantMessage([TextPart("x", annotations=[c])])
    assert m.annotations == [c]


def test_parse_with_string_content():
    assert AssistantMessage('{"result": 3}').parse(int) == 3


def test_tool_output_serialization():
    def result(output):
        return ToolResultPart(tool_call_id="1", tool_name="t", input={}, output=output)

    assert result("plain").serialize_output() == "plain"
    assert result(None).serialize_output() == "null"
    assert result(True).serialize_output() == "true"
    assert result(1.5).serialize_output() == "1.5"
    assert result({"a": [1]}).serialize_output() == '{"a": [1]}'


def test_tool_message_to_responses_input():
    f = File("data:image/png;base64,AAAA")
    plain = ToolResultPart(tool_call_id="1", tool_name="t", input={}, output=None)
    with_file = ToolResultPart(
        tool_call_id="2", tool_name="t", input={}, output="ok", output_files=[f]
    )
    m = ToolMessage([plain, with_file])
    first = to_responses_input([m])
    assert to_responses_input([m]) == first  # no mutation
    assert with_file.output == "ok"
    assert first[0] == {
        "type": "function_call_output",
        "call_id": "1",
        "output": "null",
    }
    assert first[1]["call_id"] == "2"
    text, image = first[1]["output"]
    assert json.loads(text["text"])["output"] == "ok"
    assert image["type"] == "input_image"


def test_tool_audio_is_not_sent():
    f = File(b"RIFF", media_type="audio/wav")
    part = ToolResultPart(
        tool_call_id="1", tool_name="t", input={}, output="ok", output_files=[f]
    )
    [item] = to_responses_input([ToolMessage([part])])
    _, note = item["output"]
    assert note["type"] == "input_text"
    assert f.id in note["text"]


def test_str():
    m = AssistantMessage(
        [
            "Let me check.",
            ToolCallPart(
                tool_call_id="c", tool_name="get_weather", input={"city": "Boston"}
            ),
        ]
    )
    assert str(m) == "Let me check.\n→ get_weather(city='Boston')"
    t = ToolMessage(
        [
            ToolResultPart(
                tool_call_id="c", tool_name="get_weather", input={}, output={"temp": 72}
            )
        ]
    )
    assert str(t) == '← get_weather: {"temp": 72}'


@pytest.mark.asyncio
async def test_prompt_groups_strings_and_files():
    agent = Agent(model="openai/gpt-5-nano")
    f = File("https://example.com/cat.png")
    u = UserMessage("separate")
    msgs = await agent.commands.process_messages(["describe", f, u, "again"])
    assert len(msgs) == 3
    assert msgs[0].content == [TextPart("describe"), f]
    assert msgs[1] == u
    assert msgs[2].content == [TextPart("again")]


@pytest.mark.asyncio
async def test_failed_tool_call_is_marked_as_error():
    def broken():
        """Always fails"""
        raise RuntimeError("boom")

    agent = Agent(model="openai/gpt-5-nano", tools=[broken])
    [r] = await agent.tools.run(
        agent, [ToolCallPart(tool_call_id="c", tool_name="broken", input={})], False
    )
    assert r.is_error
    assert r.output == {"error": "boom"}


def test_history_compaction():
    h = History()
    h.add_instructions("Be nice.")
    msgs = [UserMessage("a"), AssistantMessage("b"), UserMessage("c")]
    h.add(*msgs)
    h.compact("summary of a and b", 2)
    assert h.get() == msgs[2:]
    assert h.get_all() == msgs
    assert h.summary == "summary of a and b"
    instructions = h.get_instructions()
    assert instructions.startswith("Be nice.")
    assert "summary of a and b" in instructions
    h.clear()
    assert h.summary is None and h.get_all() == []


def test_parse_output_items():
    message = {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "hi",
                "annotations": [
                    {
                        "type": "url_citation",
                        "url": "u",
                        "title": "t",
                        "start_index": 1,
                        "end_index": 2,
                    },
                    {"type": "file_citation", "file_id": "f"},
                ],
            },
            {"type": "refusal", "refusal": "no"},
        ],
    }
    assert parse_output_item(message, False) == [
        TextPart("hi", annotations=[UrlCitation(start=1, end=2, title="t", url="u")]),
        TextPart("no"),
    ]

    call = {
        "type": "function_call",
        "call_id": "c",
        "name": "f",
        "arguments": '{"a": 1}',
    }
    assert parse_output_item(call, False) == [
        ToolCallPart(tool_call_id="c", tool_name="f", input={"a": 1})
    ]

    reasoning = {
        "type": "reasoning",
        "id": "rs",
        "summary": [
            {"type": "summary_text", "text": "a"},
            {"type": "summary_text", "text": "b"},
        ],
        "encrypted_content": "enc",
    }
    assert parse_output_item(reasoning, False) == []
    [r] = parse_output_item(reasoning, True)
    assert r == ReasoningPart("a\n\nb", provider_data=reasoning)
