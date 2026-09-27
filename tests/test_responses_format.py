"""Serialization of conversation items into OpenAI Responses API input items."""

import json

import pytest

from agentia.models import (
    Compaction,
    File,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    InputAudio,
    InputFile,
    InputImage,
    InputText,
    Message,
    OutputText,
    ProviderDefinedTool,
    Reasoning,
    ReasoningText,
    Refusal,
    ResponseFormatJson,
    ResponseFormatText,
    SummaryText,
    input_content_from_file,
    to_responses_input,
)


class TestMessage:
    def test_user_text_becomes_input_text(self):
        assert Message.user("hello").to_responses_item() == {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }

    def test_assistant_text_becomes_output_text(self):
        assert Message.assistant("hi there").to_responses_item() == {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi there", "annotations": []}],
        }

    def test_system_and_developer_roles(self):
        assert Message.system("be brief").role == "system"
        assert Message.developer("be brief").role == "developer"

    def test_files_are_routed_by_media_type(self):
        msg = Message.user(
            [
                "what is this?",
                File(data="https://example.com/cat.png", media_type="image/png"),
            ]
        )
        content = msg.to_responses_item()["content"]
        assert content[0] == {"type": "input_text", "text": "what is this?"}
        assert content[1]["type"] == "input_image"
        assert content[1]["image_url"] == "https://example.com/cat.png"

    def test_files_are_rejected_in_assistant_messages(self):
        with pytest.raises(ValueError):
            Message.assistant([File(data="x", media_type="image/png")])

    def test_text_property_concatenates(self):
        msg = Message(
            role="assistant", content=[OutputText(text="a"), OutputText(text="b")]
        )
        assert msg.text == "ab"

    def test_refusal_property(self):
        msg = Message(role="assistant", content=[Refusal(refusal="no")])
        assert msg.refusal == "no"
        assert Message.assistant("ok").refusal is None

    def test_parse_structured_output(self):
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        msg = Message.assistant(json.dumps({"name": "Alice"}))
        assert msg.parse(Person).name == "Alice"


class TestContent:
    def test_image_file_id(self):
        assert (
            InputImage.from_file(File(data="file-abc", media_type="image/png")).file_id
            == "file-abc"
        )

    def test_audio(self):
        c = InputAudio.from_file(File(data="Zm9v", media_type="audio/wav"))
        assert c.input_audio.format == "wav"
        assert c.input_audio.data == "Zm9v"

    def test_audio_rejects_urls(self):
        with pytest.raises(ValueError):
            InputAudio.from_file(
                File(data="https://example.com/a.wav", media_type="audio/wav")
            )

    def test_video_falls_back_to_input_file(self):
        c = input_content_from_file(
            File(
                data="https://example.com/clip.mp4",
                media_type="video/mp4",
                filename="clip.mp4",
            )
        )
        assert isinstance(c, InputFile)
        assert c.file_data == "https://example.com/clip.mp4"
        assert c.filename == "clip.mp4"


class TestFunctionCall:
    def test_serializes_with_call_id_and_arguments(self):
        call = FunctionCall.create(
            call_id="call_1",
            name="get_weather",
            input={"location": "Boston"},
            id="fc_1",
        )
        assert call.to_responses_item() == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": json.dumps({"location": "Boston"}),
            "id": "fc_1",
        }

    def test_input_parses_arguments(self):
        assert FunctionCall(call_id="c", name="n", arguments='{"a": 1}').input == {
            "a": 1
        }

    def test_input_defaults_to_empty(self):
        assert FunctionCall(call_id="c", name="n", arguments="").input == {}

    def test_non_object_arguments_raise(self):
        with pytest.raises(ValueError):
            FunctionCall(call_id="c", name="n", arguments="[1]").input


class TestFunctionCallOutput:
    def test_structured_output_is_json_encoded(self):
        out = FunctionCallOutput.create(
            call_id="call_1", output={"temperature": "72"}, name="get_weather"
        )
        assert out.to_responses_item() == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": json.dumps({"temperature": "72"}),
        }

    def test_name_and_is_error_are_not_sent(self):
        out = FunctionCallOutput.create(
            call_id="c", output="boom", name="t", is_error=True
        )
        assert out.name == "t"
        assert out.is_error is True
        assert "name" not in out.to_responses_item()
        assert "is_error" not in out.to_responses_item()

    def test_files_become_input_content(self):
        out = FunctionCallOutput.create(
            call_id="c",
            output="done",
            files=[File(media_type="image/png", data="https://example.com/a.png")],
        )
        item = out.to_responses_item()
        assert item["output"][0]["type"] == "input_text"
        assert item["output"][1]["type"] == "input_image"
        assert "files_hint" in out.text


class TestReasoning:
    def test_round_trip_payload_is_preserved(self):
        r = Reasoning(
            id="rs_1",
            content=[ReasoningText(text="step by step")],
            summary=[SummaryText(text="short summary")],
            encrypted_content="enc",
            signature="sig",
            format="openai-responses-v1",
        )
        assert r.to_responses_item() == {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "short summary"}],
            "content": [{"type": "reasoning_text", "text": "step by step"}],
            "encrypted_content": "enc",
            "signature": "sig",
            "format": "openai-responses-v1",
        }

    def test_item_without_id_cannot_be_replayed(self):
        r = Reasoning.create("thinking")
        assert r.to_responses_item() is None
        assert to_responses_input([r, Message.user("hi")]) == [
            Message.user("hi").to_responses_item()
        ]

    def test_display_text_falls_back_to_summary(self):
        assert Reasoning.create("raw", summary=["s"]).display_text == "raw"
        assert Reasoning.create(summary=["a", "b"]).display_text == "a\n\nb"
        assert Reasoning.create(summary=["a", "b"]).summary_text == "a\n\nb"


class TestCompaction:
    def test_serializes_as_assistant_message(self):
        item = Compaction(summary="a summary").to_responses_item()
        assert item["role"] == "assistant"
        assert all(c["type"] == "output_text" for c in item["content"])
        assert item["content"][1]["text"] == "a summary"

    def test_flatten_expands_nested_compactions(self):
        inner = Compaction(summary="inner", original_items=[Message.user("a")])
        outer = Compaction(summary="outer", original_items=[inner, Message.user("b")])
        assert [m.text for m in outer.flatten() if isinstance(m, Message)] == ["a", "b"]


class TestItemList:
    def test_order_is_preserved(self):
        items = [
            Message.user("go"),
            Reasoning.create("thinking", id="rs_1"),
            Message.assistant("calling a tool"),
            FunctionCall.create(call_id="c1", name="t"),
            FunctionCallOutput.create(call_id="c1", output="ok"),
        ]
        assert [i["type"] for i in to_responses_input(items)] == [
            "message",
            "reasoning",
            "message",
            "function_call",
            "function_call_output",
        ]


class TestTools:
    def test_function_tool_is_flat(self):
        schema = {"type": "object", "properties": {}, "required": []}
        tool = FunctionTool(name="t", description="d", input_schema=schema)
        assert tool.to_responses_tool() == {
            "type": "function",
            "name": "t",
            "parameters": schema,
            "strict": False,
            "description": "d",
        }

    def test_provider_defined_tool(self):
        tool = ProviderDefinedTool(
            id="web_search", name="web_search", args={"search_context_size": "low"}
        )
        assert tool.to_responses_tool() == {
            "type": "web_search",
            "search_context_size": "low",
        }


class TestResponseFormat:
    def test_text_format(self):
        assert ResponseFormatText().to_responses_text_format() == {"type": "text"}

    def test_json_object_without_schema(self):
        assert ResponseFormatJson().to_responses_text_format() == {
            "type": "json_object"
        }

    def test_json_schema_format(self):
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        fmt = ResponseFormatJson.from_model(Person).to_responses_text_format()
        assert fmt["type"] == "json_schema"
        assert fmt["name"] == "Person"
        assert fmt["strict"] is True
        assert fmt["schema"]["properties"]["name"]["type"] == "string"


class TestItemRoundTrip:
    def test_items_validate_from_json(self):
        from pydantic import TypeAdapter
        from agentia.models import Item

        adapter = TypeAdapter(list[Item])
        items = [
            Message.user("hi"),
            Reasoning.create("x", id="rs_1"),
            FunctionCall.create(call_id="c", name="t"),
            FunctionCallOutput.create(call_id="c", output="ok"),
            Compaction(summary="s"),
        ]
        dumped = adapter.dump_json(items)
        assert [i.type for i in adapter.validate_json(dumped)] == [
            "message",
            "reasoning",
            "function_call",
            "function_call_output",
            "compaction",
        ]
