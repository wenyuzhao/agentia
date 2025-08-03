import os
from typing import AsyncIterator, Literal, Any, Sequence, overload, override
from agentia.run import MessageStream, ReasoningMessageStream
from agentia.history import History
from . import LLMBackend, ModelOptions
from ..tools import ToolRegistry
from ..message import (
    AssistantMessage,
    ContentPart,
    ContentPartImage,
    ContentPartText,
    Message,
    ToolCall,
    FunctionCall,
    ToolMessage,
)
from google import genai
from google.genai.types import (
    Content,
    Part,
    FunctionResponse,
    FileData,
    FunctionCall as GenAIFunctionCall,
    ContentUnion,
    GenerateContentResponse,
    GenerateContentConfig,
    FunctionDeclaration,
    Tool,
    GenerateContentResponse,
    ThinkingConfig,
    FunctionResponseScheduling,
)
import logging


class GoogleBackend(LLMBackend):
    def __init__(
        self,
        *,
        name: str = "google",
        model: str,
        tools: ToolRegistry,
        options: ModelOptions,
        history: History,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            name=name, model=model, tools=tools, options=options, history=history
        )
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        self.api_key = api_key
        self.client = genai.Client(
            api_key=api_key, http_options={"api_version": "v1alpha"}
        )
        self.extra_headers: dict[str, str] = {}
        self.extra_body: dict[str, Any] = {}
        self.reasoning_enabled = model.endswith(":think")
        self.has_reasoning = self.reasoning_enabled and options.reasoning_tokens
        # supress 'google_genai.types' warnings
        logging.getLogger("google_genai.types").setLevel(logging.ERROR)

    def get_api_key(self) -> str:
        return self.api_key

    @classmethod
    def get_default_model(cls) -> str:
        return "[google]gemini-2.5-flash-lite"

    @overload
    async def _chat_completion_request(
        self,
        messages: Sequence[Message],
        stream: Literal[False],
        response_format: Any | None,
    ) -> AssistantMessage: ...

    @overload
    async def _chat_completion_request(
        self,
        messages: Sequence[Message],
        stream: Literal[True],
        response_format: Any | None,
    ) -> MessageStream: ...

    @override
    async def _chat_completion_request(
        self,
        messages: Sequence[Message],
        stream: bool,
        response_format: Any | None,
    ) -> AssistantMessage | MessageStream:
        contents: list[ContentUnion] = [
            self.message_to_genai_content(m) for m in messages
        ]
        config: GenerateContentConfig = GenerateContentConfig(
            system_instruction=self.history.instructions,
            tools=[
                Tool(
                    function_declarations=[
                        FunctionDeclaration(**s["function"])
                        for s in self.tools.get_schema()
                    ]
                ),
            ],
            response_mime_type=(
                "text/plain" if not response_format else "application/json"
            ),
            response_json_schema=(
                response_format["json_schema"]["schema"] if response_format else None
            ),
            thinking_config=(
                ThinkingConfig(thinking_budget=0)
                if not self.reasoning_enabled
                else ThinkingConfig(include_thoughts=self.options.reasoning_tokens)
            ),
        )
        model = self.model.removesuffix(":think")
        if stream:
            response = await self.client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            return ChatMessageStream(response, self.has_reasoning)
        else:
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            response.text
            if (
                response.candidates is None
                or response.candidates[0].content is None
                or not response.candidates[0].content.parts
            ):
                print(response)
                raise RuntimeError("response.choices is None")
            return self.__gcr_to_message(response)

    def __user_content_part_to_genai_part(self, cp: ContentPart) -> Part:
        if isinstance(cp, ContentPartText):
            return Part(text=cp.content)
        elif isinstance(cp, ContentPartImage):
            return Part(file_data=FileData(file_uri=cp.url))
        else:
            raise TypeError(f"Unsupported content part type: {type(cp)}")

    def message_to_genai_content(self, m: Message) -> Content:
        content = m.content or ""
        if m.role == "tool":
            assert isinstance(content, str)
            assert m.id is not None
            return Content(
                parts=[
                    Part(
                        function_response=GoogleBackend.tool_message_to_genai_tool_response(
                            m
                        )
                    )
                ],
            )
        if m.role == "user":
            parts = (
                [Part(text=content)]
                if isinstance(content, str)
                else [self.__user_content_part_to_genai_part(c) for c in content]
            )
            return Content(role="user", parts=parts)
        if m.role == "assistant":
            assert isinstance(content, str)
            parts: list[Part] = []
            if len(m.tool_calls) > 0:
                parts.extend(
                    self.__tool_call_to_genai_tool_call(tool_call)
                    for tool_call in m.tool_calls
                )
            parts.append(Part(text=content))
            return Content(role="model", parts=parts)
        raise RuntimeError("Unreachable")

    def __gcr_to_message(self, m: GenerateContentResponse) -> "AssistantMessage":
        content: str = ""
        thought: str | None = None
        tool_calls: list[ToolCall] = []
        assert m.candidates
        c = m.candidates[0].content
        assert c is not None and c.parts
        for part in c.parts:
            if part.text is not None:
                if part.thought:
                    thought = (thought or "") + part.text
                else:
                    content += part.text
            if part.function_call is not None:
                tool_calls.append(
                    GoogleBackend.genai_tool_call_to_tool_call(part.function_call)
                )
        return AssistantMessage(
            content=content,
            reasoning=thought,
            tool_calls=tool_calls,
        )

    def __tool_call_to_genai_tool_call(self, tool_call: ToolCall) -> Part:
        p = Part(
            function_call=GenAIFunctionCall(
                id=tool_call.id or None,
                args=tool_call.function.arguments,
                name=tool_call.function.name,
            )
        )
        return p

    @staticmethod
    def tool_message_to_genai_tool_response(
        m: ToolMessage,
        scheduling: FunctionResponseScheduling | None = None,
    ) -> FunctionResponse:
        return FunctionResponse(
            id=m.id,
            name=m.name,
            response={"output": m.content or ""},
            scheduling=scheduling,
        )

    @staticmethod
    def genai_tool_call_to_tool_call(call: GenAIFunctionCall) -> ToolCall:
        return ToolCall(
            id=call.id or "",
            function=FunctionCall(
                name=call.name or "",
                arguments=call.args or {},
            ),
            type="function",
        )


class ChatMessageStream(MessageStream):
    def __init__(
        self, response: AsyncIterator[GenerateContentResponse], has_reasoning: bool
    ):
        self.__aiter = response.__aiter__()
        self.__message = AssistantMessage(content="")
        self.__tool_calls: list[GenAIFunctionCall] = []
        self.__final_message: AssistantMessage | None = None
        self.__final_reasoning: str | None = None
        if has_reasoning:
            self.reasoning = ReasoningMessageStreamImpl(response, self)
        self.__leftover: GenerateContentResponse | None = None

    @override
    async def _ensure_non_empty(self) -> bool:
        """
        The stream can be empty (e.g. for tool calls).
        Before returning the stream to the user, we will call this method to fetch the first chunk, ahd skip empty ones.
        """
        try:
            if self.reasoning:
                non_empty = await self.reasoning._ensure_non_empty()
                if not non_empty:
                    self.reasoning = None  # No reasoning stream available
                    return self.__leftover is not None
                else:
                    return True
            if not self.__aiter:
                return False
            c = await self.__aiter.__anext__()
            self.__leftover = c
            self.__leftover_in_final_content = True
            return True
        except StopAsyncIteration:
            return False

    def __get_final_tool_calls(self) -> list[ToolCall]:
        return [
            GoogleBackend.genai_tool_call_to_tool_call(t) for t in self.__tool_calls
        ]

    def __process_chunk(self, chunk: GenerateContentResponse) -> str:
        text = chunk.text
        if text:
            if self.__message.content is None:
                self.__message.content = ""
            self.__message.content += text
        if chunk.function_calls:
            self.__tool_calls.extend(chunk.function_calls)
        return text or ""

    async def __anext_impl(self) -> str:
        if self.__final_message is not None:
            raise StopAsyncIteration()
        try:
            assert self.__aiter is not None
            chunk = await self.__aiter.__anext__()
            # print(f"Received chunk: {chunk}")
        except StopAsyncIteration:
            self.__message.tool_calls = self.__get_final_tool_calls()
            self.__final_message = self.__message
            self.__aiter = None
            raise StopAsyncIteration()
        return self.__process_chunk(chunk)

    async def __anext__(self) -> str:
        # Drain reasoning stream first
        if self.reasoning is not None and not self.__final_reasoning:
            async for _ in self.reasoning:
                ...
            assert isinstance(self.reasoning, ReasoningMessageStreamImpl)
            self.__final_reasoning = await self.reasoning
        # if there is a prefetched chunk or a leftover from reasoning stream, return it
        if self.__leftover is not None:
            leftover = self.__leftover
            self.__leftover = None
            leftover_text = self.__process_chunk(leftover)
            return leftover_text
        while True:
            delta = await self.__anext_impl()
            if len(delta) > 0:
                return delta

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    @override
    async def _wait_for_completion(self) -> AssistantMessage:
        if self.reasoning:
            assert isinstance(self.reasoning, ReasoningMessageStreamImpl)
            if self.__final_message is not None:
                self.__final_message.reasoning = self.__final_reasoning
                return self.__final_message
        async for _ in self:
            ...
        assert self.__final_message is not None
        if self.reasoning:
            self.__final_message.reasoning = self.__final_reasoning
        return self.__final_message


class ReasoningMessageStreamImpl(ReasoningMessageStream):
    def __init__(
        self,
        response: AsyncIterator[GenerateContentResponse],
        main_stream: ChatMessageStream,
    ):
        self.__aiter = response.__aiter__()
        self.__message = ""
        self.__final_message: str | None = None
        self.__delta = None
        self.__first_chunk: str | None = None
        self.__main_stream = main_stream

    @override
    async def _ensure_non_empty(self) -> bool:
        assert self.__first_chunk is None
        try:
            self.__first_chunk = await anext(self)
            return True
        except StopAsyncIteration:
            return False

    async def __anext_impl(self) -> str:
        if self.__final_message is not None:
            raise StopAsyncIteration()
        try:
            assert self.__aiter is not None
            chunk = await self.__aiter.__anext__()
            if (
                not chunk.candidates
                or not chunk.candidates[0].content
                or not chunk.candidates[0].content.parts
            ):
                raise RuntimeError("No content in chunk")
            reasoning: str | None = None
            for part in chunk.candidates[0].content.parts:
                if part.thought:
                    if reasoning is None:
                        reasoning = ""
                    reasoning += part.text or ""
                else:
                    assert not reasoning, "Non-reasoning part found in reasoning stream"
            assert reasoning is None or isinstance(reasoning, str)
            self.__message += reasoning or ""
            self.__delta = chunk
            if not reasoning:
                raise StopAsyncIteration()
            return reasoning or ""
        except StopAsyncIteration:
            self.__final_message = self.__message
            self.__main_stream._ChatMessageStream__leftover = self.__delta  # type: ignore
            self.__aiter = None
            raise StopAsyncIteration()

    async def __take_prefetched_chunk(self):
        if self.__first_chunk is not None:
            delta = self.__first_chunk
            self.__first_chunk = None
            return delta
        return None

    async def __anext__(self) -> str:
        # if there is a prefetched chunk, return it
        if c := await self.__take_prefetched_chunk():
            return c
        while True:
            delta = await self.__anext_impl()
            if len(delta) > 0:
                return delta

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    @override
    async def _wait_for_completion(self) -> str:
        async for _ in self:
            ...
        assert self.__final_message is not None
        return self.__final_message
