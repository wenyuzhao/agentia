import asyncio
from contextlib import AsyncExitStack
from typing import AsyncGenerator
from typing_extensions import Literal
from agentia.agent import Agent
from agentia.llm import LLMBackend
from agentia.llm.google import GoogleBackend
from google.genai.types import (
    Modality,
    ActivityEnd,
    FunctionDeclaration,
    Tool,
    LiveConnectConfig,
    Behavior,
    FunctionResponseScheduling,
)

from agentia.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class RealtimeSession:
    def __init__(self, agent: Agent, backend: LLMBackend):
        self.agent = agent
        assert isinstance(
            backend, GoogleBackend
        ), "Only Gemini is supported for realtime sessions."
        assert not backend.model.endswith(":think"), "Reasoning is not allowed"
        self.llm: GoogleBackend = backend
        self.exit_stack = AsyncExitStack()
        self._tool_scheduling: Literal["blocking", "idle", "silent", "interrupt"] = (
            "idle"
        )

    async def __aenter__(self):
        await self.agent.init()
        self.session = await self.exit_stack.enter_async_context(
            self.llm.client.aio.live.connect(
                model=self.llm.model,
                config=LiveConnectConfig(
                    response_modalities=[Modality.TEXT],
                    system_instruction=self.llm.history.instructions,
                    tools=[
                        Tool(
                            function_declarations=[
                                FunctionDeclaration(
                                    behavior=(
                                        Behavior.BLOCKING
                                        if self._tool_scheduling == "blocking"
                                        else Behavior.NON_BLOCKING
                                    ),
                                    **s["function"],
                                )
                                for s in self.llm.tools.get_schema()
                            ]
                        ),
                    ],
                    # enable_affective_dialog=True,
                ),
            )
        )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.exit_stack.aclose()

    async def send(self, input: str, end: bool | None = None):
        """
        Send a message to the session.
        This is used to send commands or messages to the agent in real-time.
        """
        await self.session.send_realtime_input(
            text=input, activity_end=ActivityEnd() if end else None
        )

    async def __process_tool_calls(self, tool_calls: list[ToolCall]):
        if tool_calls:
            responses: list[ToolMessage] = []
            async for event in self.llm.tools.call_tools(tool_calls):
                if isinstance(event, ToolMessage):
                    responses.append(event)
                    self.llm.history.add(event)
                else:
                    # assert is_event(event), "Event must be a Event object"
                    # if events:
                    #     self.history.add(event)
                    #     yield event
                    ...
            print(responses)
            sched: FunctionResponseScheduling | None = None
            match self._tool_scheduling:
                case "blocking":
                    sched = None
                case "idle":
                    sched = FunctionResponseScheduling.WHEN_IDLE
                case "silent":
                    sched = FunctionResponseScheduling.SILENT
                case "interrupt":
                    sched = FunctionResponseScheduling.INTERRUPT
            await self.session.send_tool_response(
                function_responses=[
                    GoogleBackend.tool_message_to_genai_tool_response(
                        m, scheduling=sched
                    )
                    for m in responses
                ]
            )
            return responses
        return []

    async def receive(self) -> AsyncGenerator[str, None]:
        """
        Receive messages from the session.
        This is a generator that yields messages as they are received.
        """
        while True:
            input_transcript = ""
            output_msg = AssistantMessage(content="")
            async for m in self.session.receive():
                if m.server_content:
                    # Update input transcript and add to history
                    if m.server_content.input_transcription:
                        if m.server_content.input_transcription.text:
                            input_transcript += (
                                m.server_content.input_transcription.text
                            )
                        if m.server_content.input_transcription.finished:
                            if input_transcript:
                                self.llm.history.messages.append(
                                    UserMessage(content=input_transcript)
                                )
                            input_transcript = ""
                    # Update output transcript
                    if m.server_content.output_transcription:
                        if m.server_content.output_transcription.text:
                            output_msg.content += (
                                m.server_content.output_transcription.text
                            )
                if m.tool_call and m.tool_call.function_calls:
                    print(m.tool_call.function_calls)
                    tool_calls = [
                        GoogleBackend.genai_tool_call_to_tool_call(t)
                        for t in m.tool_call.function_calls
                    ]
                    output_msg.tool_calls = tool_calls
                    self.llm.history.add(output_msg)
                    output_msg = AssistantMessage(content="")
                    asyncio.create_task(
                        self.__process_tool_calls(tool_calls=tool_calls)
                    )

                if m.text:
                    output_msg.content += m.text
                    yield m.text
            if output_msg.content or output_msg.tool_calls:
                self.llm.history.add(output_msg)
            # A turn is finished
            await asyncio.sleep(0.1)
