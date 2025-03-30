import base64
import os, dotenv
import asyncio
import streamlit as st

from agentia.message import ContentPartImage, ContentPartText
from agentia import (
    Agent,
    Message,
    MessageStream,
    Event,
    UserMessage,
    AssistantMessage,
    ToolCallEvent,
)
from agentia.utils.config import load_agent_from_config

dotenv.load_dotenv()

st.set_page_config(initial_sidebar_state="collapsed")

# Load agent from config
if "agent" in st.session_state:
    agent: Agent = st.session_state.agent
else:
    agent = st.session_state.agent = load_agent_from_config(os.environ["AGENTIA_AGENT"])

# Render previous messages

messages_container = st.container()


def display_message(message: Message | Event):
    match message:
        case m if (
            isinstance(m, Message) and m.role in ["user", "assistant"] and m.content
        ):
            with messages_container.chat_message(m.role):
                if isinstance(m, AssistantMessage) and m.reasoning:
                    with st.expander("💭 Thinking"):
                        st.markdown(m.reasoning)
                if isinstance(m.content, list):
                    for part in m.content:
                        if isinstance(part, ContentPartImage):
                            st.image(part.url)
                        elif isinstance(part, ContentPartText):
                            st.markdown(part.content)
                else:
                    st.markdown(m.content)
        case m if isinstance(m, ToolCallEvent) and m.result is None:
            with messages_container.chat_message("assistant"):
                st.write(
                    f":blue-badge[:material/smart_toy: **TOOL:**&nbsp;&nbsp;&nbsp;{m.tool.display_name}]"
                )


for message in agent.history.get():
    # print(message)
    display_message(message)


# Chatbox

if prompt := st.chat_input(
    "Enter your message", accept_file=True, file_type=["png", "jpeg", "jpg"]
):
    messages_container.empty()
    if len(prompt.files) > 0:
        content = []
        for file in prompt.files:
            base64_file = base64.b64encode(file.read()).decode("utf-8")
            base64_url = f"data:{file.type};base64,{base64_file}"
            content.append(ContentPartImage(base64_url))
        content.append(ContentPartText(prompt.text))
    else:
        content = prompt.text
    # Display user message
    user_message = UserMessage(content=content)
    display_message(user_message)

    # Call the model and stream the response
    async def write_stream():
        stream = agent.chat_completion(
            messages=[user_message], stream=True, events=True
        )
        async for response in stream:
            if isinstance(response, MessageStream):
                wrapper = messages_container.empty()
                with wrapper:
                    with wrapper.chat_message("assistant"):
                        with st.spinner("", show_time=True):
                            # Stream thinking
                            thinking_wrapper = messages_container.empty()
                            with thinking_wrapper:
                                if response.reasoning:
                                    t = thinking_wrapper.expander(
                                        "💭 Thinking", expanded=True
                                    )
                                    msg = t.empty()
                                    streamed_text = ""
                                    async for s in response.reasoning:
                                        streamed_text += s
                                        msg.markdown(streamed_text)
                                    if streamed_text == "":
                                        thinking_wrapper.empty()
                            # Stream content
                            msg = st.empty()
                            streamed_text = ""
                            async for s in response:
                                streamed_text += s
                                msg.markdown(streamed_text)
                    m = await response.wait_for_completion()
                    if m.tool_calls:
                        wrapper.empty()
            else:
                if isinstance(response, ToolCallEvent) and response.result is None:
                    display_message(response)
                print(response)
        messages_container.empty()

    asyncio.run(write_stream())
