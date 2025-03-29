import base64
import os, dotenv
from typing import Annotated, Literal
import asyncio
import streamlit as st

from agentia.decorators import tool
from agentia.plugins import ALL_PLUGINS
from agentia.message import ContentPartImage, ContentPartText
from agentia import (
    Agent,
    Message,
    MessageStream,
    ToolCallEvent,
    UserMessage,
    AssistantMessage,
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


def display_message(message: Message):
    if message.role not in ["user", "assistant"]:
        return
    with messages_container.chat_message(message.role):
        if isinstance(message, AssistantMessage) and message.reasoning:
            with st.expander("💭 Thinking"):
                st.markdown(message.reasoning)
        if isinstance(message.content, list):
            for part in message.content:
                if isinstance(part, ContentPartImage):
                    st.image(part.url)
                elif isinstance(part, ContentPartText):
                    st.markdown(part.content)
        else:
            st.markdown(message.content)


for message in agent.history.get_messages():
    display_message(message)

# Chatbox

if prompt := st.chat_input(
    "Enter your message", accept_file=True, file_type=["png", "jpeg", "jpg"]
):
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
        stream = agent.chat_completion(messages=[user_message], stream=True)
        async for response in stream:
            if isinstance(response, MessageStream):
                wrapper = messages_container.empty()
                with wrapper:
                    with wrapper.chat_message("assistant"):
                        # Stream thinking
                        if response.reasoning:
                            t = st.expander("💭 Thinking", expanded=True)
                            msg = t.empty()
                            streamed_text = ""
                            async for s in response.reasoning:
                                streamed_text += s
                                msg.markdown(streamed_text)
                        # Stream content
                        msg = st.empty()
                        streamed_text = ""
                        async for s in response:
                            streamed_text += s
                            msg.markdown(streamed_text)
                        msg = await response.wait_for_completion()
        st.empty()

    asyncio.run(write_stream())
