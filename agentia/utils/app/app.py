import base64
import os, dotenv
import asyncio
import streamlit as st
import shelve

from agentia.agent import CommunicationEvent
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
from agentia.utils.app.utils import session_record

dotenv.load_dotenv()

st.set_page_config(initial_sidebar_state="collapsed")

ALL_AGENTS = Agent.get_all_agents()
ALL_AGENT_IDS = [a.id for a in ALL_AGENTS]

app_config = Agent.global_cache_dir() / "streamlit-app" / "config"
app_config.parent.mkdir(parents=True, exist_ok=True)
with shelve.open(app_config) as db:
    initial_agent: str | None = db.get("last_agent", None)
    if initial_agent and initial_agent not in ALL_AGENT_IDS:
        initial_agent = None
    initial_session: str | None = db.get("last_session", None)
    if not initial_agent:
        initial_session = None
    elif initial_session and not initial_session.startswith(initial_agent + "-"):
        initial_session = None
    if (
        initial_session
        and not (Agent.global_cache_dir() / "sessions" / initial_session).is_dir()
    ):
        initial_session = None


# Load session or create new one
if "agent" in st.session_state:
    agent: Agent = st.session_state.agent
    sessions = Agent.get_all_sessions(agent.id)
elif initial_session:
    agent_id = initial_agent or ALL_AGENT_IDS[0]
    agent = st.session_state.agent = Agent.load_from_config(
        agent_id, True, session_id=initial_session
    )
    sessions = Agent.get_all_sessions(agent.id)

else:
    agent_id = initial_agent or ALL_AGENT_IDS[0]
    sessions = Agent.get_all_sessions(agent_id)
    session = sessions[0] if len(sessions) > 0 else None
    agent = st.session_state.agent = Agent.load_from_config(
        agent_id, True, session_id=session.id if session else None
    )

session_ids = [s.id for s in sessions]

if agent.session_id not in session_ids:
    sessions.append(agent.get_session_info())
    sessions.sort(reverse=True, key=lambda s: s.id)

session_ids = [s.id for s in sessions]

# Save last agent and session
with shelve.open(app_config) as db:
    db["last_agent"] = agent.id
    db["last_session"] = agent.session_id


@st.dialog("Delete All Sessions")
def delete_all(id: str, name: str):
    st.write(f"Delete all sessions for :blue[{name}]?")
    if st.button("DELETE", type="primary"):
        # Delete all sessions
        sessions = Agent.get_all_sessions(id)
        for session in sessions:
            Agent.delete_session(session.id)
        st.rerun()


with st.sidebar:
    # Agent selection
    initial_agent_index = (
        ALL_AGENT_IDS.index(agent.id) if agent.id in ALL_AGENT_IDS else 0
    )
    selected_agent = st.selectbox(
        "**Agent:**",
        options=ALL_AGENTS,
        format_func=lambda a: (
            f"{a.config.agent.icon} {a.config.agent.name}"
            if a.config.agent.icon
            else a.config.agent.name
        ),
        index=initial_agent_index,
    )
    if selected_agent and selected_agent.id != agent.id:
        print(selected_agent)
        # Find first session
        sessions = Agent.get_all_sessions(selected_agent.id)
        session = sessions[0] if len(sessions) > 0 else None
        st.session_state.agent = Agent.load_from_config(
            selected_agent.id, True, session_id=session.id if session else None
        )
        st.session_state.agent.history.reset()
        st.rerun()
    st.write("##### Sessions:")
    # All sessions
    for session in sessions:
        active = session == agent.session_id
        if session.title is None:
            if sinfo := Agent.load_session_info(session.id):
                title = sinfo.title or "New Conversation"
            else:
                title = "New Conversation"
        else:
            title = session.title

        match session_record(session.id, title, active):
            case "select" if not active:
                st.session_state.agent = Agent.load_from_config(
                    agent.id, True, session_id=session.id
                )
                st.rerun()
            case "delete":
                Agent.delete_session(session.id)
                # Switch to the first session
                sessions = Agent.get_all_sessions(agent.id)
                session = sessions[0] if len(sessions) > 0 else None
                st.session_state.agent = Agent.load_from_config(
                    agent.id, True, session_id=session.id if session else None
                )
                st.rerun()
    st.divider()
    # New/Delete all sessions
    if st.button(
        ":material/add: New Session", use_container_width=True, type="primary"
    ):
        # Create new session
        st.session_state.agent = Agent.load_from_config(agent.id, True, session_id=None)
        st.rerun()
    # Delete all sessions
    if st.button(
        ":material/delete_forever: Delete All",
        use_container_width=True,
        type="secondary",
    ):
        delete_all(agent.id, agent.name or "")


# Render previous messages

messages_container = st.container()


def display_message(message: Message | Event, expanded=False):
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
        case m if (
            isinstance(m, ToolCallEvent)
            and m.result is None
            and m.name != "_communiate"
        ):
            with messages_container.chat_message("assistant"):
                title = f":blue-badge[:material/smart_toy: **TOOL:**&nbsp;&nbsp;&nbsp;{m.display_name}]"
                st.write(title)
        case m if isinstance(m, CommunicationEvent):
            with messages_container.chat_message("assistant"):
                c = agent.colleagues[m.child].name
                direction = "->" if m.response is None else "<-"
                comm = f"{agent.name} {direction} {c}"
                title = f":blue[:material/smart_toy: **COMMUNICATE:**&nbsp;&nbsp;&nbsp;{comm}]"
                with st.expander(title, expanded=expanded):
                    st.write(m.message if not m.response else m.response)


history = agent.history.get()
for i, message in enumerate(history):
    last = i == len(history) - 1
    display_message(message, expanded=last)


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
                if isinstance(response, ToolCallEvent | CommunicationEvent):
                    display_message(response, expanded=True)
                print(response)
        messages_container.empty()
        st.rerun()

    asyncio.run(write_stream())
