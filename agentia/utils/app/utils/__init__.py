import os
from pathlib import Path
import shelve
from typing import Literal, Sequence, TypeVar, Callable
from slugify import slugify
import streamlit as st
import uuid

import tomlkit

from agentia.agent import Agent, SessionInfo


def flex(
    flex_direction: str = "row",
    align_items: str = "center",
    justify_content: str = "flex-start",
    flex_wrap: str = "wrap",
    children_styles: dict[int, str] | list[str] | None = None,
):

    id = "_" + uuid.uuid4().hex

    container_selector = f'div[data-testid="stVerticalBlock"]:has(> div.element-container > div.stHtml > span.{id})'

    children_styles = children_styles or {}
    children_css = ""

    if isinstance(children_styles, dict):
        for i, style in children_styles.items():
            children_css += (
                f"{container_selector} > div:nth-child({i+2}) {{ {style} }}\n"
            )
    elif isinstance(children_styles, list):
        for i, style in enumerate(children_styles):
            children_css += (
                f"{container_selector} > div:nth-child({i+2}) {{ {style} }}\n"
            )
    container = st.container()

    container.html(
        f"""
        <span style="display:none" class="{id}"></span>

        <style>
            {container_selector} {{
                display: flex !important;
                flex-direction: {flex_direction} !important;
                align-items: {align_items} !important;
                flex-wrap: {flex_wrap} !important;
                justify-content: {justify_content} !important;
                column-gap: 0 !important;
            }}

            {container_selector} div:has(> div.stSelectbox)  {{
                width: unset !important;
            }}

            {container_selector} div.stSelectbox  {{
                width: unset !important;
            }}

            {container_selector} div:has(> div.stButton)  {{
                width: unset !important;
            }}

            {container_selector} div.stButton  {{
                width: unset !important;
            }}

            {container_selector} div.stNumberInput  {{
                width: unset !important;
            }}

            {container_selector} div:has(> div.stHtml)  {{
                width: unset !important;
            }}

            {container_selector} div.stHtml  {{
                width: unset !important;
            }}

            {container_selector} div[data-testid="stVerticalBlock"]  {{
                width: unset !important;
            }}

            {children_css}
        </style>
        """
    )

    return container


def __session_title(sid: str, title: str, active: bool = False) -> bool:

    id = "_" + uuid.uuid4().hex

    container_selector = f'div[data-testid="stVerticalBlock"]:has(> div.element-container > div.stHtml > span.{id})'

    container = st.container()

    container.html(
        f"""
        <span style="display:none;" class="{id} session-title"></span>

        <style>
        {container_selector} {{
            max-width: 100% !important;
            width: 100% !important;
        }}
            {container_selector}, {container_selector} div, {container_selector} button, {container_selector} p {{
                column-gap: 0 !important;
                display: flex !important;
                flex-direction: row !important;
                align-items: center !important;
                flex-wrap: nowrap !important;
                flex: 1;
                max-width: inherit;
            }}

            div.element-container:has(> div.stHtml > span.{id}) {{
                width: unset !important;
                flex: unset !important;
            }}

            {container_selector} button > div[data-testid="stMarkdownContainer"] {{
                white-space: nowrap !important;
                text-overflow: ellipsis !important;
                width: 100% !important;
                max-width: 100% !important;
                text-align: left !important;
                overflow: hidden !important;
            }}
        </style>
        """
    )

    with container:
        return st.button(
            title,
            use_container_width=True,
            type="primary" if active else "tertiary",
            key=sid + "-title",
        )


def session_record(
    id: str, title: str, active: bool
) -> Literal["select", "delete"] | None:
    with flex(
        children_styles=["flex: 1; max-width: calc(100% - 32px)", ""],
        flex_wrap="nowrap",
    ):
        title_click = __session_title(id, title, active)
        del_click = st.button(
            "&nbsp;&nbsp;&nbsp;:material/close:", type="tertiary", key=id + "-del"
        )
        if title_click:
            return "select"
        if del_click:
            return "delete"


def get_initial_agent(all_agent_ids: list[str]) -> tuple[Agent, list[SessionInfo]]:
    app_config = Agent.global_cache_dir() / "streamlit-app" / "config"
    app_config.parent.mkdir(parents=True, exist_ok=True)
    with shelve.open(app_config) as db:
        initial_agent: str | None = db.get("last_agent", None)
        if initial_agent and initial_agent not in all_agent_ids:
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
        agent_id = initial_agent or all_agent_ids[0]
        agent = st.session_state.agent = Agent.load_from_config(
            agent_id, True, session_id=initial_session
        )
        sessions = Agent.get_all_sessions(agent.id)

    else:
        agent_id = initial_agent or all_agent_ids[0]
        sessions = Agent.get_all_sessions(agent_id)
        session = sessions[0] if len(sessions) > 0 else None
        agent = st.session_state.agent = Agent.load_from_config(
            agent_id, True, session_id=session.id if session else None
        )

    session_ids = [s.id for s in sessions]

    if agent.session_id not in session_ids:
        sessions.append(agent.get_session_info())
        sessions.sort(reverse=True, key=lambda s: s.id)

    # Save last agent and session
    with shelve.open(app_config) as db:
        db["last_agent"] = agent.id
        db["last_session"] = agent.session_id

    return agent, sessions


T = TypeVar("T")


def find_index(a: Sequence[T], f: Callable[[T], bool]) -> int | None:
    for i, item in enumerate(a):
        if f(item):
            return i
    return None


def new_agent():
    st.write("#### Create a new agent:")
    name = st.text_input("name", label_visibility="collapsed").strip()
    if st.button("Create", type="primary", disabled=name == ""):
        id = slugify(name)
        doc = tomlkit.document()
        table = tomlkit.table()
        table.add("name", name)
        doc.add("agent", table)
        configs_dir = Path.cwd() / "agents"
        if "AGENTIA_NEW_AGENT_DIR" in os.environ:
            configs_dir = Path(os.environ["AGENTIA_NEW_AGENT_DIR"])
        configs_dir.mkdir(parents=True, exist_ok=True)
        with open(configs_dir / f"{id}.toml", "w+") as fp:
            tomlkit.dump(doc, fp)
        st.query_params["agent"] = id
        if "initial_agent" in st.session_state:
            del st.session_state["initial_agent"]
        if "initial_doc" in st.session_state:
            del st.session_state["initial_doc"]
        st.rerun()
