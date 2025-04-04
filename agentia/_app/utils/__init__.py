import os
from pathlib import Path
from typing import Sequence, TypeVar, Callable
from slugify import slugify
import streamlit as st
import uuid

import tomlkit


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


from . import chat
