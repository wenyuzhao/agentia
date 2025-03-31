from typing import Literal
import streamlit as st
import uuid


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
