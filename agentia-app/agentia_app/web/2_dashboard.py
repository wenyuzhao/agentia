import streamlit as st
import copy
import tomlkit
from tomlkit.container import Container

from agentia import Agent
from agentia.plugins import ALL_PLUGINS, Plugin
import agentia_app.utils as utils
import agentia.utils.config as cfg
import agentia.utils.session as sess
import logging

utils.page_setup()


ALL_AGENTS = cfg.get_all_agents()


def unload_session_agent():
    if a := st.session_state.get("agent"):
        if a.id == agent:  # type: ignore
            del st.session_state["agent"]


@st.dialog("New Agent")
def new_agent_dialog():
    utils.new_agent()


@st.dialog("Delete Agent")
def delete_agent_dialog(agent: str):
    st.write("Are you sure you want to delete this agent?")
    if st.button("DELETE", type="primary"):
        sess.delete_agent(agent)
        del st.session_state["initial_agent"]
        del st.session_state["initial_doc"]
        unload_session_agent()
        del st.query_params["agent"]
        st.rerun()


if "agent" in st.query_params:
    init_agent_id = st.query_params.get("agent", "")
elif len(ALL_AGENTS) > 0:
    init_agent_id = ALL_AGENTS[0].id
else:
    st.divider()
    utils.new_agent()
    st.stop()

# Load initial agent
if "initial_agent" not in st.session_state:
    st.session_state["initial_agent"] = Agent.load_from_config(
        init_agent_id, persist=False, log_level=logging.WARNING
    )
init_agent: Agent = st.session_state["initial_agent"]
init_agent_index = utils.find_index(ALL_AGENTS, lambda x: x.id == init_agent.id) or 0
# Load initial config doc
if "initial_doc" not in st.session_state:
    assert init_agent.config_path
    st.session_state["initial_doc"] = cfg.load(init_agent.config_path)
init_doc: tomlkit.TOMLDocument = st.session_state["initial_doc"]


col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
with col1:
    st.write("# Dashboard")
with col2:
    if st.button("New Agent", use_container_width=True, type="primary"):
        new_agent_dialog()

selected_agent = st.selectbox(
    "**Select an Agent:**",
    options=ALL_AGENTS,
    format_func=lambda a: (
        f"{a.config.agent.icon} {a.config.agent.name}"
        if a.config.agent.icon
        else a.config.agent.name
    ),
    index=init_agent_index,
)

if selected_agent.id != st.query_params.get("agent"):
    st.query_params["agent"] = selected_agent.id
    del st.session_state["initial_agent"]
    del st.session_state["initial_doc"]
    st.rerun()

agent = Agent.load_from_config(
    selected_agent.id, persist=False, log_level=logging.WARNING
)

assert init_agent.config and agent.config_path
assert agent.config and agent.config_path


st.write("")

(settings_tab, plugins_tab) = st.tabs(
    ["🤖&nbsp;&nbsp;&nbsp;**Agent Settings**", "🧩&nbsp;&nbsp;&nbsp;**Plugins**"]
)


def render_settings_tab():
    assert agent.config and agent.config_path
    assert init_agent.config and init_agent.config_path
    init_config = init_agent.config
    config = agent.config
    new_config = copy.deepcopy(config)

    col1, col2 = st.columns([1, 2])
    with col1:
        new_config.agent.icon = st.text_input("Icon", init_config.agent.icon)
    with col2:
        new_config.agent.name = st.text_input("Name", init_config.agent.name)
    new_config.agent.description = st.text_input(
        "Description", init_config.agent.description
    )
    if new_config.agent.description and new_config.agent.description.strip() == "":
        new_config.agent.description = ""
    new_config.agent.instructions = st.text_area(
        "Instructions", init_config.agent.instructions, height=256
    )
    if new_config.agent.instructions and new_config.agent.instructions.strip() == "":
        new_config.agent.instructions = ""
    new_config.agent.user = st.text_area("User", init_config.agent.user)
    if new_config.agent.user and new_config.agent.user.strip() == "":
        new_config.agent.user = ""
    model_index = 0
    all_models = ["(default)", *cfg.ALL_RECOMMENDED_MODELS]
    try:
        if init_config.agent.model:
            model_index = all_models.index(init_config.agent.model)
    except ValueError:
        pass

    new_config.agent.model = st.selectbox("Model", all_models, index=model_index)
    if new_config.agent.model == "(default)":
        new_config.agent.model = None

    if st.button("Save", type="primary"):
        # Save new config
        assert agent.config_path
        doc = cfg.load(agent.config_path)
        new_dict = new_config.model_dump()
        old_dict = config.model_dump()

        def update_string_field(field: str, required: bool = False):
            if new_dict["agent"][field] != old_dict["agent"][field]:
                value = new_dict["agent"][field]
                assert isinstance(value, str | None)
                if value is not None and value.strip() == "":
                    value = None
                if value is None:
                    if required:
                        raise ValueError(f"`{field}` cannot be empty")
                    del doc["agent"][field]  # type: ignore
                else:
                    doc["agent"][field] = value  # type: ignore

        try:
            update_string_field("name", required=True)
            update_string_field("icon")
            update_string_field("description")
            update_string_field("instructions")
            update_string_field("model")
            update_string_field("user")

            cfg.save(agent.config_path, doc)
            unload_session_agent()
            st.rerun()
        except ValueError as e:
            st.error("ERROR: " + str(e))
    st.divider()
    with st.expander(":red[**DANGER ZONE**]"):
        if st.button("Delete Agent", type="primary"):
            delete_agent_dialog(agent.id)


with settings_tab:
    render_settings_tab()


def render_plugins_tab():
    assert agent.config and agent.config_path
    assert init_agent.config and init_agent.config_path

    doc = cfg.load(agent.config_path)

    all_plugins = [k for k, v in ALL_PLUGINS.items()]
    # Initial enabled plugins
    init_enabled = init_agent.config.get_enabled_plugins()
    # Current enabled plugins
    curr_enabled = agent.config.get_enabled_plugins()

    st.write("###### Enabled Plugins")
    new_enabled = sorted(
        st.multiselect(
            "Enabled Plugins",
            all_plugins,
            default=init_enabled,
            label_visibility="collapsed",
        )
    )

    if new_enabled != curr_enabled:
        doc["agent"]["plugins"] = new_enabled  # type: ignore
        cfg.save(agent.config_path, doc)
        unload_session_agent()
        st.rerun()

    # Configs

    st.write("")
    st.write("###### Plugin Configs")

    for p in all_plugins:
        P = ALL_PLUGINS[p]
        if P.__options__.__code__ == Plugin.__options__.__code__:
            continue
        with st.expander("**" + (P.NAME or p) + "**"):
            init_config: Container = init_doc.get("plugins", tomlkit.table()).get(P.id(), {})  # type: ignore
            curr_config: Container = doc.get("plugins", tomlkit.table()).get(P.id(), {})  # type: ignore
            new_config = copy.deepcopy(init_config)
            P.__options__(agent=agent.id, config=new_config)
            if new_config != curr_config:  # type: ignore
                if st.button("Save", type="primary", key=P.id() + ".save"):
                    doc.setdefault("plugins", tomlkit.table())[P.id()] = new_config  # type: ignore
                    cfg.save(agent.config_path, doc)
                    unload_session_agent()
                    st.rerun()


with plugins_tab:
    render_plugins_tab()
