import os
from pathlib import Path
import streamlit as st
import copy
import tomlkit
from tomlkit.container import Container

from agentia import Agent
from agentia.plugins import ALL_PLUGINS, Plugin
import agentia.utils.app.utils as utils
from agentia.utils.config import ALL_RECOMMENDED_MODELS, Config
from slugify import slugify

st.set_page_config(initial_sidebar_state="collapsed")


ALL_AGENTS = Agent.get_all_agents()


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


@st.dialog("New Agent")
def new_agent_dialog():
    new_agent()


if "agent" in st.query_params:
    init_agent_id = st.query_params.get("agent", "")
elif len(ALL_AGENTS) > 0:
    init_agent_id = ALL_AGENTS[0].id
else:
    st.divider()
    new_agent()
    st.stop()

# Load initial agent
if "initial_agent" not in st.session_state:
    st.session_state["initial_agent"] = Agent.load_from_config(
        init_agent_id, persist=False
    )
init_agent: Agent = st.session_state["initial_agent"]
init_agent_index = utils.find_index(ALL_AGENTS, lambda x: x.id == init_agent.id) or 0
# Load initial config doc
if "initial_doc" not in st.session_state:
    assert init_agent.config_path
    st.session_state["initial_doc"] = tomlkit.loads(init_agent.config_path.read_text())
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

agent = Agent.load_from_config(selected_agent.id, persist=False)

assert init_agent.config and agent.config_path
assert agent.config and agent.config_path


st.write("")

settings_tab, plugins_tab = st.tabs(
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
    all_models = ["(default)", *ALL_RECOMMENDED_MODELS]
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
        doc = tomlkit.parse(agent.config_path.read_text())
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

            with open(agent.config_path, "w") as fp:
                tomlkit.dump(doc, fp)
                st.rerun()
        except ValueError as e:
            st.error("ERROR: " + str(e))


with settings_tab:
    render_settings_tab()


def render_plugins_tab():
    assert agent.config and agent.config_path
    assert init_agent.config and init_agent.config_path

    doc = tomlkit.parse(agent.config_path.read_text())

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
        with open(agent.config_path, "w") as fp:
            tomlkit.dump(doc, fp)
            st.rerun()

    # Configs

    st.write("")
    st.write("###### Plugin Configs")

    for p in all_plugins:
        P = ALL_PLUGINS[p]
        if P.__options__.__code__ == Plugin.__options__.__code__:
            continue
        with st.expander("**" + (P.NAME or p) + "**"):
            init_configs: Container = init_doc.get("plugins", tomlkit.table()).get(P.id(), {})  # type: ignore
            curr_configs: Container = doc.get("plugins", tomlkit.table()).get(P.id(), {})  # type: ignore
            new_configs = copy.deepcopy(init_configs)
            P.__options__(agent=agent.id, configs=new_configs)
            if new_configs != curr_configs:  # type: ignore
                if st.button("Save", type="primary", key=P.id() + ".save"):
                    if "plugins" not in doc:
                        doc["plugins"] = tomlkit.table()
                    doc["plugins"][P.id()] = new_configs  # type: ignore
                    with open(agent.config_path, "w") as fp:
                        tomlkit.dump(doc, fp)
                        st.rerun()

    # st.write("Plugins")


with plugins_tab:
    render_plugins_tab()
