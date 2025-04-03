import streamlit as st
import copy
import tomlkit

from agentia import Agent
import agentia.utils.app.utils as utils
from agentia.utils.config import ALL_RECOMMENDED_MODELS, Config

st.set_page_config(initial_sidebar_state="collapsed")

st.write("# Dashboard")


ALL_AGENTS = Agent.get_all_agents()

initial_agent = st.session_state.get("initial_agent") or st.query_params.get("agent")
initial_agent_index = utils.find_index(ALL_AGENTS, lambda x: x.id == initial_agent) or 0

if initial_agent:
    st.session_state["initial_agent"] = initial_agent

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

st.query_params["agent"] = selected_agent.id

agent = Agent.load_from_config(selected_agent.id, persist=False)

assert agent.original_config

config = agent.original_config

if "initial_config" in st.session_state:
    initial_config: Config = st.session_state["initial_config"]
else:
    st.session_state["initial_config"] = config
    initial_config: Config = config

settings_tab, plugins_tab = st.tabs(
    ["🤖&nbsp;&nbsp;&nbsp;**Agent Settings**", "🧩&nbsp;&nbsp;&nbsp;**Plugins**"]
)

new_config = copy.deepcopy(config)

with settings_tab:
    new_config.agent.name = st.text_input("Name", initial_config.agent.name)
    new_config.agent.icon = st.text_input("Icon", initial_config.agent.icon)
    new_config.agent.description = st.text_input(
        "Description", initial_config.agent.description
    )
    if new_config.agent.description and new_config.agent.description.strip() == "":
        new_config.agent.description = ""
    new_config.agent.instructions = st.text_area(
        "Instructions", initial_config.agent.instructions
    )
    if new_config.agent.instructions and new_config.agent.instructions.strip() == "":
        new_config.agent.instructions = ""
    new_config.agent.user = st.text_area("User", initial_config.agent.user)
    if new_config.agent.user and new_config.agent.user.strip() == "":
        new_config.agent.user = ""
    model_index = 0
    all_models = ["(default)", *ALL_RECOMMENDED_MODELS]
    try:
        if initial_config.agent.model:
            model_index = all_models.index(initial_config.agent.model)
    except ValueError:
        pass

    new_config.agent.model = st.selectbox("Model", all_models, index=model_index)
    if new_config.agent.model == "(default)":
        new_config.agent.model = None

    if st.button("Save", type="primary"):
        # Save new config
        assert agent.original_config_path
        doc = tomlkit.parse(agent.original_config_path.read_text())
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

        update_string_field("name", required=True)
        update_string_field("icon")
        update_string_field("description")
        update_string_field("instructions")
        update_string_field("model")
        update_string_field("user")

        with open(agent.original_config_path, "w") as fp:
            tomlkit.dump(doc, fp)
            print("Saved")


with plugins_tab:

    st.write("Plugins")
