import streamlit as st
import agentia
import dotenv

agentia.utils._setup_logging()
dotenv.load_dotenv()

pg = st.navigation(
    [
        st.Page("1_chat.py", title="Chat", icon="💬", default=True),
        st.Page("2_dashboard.py", title="Dashboard", icon="⚙️", url_path="dashboard"),
    ]
)
pg.run()
