import base64
import shelve
from typing import Any, Literal
import streamlit as st
from streamlit_oauth import OAuth2Component
import json
import os

from agentia.agent import Agent


class OAuth2Client:
    def __init__(
        self,
        agent: str,
        plugin: str,
        kind: Literal["azure_ad", "google"],
        client: OAuth2Component,
    ):
        self.kind = kind
        self.__client = client
        self.__agent = agent
        self.__plugin = plugin

    def __refresh_token(self, token: Any) -> Any | None:
        try:
            new_token = self.__client.refresh_token(token=token)
            if new_token:
                return json.loads(json.dumps(new_token))
        except Exception as e:
            print(e)
        return None

    def __id_token(self, token: Any):
        id_token = token["id_token"]
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        payload = json.loads(base64.b64decode(payload))
        return payload

    def __user_info(self, token: Any):
        id_token = self.__id_token(token)
        avatar = None
        match self.kind:
            case "google":
                avatar = "https://google.com/favicon.ico"
            case "azure_ad":
                avatar = "https://microsoft.com/favicon.ico"

        email = id_token.get("email") or id_token.get("preferred_username") or ""
        name = id_token.get("name") or ""

        with st.chat_message("user", avatar=avatar):
            st.write(f"&nbsp;&nbsp;&nbsp;**{name}**&nbsp;&nbsp;&nbsp;*{email}*")

    def __login_button(self, scope: str) -> Any | None:
        match self.kind:
            case "google":
                name = "Continue with Google"
                icon = "https://google.com/favicon.ico"
            case "azure_ad":
                name = "Continue with Microsoft"
                icon = "https://microsoft.com/favicon.ico"
            case _:
                raise ValueError(f"Invalid kind: {self.kind}")
        origin = st.context.headers["origin"]
        result = self.__client.authorize_button(
            name=name,
            icon=icon,
            redirect_uri=f"{origin}/component/streamlit_oauth.authorize_button",
            scope=scope,
            # key=self.kind,
            extras_params={"prompt": "consent", "access_type": "offline"},
            use_container_width=True,
            pkce="S256",
        )
        if result is not None:
            if token := result.get("token"):
                return json.loads(json.dumps(token))
        return None

    def login_panel(self, scope: str):
        token = self.__load()
        # Update token
        if token:
            refreshed = self.__refresh_token(token)
            if refreshed != token:
                self.__save(refreshed)
            token = refreshed
        if token:
            self.__user_info(token)
            if st.button("Logout"):
                self.__save(None)
                st.rerun()
        else:
            if new_token := self.__login_button(scope=scope):
                self.__save(new_token)
                st.rerun()

    def __load(self) -> Any | None:
        config_file = Agent.get_config_path(self.__agent)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with shelve.open(config_file) as config:
            key = f"plugins.{self.__plugin}.token"
            token = config.get(key)
            return token

    def __save(self, token: Any):
        config_file = Agent.get_config_path(self.__agent)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with shelve.open(config_file) as config:
            key = f"plugins.{self.__plugin}.token"
            if token is None:
                del config[key]
            else:
                config[key] = token

    @staticmethod
    def azure_ad(
        agent: str,
        plugin: str,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> "OAuth2Client":
        client_id = client_id or os.environ["AUTH_AZURE_AD_CLIENT_ID"]
        client_secret = client_secret or os.environ["AUTH_AZURE_AD_CLIENT_SECRET"]
        client = OAuth2Component(
            client_id=client_id,
            client_secret=client_secret,
            authorize_endpoint="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            token_endpoint="https://login.microsoftonline.com/common/oauth2/v2.0/token",
            refresh_token_endpoint="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        )
        return OAuth2Client(agent=agent, plugin=plugin, kind="azure_ad", client=client)

    @staticmethod
    def google(
        agent: str,
        plugin: str,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> "OAuth2Client":
        client_id = client_id or os.environ["AUTH_GOOGLE_CLIENT_ID"]
        client_secret = client_secret or os.environ["AUTH_GOOGLE_CLIENT_SECRET"]
        client = OAuth2Component(
            client_id=client_id,
            client_secret=client_secret,
            authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
            token_endpoint="https://oauth2.googleapis.com/token",
            refresh_token_endpoint="https://oauth2.googleapis.com/token",
            revoke_token_endpoint="https://oauth2.googleapis.com/revoke",
        )
        return OAuth2Client(agent=agent, plugin=plugin, kind="google", client=client)
