from typing import Any

from agentia import tool, Plugin
from agentia.plugins import register_plugin
import os
from typing import Annotated, Any, override
import json

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import base64
from bs4 import BeautifulSoup
from tomlkit.container import Container


SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
]


@register_plugin("gmail")
class GmailPlugin(Plugin):
    @override
    async def init(self):
        creds = await self.__get_creds()
        self.service: Any = build("gmail", "v1", credentials=creds)

    @override
    async def __get_creds(self):

        with self.agent.open_configs_file() as cache:
            key = self.cache_key(".token")

            if key in cache:
                token = cache[key]
                try:
                    client_id = os.environ["AUTH_GOOGLE_CLIENT_ID"]
                    client_secret = os.environ["AUTH_GOOGLE_CLIENT_SECRET"]
                    token["client_id"] = client_id
                    token["client_secret"] = client_secret
                    creds = Credentials.from_authorized_user_info(token, SCOPES)
                    if creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                        token = json.loads(creds.to_json())
                        token["id_token"] = creds.id_token
                        cache[key] = token
                    return creds
                except Exception as e:
                    self.agent.log.error(f"Error loading token: {e}")
                    ...
            client_id = os.environ.get("AUTH_GOOGLE_NATIVE_CLIENT_ID")
            client_secret = os.environ.get("AUTH_GOOGLE_NATIVE_CLIENT_SECRET")
            if client_id is None or client_secret is None:
                raise RuntimeError(
                    "To configure the gmail plugin in a terminal, please set AUTH_GOOGLE_NATIVE_CLIENT_ID and AUTH_GOOGLE_NATIVE_CLIENT_SECRET"
                )
            flow = Flow.from_client_config(
                {
                    "installed": {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                },
                scopes=SCOPES,
                redirect_uri="urn:ietf:wg:oauth:2.0:oob",
            )
            auth_url, _ = flow.authorization_url(prompt="consent")
            print(f"Please login your Gmail account at here: {auth_url}")
            code = input("Paste the authorization code: ")
            flow.fetch_token(code=code)
            session = flow.authorized_session()
            token = json.loads(session.credentials.to_json())
            token["id_token"] = flow.oauth2session.token["id_token"]
            cache[key] = token
            creds = session.credentials
            return creds

    @classmethod
    @override
    def __options__(cls, agent: str, config: Container):
        from agentia_app.utils.oauth import OAuth2Client

        oauth_client = OAuth2Client.google(agent, cls.id())
        oauth_client.login_panel(
            scope=" ".join(SCOPES),
        )

    def __get_email_by_id(self, email_id: str, body: bool = False) -> dict:
        data = self.service.users().messages().get(userId="me", id=email_id).execute()
        headers = data["payload"]["headers"]
        result = {
            "id": data["id"],
            # "threadId": data["threadId"],
            # "labels": data["labelIds"],
            # "snippet": data["snippet"],
            "from": [x["value"] for x in headers if x["name"].lower() == "from"][0],
            # "to": [x["value"] for x in headers if x["name"] == "To"][0],
            "subject": [x["value"] for x in headers if x["name"].lower() == "subject"][
                0
            ],
            "date": [x["value"] for x in headers if x["name"].lower() == "received"][0]
            .split(";", maxsplit=1)[1]
            .strip(),
        }
        if body:

            def get_body_part(body_obj: Any):
                if "attachmentId" in body_obj:
                    return {"attachmentId": body_obj["attachmentId"]}
                elif "data" in body_obj:
                    content = base64.urlsafe_b64decode(body_obj["data"]).decode("utf-8")
                    if "<html>" in content:
                        try:
                            soup = BeautifulSoup(content, features="html.parser")
                            for script in soup(["script", "style"]):
                                script.extract()  # rip it out
                            content = soup.get_text()
                        except:
                            pass
                    return {"content": content}
                else:
                    return None

            body_list = []
            if "body" in data["payload"]:
                x = get_body_part(data["payload"]["body"])
                if x is not None:
                    body_list.append(x)
            if "parts" in data["payload"]:
                for p in data["payload"]["parts"]:
                    x = get_body_part(p)
                    if x is not None:
                        body_list.append(x)
            result["body_parts"] = body_list
        return result

    async def __fetch_email_list(
        self, labelIds: list[str], q: str | None = None
    ) -> Any:
        results = (
            self.service.users()
            .messages()
            .list(userId="me", labelIds=labelIds, q=q)
            .execute()
        )
        msgids = [m["id"] for m in results.get("messages", [])]
        return {"messages": [self.__get_email_by_id(m) for m in msgids]}

    @tool(display_name="Gmail Get Inbox")
    async def list_inbox_emails(
        self,
    ) -> list:
        """List all emails in the user's inbox."""
        return await self.__fetch_email_list(
            ["INBOX"], q="-category:promotions -category:social"
        )

    @tool(display_name="Gmail - GetInbox (Unread Only)")
    async def list_unread_emails(
        self,
    ) -> list:
        """List all unread emails in the user's inbox."""
        return await self.__fetch_email_list(
            ["INBOX", "UNREAD"], q="-category:promotions -category:social"
        )

    @tool(display_name="Gmail Get Email")
    async def get_email_detail(
        self,
        id: Annotated[str, "The email ID"],
    ) -> Any:
        """Get the details of an email by its ID, including the email body/content."""
        return self.__get_email_by_id(id, body=True)

    @tool(display_name="Gmail - Archive Email")
    async def archive_email(
        self,
        id: Annotated[str, "The email ID"],
    ) -> Any:
        """Archive an email by its ID."""
        data = self.service.users().messages().get(userId="me", id=id).execute()
        labels = data["labelIds"]
        self.service.users().messages().modify(
            userId="me", id=id, body={"removeLabelIds": labels}
        ).execute()
        return {"status": "success"}
