from datetime import datetime
import os

from tomlkit.container import Container
from ..decorators import *
from . import Plugin
from typing import Annotated, Any, Literal, override, TYPE_CHECKING
from dataclasses import asdict

if TYPE_CHECKING:
    from pymstodo import TaskList, Task


class MSToDoPlugin(Plugin):
    def __test_token(self, client_id: str, client_secret: str, token: Any):
        import pymstodo

        try:
            client = pymstodo.ToDoConnection(
                client_id=client_id, client_secret=client_secret, token=token
            )
            _lists = client.get_lists()
            return True
        except Exception as e:
            return False

    @override
    async def init(self):
        import pymstodo

        self.agent.log.info("MSToDoPlugin initialized")
        client_id = os.environ["AUTH_AZURE_AD_CLIENT_ID"]
        client_secret = os.environ["AUTH_AZURE_AD_CLIENT_SECRET"]

        with self.agent.open_configs_file() as cache:
            token = None
            key = self.cache_key(".token")
            if key in cache:
                if self.__test_token(client_id, client_secret, cache[key]):
                    token = cache[key]
            if token is None:
                if self.is_server():
                    raise RuntimeError("Please setup the plugin on dashboard")
                auth_url = pymstodo.ToDoConnection.get_auth_url(client_id)
                redirect_resp = input(
                    f"Go here and authorize:\n{auth_url}\n\nPaste the full redirect URL below:\n"
                )
                token = pymstodo.ToDoConnection.get_token(
                    client_id, client_secret, redirect_resp
                )
                cache[key] = token
        self.client = pymstodo.ToDoConnection(
            client_id=client_id, client_secret=client_secret, token=token
        )
        self.default_list_id = self.client.get_lists()[0].list_id

    @classmethod
    @override
    def __options__(cls, agent: str, config: Container):
        from agentia._app.utils.oauth import OAuth2Client

        oauth_client = OAuth2Client.azure_ad(agent, cls.id())
        oauth_client.login_panel(
            scope="user.read Tasks.Read Tasks.ReadWrite openid profile offline_access"
        )

    def __process_time(self, time: str) -> datetime:
        # time is in the format of YYYY-MM-DD HH:MM:SS
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

    def __fmt_task(self, task: "Task") -> Any:
        return asdict(task)

    @tool
    def get_all_task_lists(self):
        """Get the name and list_id of all the task lists. NOTE: This tool does not give you the tasks details in eahc list. Only the list name and id."""
        lists = self.client.get_lists()

        def list_to_json(tl: "TaskList"):
            return {
                "list_id": tl.list_id,
                "name": tl.displayName,
            }

        return [list_to_json(x) for x in lists]

    @tool
    def get_tasks_in_list(
        self,
        list_id: Annotated[
            str | None,
            "The id of the list to get tasks from. By default the default main task list will be used. NOTE: You can use get_all_task_lists to get the list_id of all the task lists.",
        ],
    ):
        """Get all the tasks in a specific list in Microsoft To Do."""
        if list_id is None:
            list_id = self.default_list_id
        tasks = self.client.get_tasks(list_id)
        return [self.__fmt_task(x) for x in tasks]

    @tool
    def create_task(
        self,
        list_id: Annotated[
            str | None,
            "The id of the list to add the task to. By default the default main task list will be used. NOTE: You can use get_all_task_lists to get the list_id of all the task lists.",
        ],
        title: Annotated[str, "The title of the task to create."],
        importance: Annotated[
            Literal["low", "normal", "high"] | None,
            "The importance of the task. normal by default",
        ],
        due_datetime: Annotated[
            str | None,
            "The date that the task is to be finished. In the format of `YYYY-MM-DD HH:MM:SS`. If you are unsure about the due time, set it to 23:59:59. Leave it empty if you don't want to set a due date.",
        ],
        reminder_datetime: Annotated[
            str | None,
            "The date that the reminder is to be sent to the user. In the format of `YYYY-MM-DD HH:MM:SS`. If you are unsure about the reminder time, set it to 23:59:59. Leave it empty if you don't want to set a reminder.",
        ],
    ):
        """Add a task to a specific list in Microsoft To Do."""
        list_id = list_id or self.default_list_id
        task = self.client.create_task(
            list_id=list_id,
            title=title,
            due_date=self.__process_time(due_datetime) if due_datetime else None,
        )
        task_data: Any = {"importance": importance or "normal"}
        if reminder_datetime is not None:
            d = self.__process_time(reminder_datetime)
            task_data["reminderDateTime"] = {
                "dateTime": d.strftime("%Y-%m-%dT%H:%M:%S.0000000"),
                "timeZone": "UTC",
            }
        self.client.update_task(task.task_id, list_id, **task_data)
        return {
            "task_id": task.task_id,
            "list_id": list_id,
        }

    @tool
    def update_task(
        self,
        task_id: Annotated[str, "The id of the task to update."],
        list_id: Annotated[str, "The id of the list that the task is in."],
        title: Annotated[str, "The title of the task to create."],
        importance: Annotated[
            Literal["low", "normal", "high"] | None,
            "The importance of the task. normal by default",
        ],
        due_datetime: Annotated[
            str | None,
            "The date that the task is to be finished. In the format of `YYYY-MM-DD HH:MM:SS`. If you are unsure about the due time, set it to 23:59:59. Leave it empty if you don't want to set a due date.",
        ],
        reminder_datetime: Annotated[
            str | None,
            "The date that the reminder is to be sent to the user. In the format of `YYYY-MM-DD HH:MM:SS`. If you are unsure about the reminder time, set it to 23:59:59. Leave it empty if you don't want to set a reminder.",
        ],
        status: Annotated[
            Literal[
                "notStarted", "inProgress", "completed", "waitingOnOthers", "deferred"
            ]
            | None,
            "The status of the task.",
        ],
    ):
        """Update the information a TODO task in Microsoft To Do. NOTE: You can use get_all_task_lists to get the list_id of all the task lists."""
        task_data: Any = {}
        if title:
            task_data["title"] = title
        if importance:
            task_data["importance"] = importance
        if due_datetime:
            d = self.__process_time(due_datetime)
            task_data["dueDateTime"] = {
                "dateTime": d.strftime("%Y-%m-%dT%H:%M:%S.0000000"),
                "timeZone": "UTC",
            }
        if reminder_datetime:
            d = self.__process_time(reminder_datetime)
            task_data["reminderDateTime"] = {
                "dateTime": d.strftime("%Y-%m-%dT%H:%M:%S.0000000"),
                "timeZone": "UTC",
            }
        if status:
            task_data["status"] = status

        self.client.update_task(task_id, list_id, **task_data)

        return {
            "success": True,
            "task_id": task_id,
            "list_id": list_id,
        }

    @tool
    def add_task_comment(
        self,
        task_id: Annotated[str, "The id of the task to add the comment to."],
        list_id: Annotated[str, "The id of the list that the task is in."],
        comment: Annotated[str, "The comment to add to the task."],
    ):
        """Append a comment to a task in Microsoft To Do."""
        time = datetime.now()
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        name = ""
        if self.agent.name:
            name = self.agent.name
        elif not self.agent.id_is_random:
            name = self.agent.id
        if self.agent.icon:
            name = f"{self.agent.icon} {name}".strip()
        comment = f"[{name} - {time_str}]\n{comment}"
        task = self.client.get_task(task_id=task_id, list_id=list_id)
        prev_comment = (task.body_text or "").strip()
        if prev_comment != "":
            comment = f"{prev_comment}\n\n{comment}"

        task_data: Any = {
            "body": {"content": comment, "contentType": "text"},
        }
        self.client.update_task(task_id, list_id, **task_data)
        return {
            "success": True,
            "task_id": task_id,
            "list_id": list_id,
        }
