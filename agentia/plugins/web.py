from io import BytesIO
import os
from ..decorators import *
from . import Plugin
from typing import Annotated, override
import requests
from markdownify import markdownify
import uuid
from tavily import TavilyClient


class WebPlugin(Plugin):

    @override
    async def init(self):
        self.__tavily: TavilyClient | None = None

        if api_key := self.config.get("tavily_api_key") or os.environ.get(
            "TAVILY_API_KEY"
        ):
            self.__tavily = TavilyClient(api_key=api_key)

    def __embed_file(self, content: bytes, file_ext: str):
        assert self.agent.knowledge_base is not None
        with BytesIO(content) as f:
            ext = file_ext if file_ext.startswith(".") else "." + file_ext
            f.name = str(uuid.uuid4()) + ext
            # self.agent.knowledge_base.add_document(f)
            # file_name = f.name
            raise NotImplementedError
        return {
            "file_name": file_name,
            "hint": f"This is a .{file_ext} file and it is embeded in the knowledge base. Use _file_search to query the content.",
        }

    def __get(self, url: str):
        res = requests.get(url)
        res.raise_for_status()
        content_type = res.headers.get("content-type")
        # if content_type == "application/pdf":
        #     # Add this file to the knowledge base
        #     if self.agent.knowledge_base is not None:
        #         return self.__embed_file(res.content, "pdf")
        #     return {"content": "This is a PDF file. You don't know how to view it."}
        md = markdownify(res.text)
        return {"content": md}

    @tool
    def get_webpage_content(
        self,
        url: Annotated[str, "The URL of the web page to get the content of"],
    ):
        """
        Access a web page by a URL, and fetch the content of this web page (in markdown format).
        You can always use this tool to directly access web content or access external sites.
        Use it at any time when you think you may need to access the internet.
        """
        if self.__tavily:
            result = self.__tavily.extract(
                urls=url,
                # extract_depth="advanced",
                include_images=True,
            )
            failed_results = result.get("failed_results", [])
            if len(failed_results) > 0:
                try:
                    return self.__get(url)
                except Exception as e:
                    pass
            return result
        return self.__get(url)
