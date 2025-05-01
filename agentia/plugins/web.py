import os

from agentia.plugins.knowledge_base import KnowledgeBasePlugin
from . import Plugin
from ..tools import tool
from typing import Annotated
import requests
from markdownify import markdownify
from tavily import TavilyClient


class WebPlugin(Plugin):
    def __init__(self, tavily_api_key: str | None = None):
        self.__tavily: TavilyClient | None = None

        if api_key := tavily_api_key or os.environ.get("TAVILY_API_KEY"):
            self.__tavily = TavilyClient(api_key=api_key)

    def __embed_file(self, url: str, res: requests.Response):
        kbase = self.agent.get_plugin(KnowledgeBasePlugin)
        if kbase is None:
            return None
        assert kbase.knowledge_base is not None
        if name := kbase.knowledge_base.add_doc_from_url(url):
            return {
                "file_name": name,
                "content_type": res.headers.get("content-type"),
                "hint": f"This file is embeded into the knowledge base for you. Use tools to search its content.",
            }
        return None

    def __get(self, url: str):
        res = requests.get(url)
        res.raise_for_status()
        content_type = res.headers.get("content-type")
        if content_type == "text/html":
            md = markdownify(res.text)
            return {"content": md, "content_type": content_type}
        if r := self.__embed_file(url, res):
            return r
        # If the content is not supported, return the raw content
        return {"content": res.text, "content_type": content_type}

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
