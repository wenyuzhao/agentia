import os
from . import Plugin, tool
from typing import Annotated
import httpx
from markdownify import markdownify
from tavily import AsyncTavilyClient


class WebPlugin(Plugin):
    def __init__(self, tavily_api_key: str | None = None):
        api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required for WebPlugin")
        self.__tavily = AsyncTavilyClient(api_key=api_key)

    async def __get(self, url: str):
        async with httpx.AsyncClient(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        ) as client:
            res = await client.get(url)
            res.raise_for_status()
            content_type = res.headers.get("content-type")
            if content_type == "text/html":
                md = markdownify(res.text)
                return {"content": md, "content_type": content_type}
            # raw text: .txt, .csv, .json, .md
            if content_type in ["text/plain", "application/json", "text/markdown"]:
                return {"content": res.text, "content_type": content_type}
            # If the content is not supported, return the raw content
            return {"content": res.text, "content_type": content_type}

    @tool
    async def get_webpage_content(
        self,
        url: Annotated[str, "The URL of the web page to get the content of"],
    ):
        """
        Access a web page by a URL, and fetch the content of this web page (in markdown format).
        You can always use this tool to directly access web content or access external sites.
        Use it at any time when you think you may need to access the internet.
        """
        result = await self.__tavily.extract(
            urls=url,
            # extract_depth="advanced",
            include_images=True,
        )
        failed_results = result.get("failed_results", [])
        if len(failed_results) > 0:
            try:
                return await self.__get(url)
            except Exception as e:
                pass
        return result
