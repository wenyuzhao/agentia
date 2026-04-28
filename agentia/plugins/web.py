import os
from . import Plugin, tool
from typing import Annotated, Literal, override
import httpx
from markdownify import markdownify
from tavily import AsyncTavilyClient
from ..tools import ToolResult
from ..models import File
from dataclasses import dataclass


@dataclass
class HttpGetTextResponse:
    content: str
    content_type: str


class Web(Plugin):
    def __init__(self, tavily_api_key: str | None = None):
        self.api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        self.__tavily = AsyncTavilyClient(api_key=self.api_key)

    @override
    def get_instructions(self) -> str | None:
        if not self.api_key:
            return "This agent has the WebSearch tool, but no Tavily API key is set. The tool is not usable. Please ask the user to set the TAVILY_API_KEY environment variable."
        return None

    @staticmethod
    async def http_get(url: str) -> HttpGetTextResponse | ToolResult:
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
                return HttpGetTextResponse(content=md, content_type=content_type)
            # raw text: .txt, .csv, .json, .md
            if str(content_type).startswith("text/") or content_type in [
                "application/json"
            ]:
                return HttpGetTextResponse(content=res.text, content_type=content_type)
            # images or videos
            if str(content_type).startswith(("image/", "video/", "application/pdf")):
                return ToolResult(
                    output=f"Loaded file {content_type} from {url}",
                    files=[File(data=url, media_type=content_type)],
                )
            # Raise an error for unsupported content types
            raise ValueError(f"Unsupported content type: {content_type}")

    @tool(name="WebSearch")
    async def web_search(
        self,
        query: Annotated[
            str, "The search query. Please be as specific and verbose as possible."
        ],
        topic: Annotated[
            Literal["general", "news", "finance"] | None,
            "The topic of the search. Default is 'general'.",
        ] = None,
        time_range: Annotated[
            Literal["day", "week", "month", "year"] | None,
            "The time range back from the current date based on publish date or last updated date.",
        ] = None,
        max_results: Annotated[
            int | None,
            "The maximum number of search results to return. Default is 5. It must be between 0 and 20.",
        ] = None,
    ):
        """
        Perform web search on the given query.
        Returning the top related search results in json format.
        When necessary, you need to combine this tool with the get_webpage_content tools (if available), to browse the web in depth by jumping through links.
        """

        if not self.api_key:
            raise ValueError(
                "Tavily API key is required for web search. Ask the user to set the TAVILY_API_KEY environment variable."
            )
        tavily_results = await self.__tavily.search(
            query=query,
            search_depth="advanced",
            # max_results=10,
            include_answer=True,
            include_images=True,
            include_image_descriptions=True,
            topic=topic,  # type: ignore
            time_range=time_range,  # type: ignore
            max_results=max_results,  # type: ignore
        )
        return tavily_results

    @tool(name="WebFetch")
    async def get_webpage_content(
        self,
        url: Annotated[str, "The URL of the web page to get the content of"],
    ):
        """
        Access a web page by a URL, and fetch the content of this web page (in markdown format).
        You can always use this tool to directly access web content or access external sites.
        Use it at any time when you think you may need to access the internet.

        You can also use this tool to fetch images or videos by their URLs. Images or videos are determined by their content type or file extension.
        """
        try:
            r = await Web.http_get(url)
            if not self.api_key or isinstance(r, ToolResult):
                return r
        except Exception as e:
            if not self.api_key:
                raise e
            pass

        result = await self.__tavily.extract(urls=url, include_images=True)
        failed_results = result.get("failed_results", [])
        if len(failed_results) > 0:
            try:
                return await Web.http_get(url)
            except Exception:
                pass
        return result
