from ..decorators import *
from . import Plugin
from typing import TYPE_CHECKING, Annotated, Union, override
from dataforseo_client import (
    configuration as dfs_config,
    api_client as dfs_api_provider,
)
from dataforseo_client.models.serp_google_organic_live_regular_request_info import (
    SerpGoogleOrganicLiveRegularRequestInfo,
)
from dataforseo_client.models.serp_google_maps_live_advanced_request_info import (
    SerpGoogleMapsLiveAdvancedRequestInfo,
)

if TYPE_CHECKING:
    from dataforseo_client.models.serp_google_organic_live_regular_response_info import (
        SerpGoogleOrganicLiveRegularResponseInfo,
    )
    from dataforseo_client.models.serp_google_news_live_advanced_response_info import (
        SerpGoogleNewsLiveAdvancedResponseInfo,
    )
    from dataforseo_client.models.serp_google_maps_live_advanced_response_info import (
        SerpGoogleMapsLiveAdvancedResponseInfo,
    )
from dataforseo_client.models.serp_google_news_live_advanced_request_info import (
    SerpGoogleNewsLiveAdvancedRequestInfo,
)
from dataforseo_client.models.work_hours import WorkHours
from dataforseo_client.models.work_day_info import WorkDayInfo
import os

from tomlkit.container import Container
from tavily import TavilyClient


class SearchPlugin(Plugin):
    @override
    async def init(self):
        username = os.environ["AUTH_DATAFORSEO_USERNAME"]
        password = os.environ["AUTH_DATAFORSEO_PASSWORD"]
        self.__country = self.config.get("country", "Australia")
        if self.__country not in _ALL_COUNTRIES:
            raise ValueError(f"Invalid country: {self.__country}")

        self.__client = dfs_api_provider.ApiClient(
            dfs_config.Configuration(username=username, password=password)
        )
        from dataforseo_client.api.serp_api import SerpApi

        self.__api = SerpApi(self.__client)

        self.__tavily: TavilyClient | None = None

        if api_key := os.environ.get("TAVILY_API_KEY"):
            self.__tavily = TavilyClient(api_key=api_key)

    @classmethod
    @override
    def __options__(cls, agent: str, config: Container):
        import streamlit as st

        v = config.get("country", "Australia")
        index = _ALL_COUNTRIES.index(v) if v in _ALL_COUNTRIES else None

        config["country"] = st.selectbox(
            "Select the country for the search",
            options=_ALL_COUNTRIES,
            index=index,
        )

        if "Australia" == config["country"]:
            del config["country"]

    def __process_result(
        self,
        res: Union[
            "SerpGoogleOrganicLiveRegularResponseInfo",
            "SerpGoogleNewsLiveAdvancedResponseInfo",
            "SerpGoogleMapsLiveAdvancedResponseInfo",
        ],
    ):
        if (
            not res.tasks
            or len(res.tasks) == 0
            or (res.tasks_error is not None and res.tasks_error > 0)
        ):
            return {"error": "Failed to get search result"}
        return [r.to_dict() for r in res.tasks[0].result or []]

    async def __get_keywords(self, query: str) -> str:
        agent = self.agent.anonymized(
            instructions="Extract keywords from the given query. Just return the keywords without any additional text.",
        )
        keywords = await agent.chat_completion(query)
        return keywords

    @tool
    async def web_search(
        self,
        query: Annotated[
            str, "The search query. Please be as specific and verbose as possible."
        ],
    ):
        """
        Perform web search on the given query.
        Returning the top related search results in json format.
        When necessary, you need to combine this tool with the get_webpage_content tools (if available), to browse the web in depth by jumping through links.
        """

        if self.__tavily:
            tavily_results = self.__tavily.search(
                query=query,
                search_depth="advanced",
                # max_results=10,
                include_answer=True,
                include_images=True,
                include_image_descriptions=True,
            )
            return tavily_results
        else:
            keywords = await self.__get_keywords(query)
            dfs_results = self.__api.google_organic_live_regular(
                [
                    SerpGoogleOrganicLiveRegularRequestInfo(
                        language_code="en",
                        location_name=self.__country,
                        keyword=keywords,
                        depth=10,
                    )
                ]
            )
            dfs_results = self.__process_result(dfs_results)
            return dfs_results

    @tool
    async def news_search(
        self,
        query: Annotated[
            str, "The search query. Please be as specific and verbose as possible."
        ],
    ):
        """
        Perform news search on the given query.
        Returning the top related results in json format.
        """

        if self.__tavily:
            tavily_results = self.__tavily.search(
                query=query,
                search_depth="advanced",
                topic="news",
                # max_results=10,
                include_answer=True,
                include_images=True,
                include_image_descriptions=True,
            )
            return tavily_results
        else:
            keywords = await self.__get_keywords(query)
            response = self.__api.google_news_live_advanced(
                [
                    SerpGoogleNewsLiveAdvancedRequestInfo(
                        language_code="en",
                        location_name=self.__country,
                        keyword=keywords,
                        depth=10,
                    )
                ]
            )
            return self.__process_result(response)

    if "TAVILY_API_KEY" in os.environ:

        @tool
        async def finance_search(
            self,
            query: Annotated[
                str, "The search query. Please be as specific and verbose as possible."
            ],
        ):
            """
            Search for finance-related news and information on the given query.
            Returning the top related results in json format.
            """
            assert self.__tavily is not None
            tavily_results = self.__tavily.search(
                query=query,
                search_depth="advanced",
                topic="finance",
                # max_results=10,
                include_answer=True,
                include_images=True,
                include_image_descriptions=True,
            )
            return tavily_results

    @tool
    async def google_map_search(
        self,
        query: Annotated[
            str, "The search query. Please be as specific and verbose as possible."
        ],
    ):
        """
        Perform Google Map Search on the given query.
        Returning the top 10 search result in json format.
        This is helpful to get the address, website, opening hours, and contact information of a place or store.
        """
        keywords = await self.__get_keywords(query)
        response = self.__api.google_maps_live_advanced(
            [
                SerpGoogleMapsLiveAdvancedRequestInfo(
                    language_code="en",
                    location_name=self.__country,
                    keyword=keywords,
                    depth=10,
                )
            ]
        )
        return self.__process_result(response)


_ALL_COUNTRIES = [
    # Default country
    "Australia",
    # Other countries
    "Afghanistan",
    "Albania",
    "Antarctica",
    "Algeria",
    "American Samoa",
    "Andorra",
    "Angola",
    "Antigua and Barbuda",
    "Azerbaijan",
    "Argentina",
    "Austria",
    "The Bahamas",
    "Bahrain",
    "Bangladesh",
    "Armenia",
    "Barbados",
    "Belgium",
    "Bhutan",
    "Bolivia",
    "Bosnia and Herzegovina",
    "Botswana",
    "Brazil",
    "Belize",
    "Solomon Islands",
    "Brunei",
    "Bulgaria",
    "Myanmar (Burma)",
    "Burundi",
    "Cambodia",
    "Cameroon",
    "Canada",
    "Cabo Verde",
    "Central African Republic",
    "Sri Lanka",
    "Chad",
    "Chile",
    "China",
    "Christmas Island",
    "Cocos (Keeling) Islands",
    "Colombia",
    "Comoros",
    "Republic of the Congo",
    "Democratic Republic of the Congo",
    "Cook Islands",
    "Costa Rica",
    "Croatia",
    "Cyprus",
    "Czechia",
    "Benin",
    "Denmark",
    "Dominica",
    "Dominican Republic",
    "Ecuador",
    "El Salvador",
    "Equatorial Guinea",
    "Ethiopia",
    "Eritrea",
    "Estonia",
    "South Georgia and the South Sandwich Islands",
    "Fiji",
    "Finland",
    "France",
    "French Polynesia",
    "French Southern and Antarctic Lands",
    "Djibouti",
    "Gabon",
    "Georgia",
    "The Gambia",
    "Germany",
    "Ghana",
    "Kiribati",
    "Greece",
    "Grenada",
    "Guam",
    "Guatemala",
    "Guinea",
    "Guyana",
    "Haiti",
    "Heard Island and McDonald Islands",
    "Vatican City",
    "Honduras",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Cote d'Ivoire",
    "Jamaica",
    "Japan",
    "Kazakhstan",
    "Jordan",
    "Kenya",
    "South Korea",
    "Kuwait",
    "Kyrgyzstan",
    "Laos",
    "Lebanon",
    "Lesotho",
    "Latvia",
    "Liberia",
    "Libya",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Madagascar",
    "Malawi",
    "Malaysia",
    "Maldives",
    "Mali",
    "Malta",
    "Mauritania",
    "Mauritius",
    "Mexico",
    "Monaco",
    "Mongolia",
    "Moldova",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Oman",
    "Namibia",
    "Nauru",
    "Nepal",
    "Netherlands",
    "Curacao",
    "Sint Maarten",
    "Caribbean Netherlands",
    "New Caledonia",
    "Vanuatu",
    "New Zealand",
    "Nicaragua",
    "Niger",
    "Nigeria",
    "Niue",
    "Norfolk Island",
    "Norway",
    "Northern Mariana Islands",
    "United States Minor Outlying Islands",
    "Micronesia",
    "Marshall Islands",
    "Palau",
    "Pakistan",
    "Panama",
    "Papua New Guinea",
    "Paraguay",
    "Peru",
    "Philippines",
    "Pitcairn Islands",
    "Poland",
    "Portugal",
    "Guinea-Bissau",
    "Timor-Leste",
    "Qatar",
    "Romania",
    "Rwanda",
    "Saint Helena, Ascension and Tristan da Cunha",
    "Saint Kitts and Nevis",
    "Saint Lucia",
    "Saint Pierre and Miquelon",
    "Saint Vincent and the Grenadines",
    "San Marino",
    "Sao Tome and Principe",
    "Saudi Arabia",
    "Senegal",
    "Serbia",
    "Seychelles",
    "Sierra Leone",
    "Singapore",
    "Slovakia",
    "Vietnam",
    "Slovenia",
    "Somalia",
    "South Africa",
    "Zimbabwe",
    "Spain",
    "Suriname",
    "Eswatini",
    "Sweden",
    "Switzerland",
    "Tajikistan",
    "Thailand",
    "Togo",
    "Tokelau",
    "Tonga",
    "Trinidad and Tobago",
    "United Arab Emirates",
    "Tunisia",
    "Turkiye",
    "Turkmenistan",
    "Tuvalu",
    "Uganda",
    "Ukraine",
    "North Macedonia",
    "Egypt",
    "United Kingdom",
    "Guernsey",
    "Jersey",
    "Isle of Man",
    "Tanzania",
    "United States",
    "Burkina Faso",
    "Uruguay",
    "Uzbekistan",
    "Venezuela",
    "Wallis and Futuna",
    "Samoa",
    "Yemen",
    "Zambia",
]

# Patch a method...


def work_hours_from_dict(obj: dict[str, Any] | None) -> WorkHours | None:
    if obj is None:
        return None
    if not isinstance(obj, dict):
        return WorkHours.model_validate(obj)
    _obj = WorkHours.model_validate(
        {
            "timetable": (
                dict(
                    (
                        _k,
                        (
                            [WorkDayInfo.from_dict(_item) for _item in _v]
                            if _v is not None
                            else None
                        ),
                    )
                    for _k, _v in obj.get("timetable", {}).items()
                )
                if obj.get("timetable") is not None
                else {}
            ),
            "current_status": obj.get("current_status"),
        }
    )
    return _obj


WorkHours.from_dict = work_hours_from_dict
