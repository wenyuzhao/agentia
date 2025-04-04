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

    @tool
    def google_search(self, keywords: Annotated[str, "The keywords to search"]):
        """Perform Google Search using the given keywords. Returning the top 10 search result in json format. When necessary, you need to combine this tool with the getWebPageContent tools (if available), to browse the web in depth by jumping through links."""
        response = self.__api.google_organic_live_regular(
            [
                SerpGoogleOrganicLiveRegularRequestInfo(
                    language_code="en",
                    location_name=self.__country,
                    keyword=keywords,
                    depth=10,
                )
            ]
        )
        return self.__process_result(response)

    @tool
    def google_news_search(self, keywords: Annotated[str, "The keywords to search"]):
        """Perform Google News Search using the given keywords, to search news related to the given topics. Returning the top 5 search result in json format."""
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

    @tool
    def google_map_search(self, keywords: Annotated[str, "The keywords to search"]):
        """Perform Google Map Search using the given keywords. Returning the top 10 search result in json format. This is helpful to get the address, website, opening hours, and contact information of a place or store."""
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
