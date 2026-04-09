from typing import override
from ._openai_api import OpenAIAPIProvider
from .openrouter import OpenRouter as _OpenRouter
import os

# Model list: https://developers.cloudflare.com/ai-gateway/usage/providers/


class Cloudflare(OpenAIAPIProvider):
    def __init__(self, model: str, api_key: str | None = None):
        api_key = (
            api_key
            or os.environ.get("CLOUDFLARE_AI_GATEWAY_API_KEY")
            or os.environ.get("CLOUDFLARE_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "CLOUDFLARE_AI_GATEWAY_API_KEY or CLOUDFLARE_API_KEY environment variable not set"
            )
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID environment variable not set")
        base_url = f"https://gateway.ai.cloudflare.com/v1/{account_id}/default/compat"
        super().__init__(
            name="cloudflare", model=model, api_key=api_key, base_url=base_url
        )
        # self.extra_headers["cf-aig-authorization"] = f"Bearer {api_key}"

        self.extra_body["modalities"] = ["text", "image"]

    @override
    async def _fetch_context_length(self) -> int:
        # Cloudflare AI Gateway doesn't expose context length directly,
        # so we query OpenRouter's model catalog as a fallback.
        return await _OpenRouter.fetch_context_length(self.model)
