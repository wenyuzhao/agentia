"""Pydantic models for the stateless OpenAI Responses API.

Agentia models the Responses API directly: the conversation is a list of `Item`s, not
chat messages, and stream events keep the API's own `type` strings. Nothing is stored
server-side, so every request carries the full input.

Responses API types
-------------------
- `base`     -- `usage`, tool definitions, `url_citation` annotations, `tool_choice`
- `content`  -- content parts: `input_text`, `input_image`, `input_file`, `input_audio`,
                `output_text`, `refusal`, `reasoning_text`, `summary_text`
- `items`    -- the `ItemField` union: `message`, `reasoning`, `function_call`,
                `function_call_output`, `compaction`
- `options`  -- the `text.format` shapes: `text`, `json_object`, `json_schema`
- `stream`   -- 22 `response.*` server-sent event types, plus `error`

Agentia extensions
------------------
- `base`     -- `File`, `FinishReason`, `UserConsentRequest`, `ToolSchema`
- `content`  -- the `InputContent` / `OutputContent` / `Content` unions,
                `input_content_from_file()`
- `items`    -- `Item` / `InputItem` / `MessageInput` aliases, `Message.user()` and the
                other role helpers, `to_responses_input()`, and client-side
                `Compaction` semantics
- `options`  -- the `ResponseFormat` wrapper types themselves
- `stream`   -- `agentia.run.started`, `agentia.run.completed`,
                `agentia.input_item.added`, and `ResponseInfo`

See `docs/usage.md` for the full mapping.
"""

from .base import *
from .content import *
from .items import *
from .options import *
from .stream import *
