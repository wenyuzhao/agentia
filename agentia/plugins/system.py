from . import Plugin, tool
import os
import subprocess
import tempfile
from typing import Annotated, override
from pathlib import Path
from ..tools import ToolResult
from ..models.base import File
import base64


_DEFAULT_MAX_LINES = 1000
_DEFAULT_MAX_BYTES = 256 * 1024
_DEFAULT_HEAD_RATIO = 0.3

_FILE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


def _truncate(
    content: str,
    *,
    max_head_lines: int,
    max_tail_lines: int,
    save_full: bool = False,
) -> tuple[str, str | None]:  # returns (truncated_content, full_content_path)
    """
    Keep up to `max_head_lines` from the start and `max_tail_lines` from the end;
    lines in between are replaced with a `[... N lines omitted ...]` marker.
    Set `max_tail_lines=0` for head-only truncation, or `max_head_lines=0` for tail-only.
    The byte budget (`_DEFAULT_MAX_BYTES`) is split between head and tail in proportion to their line caps.
    """
    max_lines = max_head_lines + max_tail_lines
    if not content or max_lines == 0:
        return "", None
    total_bytes = len(content.encode("utf-8"))
    lines = content.split("\n")
    total_lines = len(lines)
    if total_bytes <= _DEFAULT_MAX_BYTES and total_lines <= max_lines:
        return content, None
    max_head_bytes = _DEFAULT_MAX_BYTES * max_head_lines // max_lines
    max_tail_bytes = _DEFAULT_MAX_BYTES - max_head_bytes

    def _take(reverse: bool, lim_lines: int, lim_bytes: int) -> list[str]:
        kept: list[str] = []
        used = 0
        for line in reversed(lines) if reverse else lines:
            if len(kept) >= lim_lines:
                break
            sz = len(line.encode("utf-8")) + (1 if kept else 0)
            if kept and used + sz > lim_bytes:
                break
            kept.append(line)
            used += sz
        return list(reversed(kept)) if reverse else kept

    head = _take(False, max_head_lines, max_head_bytes)
    tail = _take(True, max_tail_lines, max_tail_bytes)

    if len(head) + len(tail) >= total_lines:
        body = content
    elif not tail:
        body = "\n".join(head)
    elif not head:
        body = "\n".join(tail)
    else:
        omitted = total_lines - len(head) - len(tail)
        body = "\n".join(head + [f"[... {omitted} lines omitted ...]"] + tail)

    path: str | None = None
    if save_full:
        fd, path = tempfile.mkstemp(prefix="agentia-bash-", suffix=".log", text=True)
        with os.fdopen(fd, "w") as f:
            f.write(content)
    return body, path


class System(Plugin):
    @override
    def get_instructions(self) -> str | None:
        return f"CURRENT WORKING DIRECTORY: `{str(Path.cwd())}`"

    @tool(name="Bash")
    async def run_bash_command(
        self,
        command: Annotated[str, "The bash command to run."],
        cwd: Annotated[
            str | None, "The optional working directory for the command."
        ] = None,
        timeout: Annotated[int | None, "Optional timeout in seconds."] = None,
    ) -> str:
        """
        Run a bash command and return its combined stdout and stderr.

        **Truncation:**
            Output is truncated to ~1000 lines or 256KB (whichever fires first), keeping the first 30% from the head and the last 70% from the tail.
            When truncation happens, the full output is spooled to a temp file whose path is included in the result.

        **Non-zero exit code:**
            When the command exits with a non-zero code, it is appended to the output.
        """
        await self.agent.user_consent_guard("Run command: " + command)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[Command timed out after {timeout}s: {command}]"

        output: str = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n\n"
            output += result.stderr

        head_lines = int(_DEFAULT_MAX_LINES * _DEFAULT_HEAD_RATIO)
        body, full_path = _truncate(
            output,
            max_head_lines=head_lines,
            max_tail_lines=_DEFAULT_MAX_LINES - head_lines,
            save_full=True,
        )
        if not body:
            body = "(no output)"
        if full_path:
            body += f"\n\n[Output truncated. Full output saved to: {full_path}]"
        if result.returncode != 0:
            body += f"\n\n[Command exited with code {result.returncode}]"
        return body

    @tool(name="Read")
    async def read_file(
        self,
        path: Annotated[str, "Path to the file"],
        offset: Annotated[int | None, "Start line number (1-indexed)"] = None,
        limit: Annotated[int | None, "Max number of lines to read"] = None,
    ) -> str | ToolResult:
        """
        Read the contents of a file so that you can inspect their content.
            * Text files: return the UTF-8 text content (maybe truncated).
            * Images and PDFs: return as attachments with appropriate media types.
        """
        p = Path(path)
        if not p.exists():
            raise ValueError(f"File '{path}' does not exist.")
        if p.is_dir():
            raise ValueError(f"'{path}' is a directory, not a file.")

        # Non-text files
        suffix = p.suffix.lower()
        if suffix in _FILE_MIME_TYPES:
            media_type = _FILE_MIME_TYPES[suffix]
            encoded = base64.b64encode(p.read_bytes()).decode()
            url = f"data:{media_type};base64,{encoded}"
            return ToolResult(
                files=[File(media_type=media_type, data=url)],
                output=f"Loaded file '{path}' ({media_type}).",
            )

        # Text files
        text = p.read_text(encoding="utf-8")
        all_lines = text.splitlines()
        total = len(all_lines)
        start = max(1, offset) if offset else 1
        if total > 0 and start > total:
            raise ValueError(
                f"Offset {offset} is out of range for file with {total} lines."
            )
        max_lines = min(limit, _DEFAULT_MAX_LINES) if limit else _DEFAULT_MAX_LINES
        sliced = "\n".join(all_lines[start - 1 : start - 1 + max_lines])
        body, _ = _truncate(sliced, max_head_lines=max_lines, max_tail_lines=0)
        kept = body.count("\n") + 1 if body else 0
        end = start + kept - 1
        if end < total:
            body += (
                f"\n\n[Showed lines {start}-{end} of {total}. "
                f"Pass offset={end + 1} to continue.]"
            )
        return body

    @tool(name="Write")
    async def write_file(
        self,
        path: Annotated[str, "Path to the file"],
        content: Annotated[str, "Full content to write"],
    ) -> str:
        """
        Write `content` to `path`. Missing parent directories are created automatically. Existing content is overwritten.
        """
        await self.agent.user_consent_guard(f"Write file: {path}")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} characters to '{path}'."

    @tool(name="Edit")
    async def edit_file(
        self,
        path: Annotated[str, "Path to the file"],
        old_text: Annotated[
            str,
            "Exact text to replace. Must appear verbatim: including whitespace and newlines, and exactly once in the file.",
        ],
        new_text: Annotated[str, "Replacement text."],
    ) -> str:
        """
        Surgically replace a unique substring in a file. `old_text` must match
        exactly and appear exactly once. The edit fails — leaving the file
        untouched — if there are zero or multiple matches, or if `new_text`
        equals `old_text`. For larger rewrites, use Write instead.
        """
        await self.agent.user_consent_guard(f"Edit file: {path}")
        p = Path(path)
        if not p.exists():
            raise ValueError(f"File '{path}' does not exist.")
        if p.is_dir():
            raise ValueError(f"'{path}' is a directory, not a file.")
        content = p.read_text()
        matches = content.count(old_text)
        if matches == 0:
            raise ValueError(
                f"`old_text` was not found in '{path}'. The match must be exact, including whitespace and newlines."
            )
        if matches > 1:
            raise ValueError(
                f"`old_text` matches {matches} places in '{path}'. Include more surrounding context so the match is unique."
            )
        if old_text == new_text:
            raise ValueError(
                f"No change applied to '{path}': `old_text` and `new_text` are identical."
            )
        p.write_text(content.replace(old_text, new_text, 1))
        return f"Edited '{path}': replaced {len(old_text)} characters with {len(new_text)}."
