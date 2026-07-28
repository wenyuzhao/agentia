from . import Plugin, tool
import asyncio
import os
import signal
import tempfile
import weakref
from typing import Annotated, Any, Iterable, Literal, override
from pathlib import Path
from ..tools import ToolResult
from ..models.base import File
import base64
from asyncio.subprocess import STDOUT


_DEFAULT_MAX_LINES = 1000
_DEFAULT_MAX_BYTES = 256 * 1024
_DEFAULT_HEAD_RATIO = 0.3


def _terminate_background_pids(pids: set[int]) -> None:
    for pid in list(pids):
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


_FILE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
}

Modality = Literal["image", "audio", "video", "pdf"]
_ALL_MODALITIES: tuple[Modality, ...] = ("image", "audio", "video", "pdf")


def _modality_of(media_type: str) -> Modality | None:
    """Map a media type to its high-level modality, or None if unrecognized."""
    major = media_type.split("/", 1)[0]
    if major in ("image", "audio", "video"):
        return major  # type: ignore[return-value]
    if media_type == "application/pdf":
        return "pdf"
    return None


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
    def __init__(
        self,
        modalities: Iterable[Modality] | None = None,
    ) -> None:
        super().__init__()
        # Modalities the `LoadFile` tool is allowed to load. `None` means all.
        self.modalities: set[Modality] = (
            set(modalities) if modalities is not None else set(_ALL_MODALITIES)
        )
        self._background_pids: set[int] = set()
        weakref.finalize(self, _terminate_background_pids, self._background_pids)

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
        timeout: Annotated[
            int | None, "Optional timeout in seconds. Set to 0 to disable."
        ] = 5 * 60,
        background: Annotated[
            bool,
            "If true, start the command detached and return immediately with its PID.",
        ] = False,
    ) -> str:
        """
        Run a bash command and return its combined stdout and stderr (no need to do the redirection yourself).

        **Truncation:**
            * Large output is truncated with the full output spooled to a temp file.

        **Non-zero exit code:**
            * When the command exits with a non-zero code, it is appended to the output.

        **Background / timeout:**
            * If `background=True`, the process is started detached and the call returns immediately with its PID and the path of the log file its stdout/stderr is streaming to.
            * While running background tasks, any new output is polled and enqueued regularly.
            * If `timeout` fires for a foreground command, the process is left running in the background. `timeout` is set to 5 minutes by default, and has no effect if `background=True`.
                * set timeout = 0 to disable the timeout and run indefinitely in the foreground. e.g. when running `sleep 1h`.
            * Use background tasks for long-running or daemon processes (e.g. dev servers). Only use this when necessary.
        """
        if timeout is not None and timeout <= 0:
            timeout = None

        await self.agent.user_consent_guard("Run command: " + command)

        # Stream both stdout and stderr to a single temp log file so we can
        # both return the combined output and (in background mode) tail it.
        fd, log_path = tempfile.mkstemp(prefix="agentia-bash-", suffix=".log")
        os.close(fd)
        env = os.environ.copy()
        # Pass session-local environment variables to the subprocess
        env.update(self.agent.env)
        with open(log_path, "wb") as log_write:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=log_write,
                stderr=STDOUT,
                cwd=cwd,
                env=env,
                # Detach into a new session so timeouts/background mode leave
                # the process alive without us having to babysit its tty.
                start_new_session=True,
            )

        if background:
            # Hand monitoring off to a fire-and-forget task and return now.
            # The task tails `log_path` and feeds output back via agent.enqueue
            # so the model sees progress without us blocking the tool call.
            self._background_pids.add(proc.pid)
            return (
                f"[Started background process pid={proc.pid}. Output streaming to: {log_path}. "
                f"New output is enqueued regularly. Use the `kill` command to terminate.]"
            )

        try:
            returncode = await asyncio.wait_for(proc.wait(), timeout=timeout)
        except TimeoutError:
            self._background_pids.add(proc.pid)
            return (
                f"[Command timed out after {timeout}s. Process is still running in background pid={proc.pid}. "
                f"Output streaming to: {log_path}. New output is enqueued regularly. Use the `kill` command to terminate.]"
            )

        output = Path(log_path).read_text(encoding="utf-8", errors="replace")
        try:
            os.unlink(log_path)
        except OSError:
            pass

        # Head-heavy split keeps the start of the output (often the most
        # diagnostic) while still preserving the tail where errors usually land.
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
        if returncode != 0:
            body += f"\n\n[Command exited with code {returncode}]"
        return body

    @tool(name="LoadFile")
    async def load_file(
        self,
        path: Annotated[
            str, "Local path, data: URL, or web URL to the media file"
        ],
    ) -> Any:
        """
        Load a media file (image, PDF, audio, video, ...) so that you can inspect its content.
            * Local paths, `data:` URLs, and web URLs are supported.
            * The media is returned as an attachment with the appropriate media type.
            * This tool does NOT read text files. To read a text file, use the Bash tool (e.g. `cat`) instead.
        """
        from .web import Web

        # Base64-encoded data URL
        if path.startswith("data:"):
            # get content type from data URL, default to application/octet-stream
            content_type = "application/octet-stream"
            if ";" in path and path.index(";") < path.index(","):
                content_type = path[5 : path.index(";")]
            self.__check_modality(path, content_type)
            return ToolResult(
                files=[File(media_type=content_type, data=path)],
                output=f"Loaded data URL ({len(path)} characters).",
            )

        # Web content
        if path.startswith(("http://", "https://")):
            result = await Web.http_get(path)
            if isinstance(result, ToolResult):
                for f in result.files or []:
                    self.__check_modality(path, f.media_type)
                return result
            # Textual web content is not a media file.
            raise ValueError(
                f"'{path}' returned text content ({result.content_type}), not a media file. "
                "Use the WebFetch tool to fetch textual web pages instead."
            )

        # Check path
        p = Path(path)
        if not p.exists():
            raise ValueError(f"File '{path}' does not exist.")
        if p.is_dir():
            raise ValueError(f"'{path}' is a directory, not a file.")

        # Only known media types are supported; text files should be read via Bash.
        suffix = p.suffix.lower()
        if suffix not in _FILE_MIME_TYPES:
            raise ValueError(
                f"'{path}' is not a supported media file. This tool only loads media files "
                f"(one of: {', '.join(sorted(_FILE_MIME_TYPES))}). "
                "To read a text file, use the Bash tool (e.g. `cat`) instead."
            )

        media_type = _FILE_MIME_TYPES[suffix]
        self.__check_modality(path, media_type)
        encoded = base64.b64encode(p.read_bytes()).decode()
        url = f"data:{media_type};base64,{encoded}"
        return ToolResult(
            files=[File(media_type=media_type, data=url)],
            output=f"Loaded file '{path}' ({media_type}).",
        )

    def __check_modality(self, path: str, media_type: str) -> None:
        """Reject media whose modality is not among the enabled ones."""
        modality = _modality_of(media_type)
        if modality not in self.modalities:
            enabled = ", ".join(sorted(self.modalities)) or "none"
            raise ValueError(
                f"'{path}' has media type '{media_type}', which is not an enabled "
                f"modality (enabled: {enabled})."
            )
