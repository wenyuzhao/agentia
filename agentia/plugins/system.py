from . import Plugin, tool
import asyncio
import os
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
        timeout: Annotated[int | None, "Optional timeout in seconds."] = 5 * 60,
        background: Annotated[
            bool,
            "If true, start the command detached and return immediately with its PID. Use `TaskStop` to terminate it later.",
        ] = False,
    ) -> str:
        """
        Run a bash command and return its combined stdout and stderr.

        **Truncation:**
            Output is truncated to ~1000 lines or 256KB (whichever fires first), keeping the first 30% from the head and the last 70% from the tail.
            When truncation happens, the full output is spooled to a temp file whose path is included in the result.

        **Non-zero exit code:**
            When the command exits with a non-zero code, it is appended to the output.

        **Background / timeout:**
            If `background=True`, the process is started detached and the call returns immediately with its PID and the path of the log file its stdout/stderr is streaming to.
            If `timeout` fires for a foreground command, the process is left running in the background.
            `timeout` is set to 5 minutes by default, and has no effect if `background=True`.
        """
        await self.agent.user_consent_guard("Run command: " + command)

        fd, log_path = tempfile.mkstemp(prefix="agentia-bash-", suffix=".log")
        os.close(fd)
        with open(log_path, "wb") as log_write:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=log_write,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                start_new_session=True,
            )

        if background:
            return f"[Started background process pid={proc.pid}. Output streaming to: {log_path}. Use `TaskStop` tool to terminate.]"

        try:
            returncode = await asyncio.wait_for(proc.wait(), timeout=timeout)
        except TimeoutError:
            return (
                f"[Command timed out after {timeout}s. Process is still running in background pid={proc.pid}. "
                f"Output streaming to: {log_path}. Use `TaskStop` tool to terminate.]"
            )

        output = Path(log_path).read_text(encoding="utf-8", errors="replace")
        try:
            os.unlink(log_path)
        except OSError:
            pass

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

    @tool(name="TaskStop")
    async def task_stop(
        self,
        pid: Annotated[int, "Process ID of the task to stop."],
    ) -> str:
        """
        Stop a running process by PID using the `kill` command. Sends SIGTERM first;
        if the process is still alive after a brief grace period, escalates to SIGKILL.
        Use this to terminate background processes started by `Bash` (either via `background=True` or after a timeout).
        """
        await self.agent.user_consent_guard(f"Kill process: {pid}")

        async def _kill(signal: str) -> tuple[int, str]:
            proc = await asyncio.create_subprocess_exec(
                "kill",
                signal,
                str(pid),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            assert proc.returncode is not None
            return proc.returncode, stderr.decode(errors="replace").strip()

        async def _alive() -> bool:
            proc = await asyncio.create_subprocess_exec(
                "kill",
                "-0",
                str(pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0

        rc, err = await _kill("-TERM")
        if rc != 0:
            msg = err or f"kill exited with code {rc}"
            return f"[Failed to kill pid={pid}: {msg}]"

        for _ in range(20):
            await asyncio.sleep(1)
            if not await _alive():
                return f"[Sent SIGTERM to pid={pid}]"

        rc, err = await _kill("-KILL")
        if rc != 0:
            msg = err or f"kill -9 exited with code {rc}"
            return f"[SIGTERM did not stop pid={pid}; SIGKILL also failed: {msg}]"
        return f"[SIGTERM did not stop pid={pid}; sent SIGKILL]"

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
