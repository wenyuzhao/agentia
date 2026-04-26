import os
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence, Union

import frontmatter
from pydantic import BaseModel, Field


class MarkdownDoc(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: str
    attachments: dict[str, Union["MarkdownDoc", Path]] = Field(default_factory=dict)


_MARKDOWN_EXTS = {".md", ".markdown"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
_PDF_EXTS = {".pdf"}

# @path/to/file.ext — must not be preceded by an alphanumeric/underscore (avoids matching emails like user@host.com).
_FILE_REF_RE = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z0-9_./\-]+\.[A-Za-z0-9]+)")

# !`command` — inline bash command substitution.
_BASH_RE = re.compile(r"!`([^`]+)`")

# ${VAR} — environment-variable-like substitution.
_BRACE_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# $ARGUMENTS[N] — indexed access into the argument list.
_ARGS_INDEX_RE = re.compile(r"\$ARGUMENTS\[(\d+)\]")

# $N — shorthand for $ARGUMENTS[N].
_DOLLAR_NUM_RE = re.compile(r"\$(\d+)")

# $name — named argument (resolved against the `arguments` frontmatter list).
_DOLLAR_NAME_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def load_markdown(
    md: str,
    *,
    arguments: Sequence[str] | None = None,
) -> MarkdownDoc:
    """Load a markdown document from a path, resolving inline directives.

    The input may be either a path to a markdown file or raw markdown
    content. Three additional pieces of syntax are supported:

    - ``@path/to/file.{md,png,pdf,...}`` references another file relative to
      the source markdown's directory. Markdown files are loaded recursively
      into ``attachments`` as ``MarkdownDoc``; images and PDFs are stored as
      ``Path`` objects. Unsupported or missing files are left untouched.
    - ``!`cmd``` runs ``cmd`` through the shell and substitutes the captured
      stdout in place.
    - String substitutions: ``$ARGUMENTS``, ``$ARGUMENTS[N]``, ``$N``,
      ``$name`` (matched against the frontmatter ``arguments`` list),
      ``${CLAUDE_SKILL_DIR}`` (the source file's directory), and
      ``${CLAUDE_SESSION_ID}`` and other ``${VAR}`` references resolved via
      the environment.
    """
    text, base_dir = _read_source(md)

    fm = frontmatter.loads(text)
    metadata: dict[str, Any] = dict(fm.metadata)
    content = fm.content

    args = list(arguments) if arguments is not None else []
    arg_names: list[str] = []
    raw_arg_names = metadata.get("arguments")
    if isinstance(raw_arg_names, list):
        arg_names = [str(n) for n in raw_arg_names]

    content = _apply_bash_substitution(content, base_dir)
    content = _apply_substitutions(content, args, arg_names, base_dir)
    attachments = _collect_attachments(content, base_dir, arguments=arguments)

    return MarkdownDoc(metadata=metadata, content=content, attachments=attachments)


def _read_source(md: str) -> tuple[str, Path]:
    path = Path(md)
    if path.is_file():
        return path.read_text(encoding="utf-8"), path.resolve().parent
    return md, Path.cwd().resolve()


def _apply_bash_substitution(content: str, cwd: Path) -> str:
    def repl(match: re.Match[str]) -> str:
        cmd = match.group(1)
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(cwd),
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return f"<error: `{cmd}` timed out>"
        except OSError as e:
            return f"<error executing `{cmd}`: {e}>"
        return result.stdout.rstrip("\n")

    return _BASH_RE.sub(repl, content)


def _apply_substitutions(
    content: str,
    args: list[str],
    arg_names: list[str],
    base_dir: Path,
) -> str:
    def brace_repl(match: re.Match[str]) -> str:
        var = match.group(1)
        if var == "CLAUDE_SKILL_DIR":
            return str(base_dir)
        return os.environ.get(var, match.group(0))

    content = _BRACE_VAR_RE.sub(brace_repl, content)

    def args_index_repl(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return args[idx] if 0 <= idx < len(args) else ""

    content = _ARGS_INDEX_RE.sub(args_index_repl, content)

    # $ARGUMENTS must be replaced before the named-argument regex below (which
    # would otherwise greedily match it as `ARGUMENTS`).
    content = content.replace("$ARGUMENTS", " ".join(args))

    def num_repl(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return args[idx] if 0 <= idx < len(args) else ""

    content = _DOLLAR_NUM_RE.sub(num_repl, content)

    name_to_value = {
        name: (args[i] if i < len(args) else "")
        for i, name in enumerate(arg_names)
    }

    def name_repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in name_to_value:
            return name_to_value[name]
        return match.group(0)

    content = _DOLLAR_NAME_RE.sub(name_repl, content)

    return content


def _collect_attachments(
    content: str,
    base_dir: Path,
    *,
    arguments: Sequence[str] | None,
) -> dict[str, MarkdownDoc | Path]:
    attachments: dict[str, MarkdownDoc | Path] = {}
    for match in _FILE_REF_RE.finditer(content):
        ref = match.group(1)
        if ref in attachments:
            continue
        ref_path = Path(ref)
        resolved = ref_path if ref_path.is_absolute() else base_dir / ref_path
        if not resolved.is_file():
            continue
        ext = resolved.suffix.lower()
        if ext in _MARKDOWN_EXTS:
            attachments[ref] = load_markdown(str(resolved), arguments=arguments)
        elif ext in _IMAGE_EXTS or ext in _PDF_EXTS:
            attachments[ref] = resolved
    return attachments
