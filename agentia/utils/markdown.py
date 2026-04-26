import os
import re
import subprocess
from pathlib import Path
from typing import Any, Literal, Union

import frontmatter
from pydantic import BaseModel, Field


class Attachment(BaseModel):
    path: Path
    content: Union["MarkdownDoc", Path]

    def get_content(self) -> str:
        if isinstance(self.content, MarkdownDoc):
            return self.content.content.strip() or "(no content)"
        elif not self.path.is_file():
            return f"<missing file: {self.path}>"
        elif self.path.suffix.lower() in _TEXT_EXTS:
            return self.path.read_text(encoding="utf-8").strip() or "(no content)"
        elif self.path.suffix.lower() in _BINARY_EXTS:
            return (
                f"File: {self.path}. Please use the `Read` tool to access its content."
            )
        else:
            return f"Unsupported file type: {self.path.suffix}"


class MarkdownDoc(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: str
    attachments: dict[str, Attachment] = Field(default_factory=dict)

    def flatten_attachments(self) -> dict[str, Attachment]:
        flat: dict[str, Attachment] = {}
        for ref, attachment in self.attachments.items():
            flat[str(attachment.path)] = attachment
            if isinstance(attachment.content, MarkdownDoc):
                flat.update(attachment.content.flatten_attachments())
        return flat


Attachment.model_rebuild()

_MARKDOWN_EXTS = {".md", ".markdown"}
_TEXT_EXTS = {
    ".txt",
    ".text",
    ".csv",
    ".json",
    ".jsonl",
    ".toml",
    ".yaml",
    ".yml",
    ".svg",
}
_BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".pdf"}

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
    md: str | Path,
    *,
    args: list[str] | None = None,
    substitutions: list[Literal["bash", "args", "file"]] | None = None,
) -> MarkdownDoc:
    """Load a markdown document from a path, resolving inline directives.

    The input may be either a path to a markdown file or raw markdown
    content. Three additional pieces of syntax are supported, each
    controllable via a flag:

    - ``@path/to/file.{md,png,pdf,...}`` (``file_refs``) references another
      file relative to the source markdown's directory. Markdown files are
      loaded recursively into ``attachments`` as ``MarkdownDoc``; images
      and PDFs are stored as ``Path`` objects. Unsupported or missing
      files are left untouched.
    - ``!`cmd``` (``bash``) runs ``cmd`` through the shell and substitutes
      the captured stdout in place.
    - String substitutions (``substitutions``): ``$ARGUMENTS``,
      ``$ARGUMENTS[N]``, ``$N``, ``$name`` (matched against the frontmatter
      ``arguments`` list), ``${CLAUDE_SKILL_DIR}`` (the source file's
      directory), and ``${CLAUDE_SESSION_ID}`` and other ``${VAR}``
      references resolved via the environment.

    Set any flag to ``False`` to leave the corresponding syntax untouched.
    """
    text, base_dir = _read_source(md)

    fm = frontmatter.loads(text)
    metadata: dict[str, Any] = dict(fm.metadata)
    content = fm.content

    args = args if args is not None else []
    arg_names: list[str] = []
    raw_arg_names = metadata.get("arguments")
    if isinstance(raw_arg_names, list):
        arg_names = [str(n) for n in raw_arg_names]

    substitutions = substitutions or []
    if "bash" in substitutions:
        content = _apply_bash_substitution(content, base_dir)
    if "args" in substitutions:
        content = _apply_substitutions(content, args, arg_names, base_dir)
    if "file" in substitutions:
        attachments = _collect_attachments(
            content, base_dir, args=args, substitutions=substitutions
        )
    else:
        attachments = {}

    return MarkdownDoc(metadata=metadata, content=content, attachments=attachments)


def _read_source(md: str | Path) -> tuple[str, Path]:
    if isinstance(md, Path) or (isinstance(md, str) and os.path.isfile(md)):
        path = Path(md)
        return path.read_text(encoding="utf-8"), path.resolve().parent
    else:
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
        if var == "CLAUDE_SKILL_DIR" or var == "SKILL_DIR":
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
        name: (args[i] if i < len(args) else "") for i, name in enumerate(arg_names)
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
    args: list[str] | None,
    substitutions: list[Literal["bash", "args", "file"]],
) -> dict[str, Attachment]:
    attachments: dict[str, Attachment] = {}
    print("_collect_attachments")
    for match in _FILE_REF_RE.finditer(content):
        print(f"Found file reference: {match.group(0)}")
        ref = match.group(1)
        if ref in attachments:
            print(f"Reference '{ref}' already collected, skipping.")
            continue
        ref_path = Path(ref)
        resolved = ref_path if ref_path.is_absolute() else base_dir / ref_path
        if not resolved.is_file():
            print(
                f"Resolved path '{resolved}' does not exist or is not a file, skipping."
            )
            continue
        ext = resolved.suffix.lower()
        if ext in _MARKDOWN_EXTS:
            doc = load_markdown(
                str(resolved),
                args=args,
                substitutions=substitutions,
            )
            attachments[ref] = Attachment(path=resolved, content=doc)
        elif ext in _TEXT_EXTS or ext in _BINARY_EXTS:
            attachments[ref] = Attachment(path=resolved, content=resolved)
        else:
            print(f"Unsupported file type '{ext}' for path '{resolved}', skipping.")
    return attachments
