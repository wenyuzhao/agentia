from pathlib import Path

import pytest

from agentia.utils.markdown import MarkdownDoc, load_markdown


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_load_markdown_parses_frontmatter_and_content(workspace: Path):
    md = write(
        workspace / "doc.md",
        "---\ntitle: Hello\n---\n# Heading\n\nbody",
    )
    doc = load_markdown(str(md))
    assert doc.metadata == {"title": "Hello"}
    assert doc.content.strip() == "# Heading\n\nbody"
    assert doc.attachments == {}


def test_load_markdown_resolves_at_path_attachments(workspace: Path):
    write(workspace / "child.md", "child content")
    write(workspace / "img" / "pic.png", "")
    main = write(
        workspace / "main.md",
        "see @child.md and the screenshot @img/pic.png here.",
    )
    doc = load_markdown(str(main))
    assert "@child.md" in doc.content
    assert "@img/pic.png" in doc.content
    assert isinstance(doc.attachments["child.md"], MarkdownDoc)
    assert doc.attachments["child.md"].content.strip() == "child content"
    assert doc.attachments["img/pic.png"] == (workspace / "img" / "pic.png")


def test_load_markdown_ignores_email_like_at_tokens(workspace: Path):
    md = write(workspace / "doc.md", "contact me at user@example.com please")
    doc = load_markdown(str(md))
    assert doc.attachments == {}


def test_load_markdown_skips_missing_or_unsupported_refs(workspace: Path):
    write(workspace / "data.csv", "x,y")
    md = write(
        workspace / "doc.md",
        "missing @nope.md and unsupported @data.csv",
    )
    doc = load_markdown(str(md))
    assert doc.attachments == {}


def test_load_markdown_runs_bash_substitution(workspace: Path):
    md = write(workspace / "doc.md", "hello !`echo world` end")
    doc = load_markdown(str(md))
    assert doc.content.strip() == "hello world end"


def test_load_markdown_bash_runs_in_source_directory(workspace: Path):
    md = write(workspace / "nested" / "doc.md", "cwd=!`pwd`")
    doc = load_markdown(str(md))
    assert str((workspace / "nested").resolve()) in doc.content


def test_load_markdown_substitutes_arguments(workspace: Path):
    md = write(
        workspace / "doc.md",
        "all=$ARGUMENTS first=$ARGUMENTS[0] second=$1",
    )
    doc = load_markdown(str(md), arguments=["alpha", "beta"])
    assert "all=alpha beta" in doc.content
    assert "first=alpha" in doc.content
    assert "second=beta" in doc.content


def test_load_markdown_substitutes_named_arguments(workspace: Path):
    md = write(
        workspace / "doc.md",
        "---\narguments: [issue, branch]\n---\nfix $issue on $branch ($unknown)",
    )
    doc = load_markdown(str(md), arguments=["123", "main"])
    assert "fix 123 on main" in doc.content
    assert "$unknown" in doc.content


def test_load_markdown_substitutes_skill_dir_and_env(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sess-42")
    md = write(
        workspace / "doc.md",
        "dir=${CLAUDE_SKILL_DIR} sid=${CLAUDE_SESSION_ID}",
    )
    doc = load_markdown(str(md))
    assert f"dir={workspace.resolve()}" in doc.content
    assert "sid=sess-42" in doc.content


def test_load_markdown_unknown_brace_var_is_preserved(workspace: Path):
    md = write(workspace / "doc.md", "x=${DEFINITELY_NOT_SET_VAR}")
    doc = load_markdown(str(md))
    assert "${DEFINITELY_NOT_SET_VAR}" in doc.content


def test_load_markdown_recursive_attachments_inherit_arguments(workspace: Path):
    write(workspace / "child.md", "I am $0")
    md = write(workspace / "main.md", "see @child.md")
    doc = load_markdown(str(md), arguments=["alice"])
    child = doc.attachments["child.md"]
    assert isinstance(child, MarkdownDoc)
    assert "I am alice" in child.content
