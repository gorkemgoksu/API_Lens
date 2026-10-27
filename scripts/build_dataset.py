"""Build the deterministic APILens dataset from pinned documentation sources."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import tempfile
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import spans_to_tags, tokenize, validate


RAW_DIR = ROOT / "data" / "raw"
SOURCES_PATH = RAW_DIR / "SOURCES.json"
TEXT_SUFFIXES = {".rst", ".md", ".txt"}

API_ROLES = {
    "func",
    "meth",
    "class",
    "mod",
    "attr",
    "data",
    "const",
    "exc",
    "obj",
}
FUNCTION_ROLES = {"func", "meth"}
DEFAULT_API_ROLES = {"obj", "py:obj", "autolink"}
INVENTORY_ROLE_TO_TYPE = {
    "function": "FUNCTION",
    "method": "METHOD",
    "classmethod": "METHOD",
    "staticmethod": "METHOD",
    "class": "CLASS",
    "module": "MODULE",
    "attribute": "ATTRIBUTE",
    "property": "ATTRIBUTE",
    "data": "ATTRIBUTE",
    "exception": "EXCEPTION",
}
CANDIDATE_ROLE_TO_TYPE = {
    "func": "FUNCTION",
    "meth": "METHOD",
    "class": "CLASS",
    "mod": "MODULE",
    "attr": "ATTRIBUTE",
    "data": "ATTRIBUTE",
    "const": "ATTRIBUTE",
    "exc": "EXCEPTION",
}
SKIPPED_DIRECTIVES = {
    "autosummary",
    "code",
    "code-block",
    "doctest",
    "ipython",
    "ipython::",
    "jupyter-execute",
    "literalinclude",
    "plot",
    "testcode",
    "testoutput",
    "toctree",
}

DIRECTIVE_RE = re.compile(r"^(?P<indent>\s*)\.\.\s+(?P<name>[\w:-]+)::\s*(?P<arg>.*)$")
MYST_DIRECTIVE_RE = re.compile(
    r"^\s*(?:`{3,}|:{3,})\{(?P<name>(?:py:)?(?:module|currentmodule))\}\s*(?P<arg>\S+)?\s*$"
)
MYST_CONTAINER_RE = re.compile(
    r"^\s*:{3,}\{(?P<name>[\w:-]+)\}\s*(?P<arg>.*)$"
)
HEADING_UNDERLINE_RE = re.compile(r"^\s*([=\-~^\"#*+:`])\1{2,}\s*$")
GRID_TABLE_RE = re.compile(r"^\s*(?:\+[=+\-]+\+|\|.*\|)\s*$")
MARKDOWN_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)+\s*\|?\s*$")
OPTION_RE = re.compile(r"^\s*:[\w-]+:(?:\s|$)")
PROMPT_RE = re.compile(r"^\s*(?:>>>|\.\.\.)\s?")
SUBSTITUTION_RE = re.compile(r"^\s*\.\.\s+\|[^|]+\|\s+")
SUBSTITUTION_REF_RE = re.compile(r"\|(?P<name>[A-Za-z][\w-]*)\|")
JINJA_LINE_RE = re.compile(r"^\s*(?:\{\{.*\}\}|\{%.*%\}|\{#.*#\})\s*$")
HEADER_ASSIGNMENT_RE = re.compile(
    r"^\s*header\s*=\s*(?:[rubf]{0,2})?(?P<quote>'''|\"\"\")(?P<body>.*?)(?P=quote)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
CURRENTMODULE_RE = re.compile(r"^\s*\.\.\s+currentmodule::\s*(?P<module>\S+)\s*$", re.MULTILINE)
BADGE_SUBSTITUTION_RE = re.compile(
    r"^\s*\.\.\s+\|(?P<name>[^|\n]+)\|\s+replace::\s+"
    r":raw-html:`<span\s+class=(?P<quote>[\"'])badge(?:\s+[^\"']*)?(?P=quote)>"
    r"(?P<label>[^<\n]+)</span>`",
    re.IGNORECASE | re.MULTILINE,
)
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+] |(?:\d+|#)[.)]\s+)")
LIST_CONTINUATION_RE = re.compile(r"\b(?:and|or)$", re.IGNORECASE)
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
ABBREVIATION_RE = re.compile(r"\b(?:e\.g\.|i\.e\.|etc\.|vs\.|cf\.)", re.IGNORECASE)
MARKUP_RESIDUE_RE = re.compile(
    r"(?:`{3,}|:{1,2}[A-Za-z][\w:-]*:`|\{(?:py:)?[\w:-]+\}`|^\s*\.\.\s|"
    r"\|[A-Za-z][\w-]*\||\\[A-Za-z]+|</?[A-Za-z][^>\n]*>|\{[#%{]|[#%}]\}|"
    r"\[[^\]\n]+\]\(https?://|\b(?:starts with|as in|lines of|is given by)\s*$)",
    re.IGNORECASE | re.MULTILINE,
)

INLINE_RE = re.compile(
    r"(?P<double>``(?P<double_body>[^`\n]+)``)"
    r"|(?P<strong>\*\*(?P<strong_body>[^*\n]+)\*\*)"
    r"|(?P<emphasis>(?<!\*)\*(?P<emphasis_body>[^*\n]+)\*(?!\*))"
    r"|(?P<markdown_link>\[(?P<markdown_title>[^\]\n]+)\]\((?P<markdown_target>https?://[^)\n]+)\))"
    r"|(?P<rst_link>`(?P<link_title>[^`<>]+?)\s*<(?P<link_target>https?://[^`>]+)>`_{1,2})"
    r"|(?P<rst_role>:(?P<rst_role_name>(?:[\w-]+:)?[\w-]+):`(?P<rst_body>[^`\n]+)`)"
    r"|(?P<myst_role>\{(?P<myst_role_name>(?:[\w-]+:)?[\w-]+)\}`(?P<myst_body>[^`\n]+)`)"
    r"|(?P<single>`(?P<single_body>[^`\n]+)`)",
)
EXPLICIT_TITLE_RE = re.compile(r"^(?P<title>.+?)\s*<(?P<target>[^<>]+)>$")


@dataclass(frozen=True)
class Candidate:
    """One rendered API candidate with half-open character offsets."""

    char_start: int
    char_end: int
    role: str
    raw_target: str


@dataclass(frozen=True)
class LiteralSpan:
    """One rendered double-backtick literal used by the later R5 rule."""

    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class Paragraph:
    """One rendered documentation paragraph and its extraction metadata."""

    library: str
    doc_path: str
    source_type: str
    current_module: str | None
    text: str
    candidates: tuple[Candidate, ...]
    literal_spans: tuple[LiteralSpan, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        result = asdict(self)
        result["candidates"] = [asdict(candidate) for candidate in self.candidates]
        result["literal_spans"] = [asdict(span) for span in self.literal_spans]
        return result


@dataclass(frozen=True)
class ResolvedCandidate:
    """An API candidate annotated with its inventory resolution outcome."""

    char_start: int
    char_end: int
    role: str
    raw_target: str
    normalized_target: str
    status: str
    resolved_name: str | None
    entity_type: str | None
    inventory_roles: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedParagraph:
    """A paragraph whose candidates have been resolved against inventories."""

    paragraph: Paragraph
    candidates: tuple[ResolvedCandidate, ...]
    drop_unresolved: bool
    drop_ambiguous: bool


@dataclass(frozen=True)
class ResolvedSentence:
    """A rendered sentence with paragraph-relative metadata shifted locally."""

    library: str
    doc_path: str
    source_type: str
    current_module: str | None
    text: str
    candidates: tuple[ResolvedCandidate, ...]
    literal_spans: tuple[LiteralSpan, ...]


@dataclass(frozen=True)
class CleanedSentence:
    """A kept sentence represented with tokens, IOB2 tags, and entity spans."""

    library: str
    doc_path: str
    source_type: str
    text: str
    tokens: tuple[str, ...]
    tags: tuple[str, ...]
    spans: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class CleaningOutcome:
    """The result of applying the ordered R1-R7 cleaning rules."""

    sentence: ResolvedSentence
    drop_rule: str | None
    cleaned: CleanedSentence | None


@dataclass(frozen=True)
class BuildRecord:
    """A cleaned sentence with deterministic identity and assigned split."""

    id: str
    split: str
    cleaned: CleanedSentence

    def to_json(self) -> dict[str, object]:
        """Return the final JSONL record required by the data contract."""
        return {
            "id": self.id,
            "library": self.cleaned.library,
            "doc_path": self.cleaned.doc_path,
            "source_type": self.cleaned.source_type,
            "tokens": list(self.cleaned.tokens),
            "tags": list(self.cleaned.tags),
            "spans": list(self.cleaned.spans),
        }


def classify_source_type(doc_path: str) -> str:
    """Map a documentation path to one of the four fixed source types."""
    normalized = doc_path.replace("\\", "/").lower()
    parts = set(Path(normalized).parts)
    filename = Path(normalized).name
    if (
        {"whatsnew", "whats_new", "release", "releases", "release-notes", "release_notes"} & parts
        or "changelog" in filename
        or "whatsnew" in filename
        or "release-note" in normalized
    ):
        return "changelog"
    if {"tutorial", "tutorials", "examples"} & parts or "tutorial" in filename:
        return "tutorial"
    if (
        {"guide", "user", "user_guide", "getting_started", "howto", "install", "development"} & parts
        or "guide" in filename
    ):
        return "guide"
    return "reference"


def _split_explicit_title(body: str) -> tuple[str | None, str]:
    match = EXPLICIT_TITLE_RE.fullmatch(body.strip())
    if match:
        return match.group("title").strip(), match.group("target").strip()
    return None, body.strip()


def _visible_target(body: str, role: str) -> tuple[str, bool]:
    explicit_title, target = _split_explicit_title(body)
    if explicit_title is not None:
        return explicit_title, True
    display_target = target.lstrip("!")
    shortened = display_target.startswith("~")
    display_target = display_target.lstrip("~")
    display_target = display_target.lstrip(".")
    if shortened:
        display_target = display_target.rsplit(".", 1)[-1]
    local_role = role.split(":")[-1]
    if local_role in FUNCTION_ROLES and not display_target.endswith("()"):
        display_target += "()"
    return display_target, False


def _render_non_api(body: str) -> str:
    explicit_title, target = _split_explicit_title(body)
    if explicit_title is not None:
        return explicit_title
    return target.lstrip("!~.")


def parse_badge_substitutions(text: str) -> dict[str, str]:
    """Return raw-HTML badge substitutions and their visible labels."""
    return {
        match.group("name").strip(): html.unescape(re.sub(r"\s+", " ", match.group("label")).strip())
        for match in BADGE_SUBSTITUTION_RE.finditer(text)
    }


def render_inline(
    text: str,
    default_role: str | None,
    substitutions: dict[str, str] | None = None,
) -> tuple[str, tuple[Candidate, ...], tuple[LiteralSpan, ...]]:
    """Render inline reST/MyST markup and track candidate offsets."""
    output: list[str] = []
    candidates: list[Candidate] = []
    literal_spans: list[LiteralSpan] = []
    output_length = 0
    cursor = 0

    def append(value: str) -> tuple[int, int]:
        nonlocal output_length
        start = output_length
        output.append(value)
        output_length += len(value)
        return start, output_length

    escaped_backtick = "\ue000"
    if substitutions:
        text = SUBSTITUTION_REF_RE.sub(
            lambda match: substitutions.get(match.group("name"), match.group(0)),
            text,
        )
    safe_text = text.replace("\\`", escaped_backtick)

    def render_plain(value: str) -> str:
        return (
            value.replace(escaped_backtick, "`")
            .replace("\\ ", "")
            .replace("\\.", ".")
            .replace("\\,", ",")
        )

    for match in INLINE_RE.finditer(safe_text):
        append(render_plain(safe_text[cursor : match.start()]))
        cursor = match.end()

        if match.group("double"):
            visible = match.group("double_body")
            start, end = append(visible)
            literal_spans.append(LiteralSpan(start, end, visible))
            continue
        if match.group("strong"):
            append(match.group("strong_body"))
            continue
        if match.group("emphasis"):
            append(match.group("emphasis_body"))
            continue
        if match.group("markdown_link"):
            append(render_plain(match.group("markdown_title").strip()))
            continue
        if match.group("rst_link"):
            append(render_plain(match.group("link_title").strip()))
            continue

        role: str | None = None
        body: str
        if match.group("rst_role"):
            role = match.group("rst_role_name")
            body = match.group("rst_body")
        elif match.group("myst_role"):
            role = match.group("myst_role_name")
            body = match.group("myst_body")
        else:
            body = match.group("single_body")
            if default_role in DEFAULT_API_ROLES:
                role = "obj"

        local_role = role.split(":")[-1] if role else None
        is_python_role = role is not None and (":" not in role or role.startswith("py:"))
        if local_role in API_ROLES and is_python_role:
            visible, _ = _visible_target(body, role or "obj")
            start, end = append(visible)
            if visible:
                candidates.append(Candidate(start, end, role or "obj", body.strip()))
        else:
            append(_render_non_api(body))

    append(render_plain(safe_text[cursor:]))
    return "".join(output), tuple(candidates), tuple(literal_spans)


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_heading(lines: Sequence[str], index: int) -> bool:
    if index + 1 >= len(lines) or not lines[index].strip():
        return False
    next_line = lines[index + 1].strip()
    return next_line != ":::" and bool(HEADING_UNDERLINE_RE.fullmatch(lines[index + 1]))


def _strip_list_marker(line: str) -> str:
    return LIST_MARKER_RE.sub("", line, count=1).strip()


def iter_raw_paragraphs(
    text: str,
    header_current_module: str | None = None,
) -> Iterator[tuple[str | None, str]]:
    """Yield ``(current_module, markup_paragraph)`` while skipping non-prose blocks."""
    lines = text.splitlines()
    current_module: str | None = None
    paragraph: list[str] = []
    paragraph_module: str | None = None
    skip_indent: int | None = None
    in_fence: str | None = None
    in_myst_container = False
    awaiting_literal = False
    index = 0

    def flush() -> tuple[str | None, str] | None:
        nonlocal paragraph, paragraph_module, awaiting_literal
        if not paragraph:
            return None
        joined = " ".join(part.strip() for part in paragraph if part.strip())
        paragraph = []
        module = paragraph_module
        paragraph_module = None
        if joined.endswith("::"):
            joined = joined[:-1]
            awaiting_literal = True
        return (module, joined) if joined else None

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        indent = _leading_spaces(line)

        if in_myst_container and stripped == ":::":
            pending = flush()
            if pending:
                yield pending
            in_myst_container = False
            index += 1
            continue

        if in_fence is not None:
            if stripped.startswith(in_fence):
                in_fence = None
            index += 1
            continue

        if skip_indent is not None:
            if not stripped or indent > skip_indent:
                index += 1
                continue
            skip_indent = None

        if awaiting_literal:
            if not stripped:
                index += 1
                continue
            awaiting_literal = False
            if indent > 0:
                skip_indent = 0
                index += 1
                continue

        if JINJA_LINE_RE.fullmatch(line):
            pending = flush()
            if pending:
                yield pending
            if stripped == "{{ header }}":
                current_module = header_current_module
            index += 1
            continue

        myst_directive = MYST_DIRECTIVE_RE.match(line)
        if myst_directive:
            pending = flush()
            if pending:
                yield pending
            argument = myst_directive.group("arg")
            if argument:
                current_module = argument.strip()
            marker = stripped[:3]
            in_fence = marker
            index += 1
            continue

        myst_container = MYST_CONTAINER_RE.match(line)
        if myst_container:
            pending = flush()
            if pending:
                yield pending
            local_name = myst_container.group("name").split(":")[-1].lower()
            if local_name in SKIPPED_DIRECTIVES:
                in_fence = ":::"
            else:
                in_myst_container = True
            index += 1
            continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            pending = flush()
            if pending:
                yield pending
            in_fence = stripped[:3]
            index += 1
            continue

        if _is_heading(lines, index):
            pending = flush()
            if pending:
                yield pending
            index += 2
            continue
        if MARKDOWN_HEADING_RE.match(line):
            pending = flush()
            if pending:
                yield pending
            index += 1
            continue
        if re.fullmatch(r"\([^)]+\)=", stripped):
            pending = flush()
            if pending:
                yield pending
            index += 1
            continue
        if HEADING_UNDERLINE_RE.fullmatch(line):
            index += 1
            continue

        directive = DIRECTIVE_RE.match(line)
        if directive:
            pending = flush()
            if pending:
                yield pending
            name = directive.group("name").lower()
            argument = directive.group("arg").strip()
            local_name = name.split(":")[-1]
            if local_name in {"module", "currentmodule"}:
                current_module = argument or None
            if local_name in SKIPPED_DIRECTIVES or local_name in {"module", "currentmodule"}:
                skip_indent = len(directive.group("indent"))
            index += 1
            continue

        if (
            SUBSTITUTION_RE.match(line)
            or stripped.startswith(".. ")
            or OPTION_RE.match(line)
            or PROMPT_RE.match(line)
        ):
            pending = flush()
            if pending:
                yield pending
            if stripped.startswith(".. "):
                skip_indent = indent
            index += 1
            continue

        if GRID_TABLE_RE.fullmatch(line) or (
            index + 1 < len(lines) and "|" in line and MARKDOWN_TABLE_RULE_RE.fullmatch(lines[index + 1])
        ):
            pending = flush()
            if pending:
                yield pending
            index += 1
            while index < len(lines) and ("|" in lines[index] or GRID_TABLE_RE.fullmatch(lines[index])):
                index += 1
            continue

        if not stripped:
            pending = flush()
            if pending:
                yield pending
            index += 1
            continue

        if (
            LIST_MARKER_RE.match(line)
            and paragraph
            and not LIST_CONTINUATION_RE.search(paragraph[-1].rstrip())
        ):
            pending = flush()
            if pending:
                yield pending
        content = _strip_list_marker(line)
        if content:
            if not paragraph:
                paragraph_module = current_module
            paragraph.append(content)
        index += 1

    pending = flush()
    if pending:
        yield pending


def extract_document(
    library: str,
    doc_path: str,
    text: str,
    default_role: str | None,
    *,
    header_current_module: str | None = None,
    substitutions: dict[str, str] | None = None,
) -> list[Paragraph]:
    """Extract rendered paragraphs and candidates from one documentation file."""
    if any(line.strip() == "{{ header }}" for line in text.splitlines()) and not header_current_module:
        raise ValueError(f"{library}/{doc_path} uses {{{{ header }}}} without a header currentmodule")
    paragraphs: list[Paragraph] = []
    source_type = classify_source_type(doc_path)
    for current_module, markup in iter_raw_paragraphs(text, header_current_module):
        markup = re.sub(r"\s+", " ", markup).strip()
        rendered, candidates, literal_spans = render_inline(markup, default_role, substitutions)
        if not rendered:
            continue
        # Whitespace is already normalized before rendering because source lines
        # are joined with one space; assert candidate offsets remain trustworthy.
        if any(rendered[c.char_start : c.char_end] == "" for c in candidates):
            raise ValueError(f"empty rendered candidate in {library}/{doc_path}")
        paragraphs.append(
            Paragraph(
                library,
                doc_path,
                source_type,
                current_module,
                rendered,
                candidates,
                literal_spans,
            )
        )
    return paragraphs


def load_sources(path: Path = SOURCES_PATH) -> list[dict[str, object]]:
    """Load pinned source metadata, excluding the inventory-only Python entry."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [source for source in payload["sources"] if source["docs_subdir"] is not None]


def load_inventory_index(inventory_dir: Path = RAW_DIR / "inventories") -> dict[str, set[str]]:
    """Load all Python-domain objects into a full-name to role-set index."""
    try:
        import sphobjinv
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError("stage resolve requires sphobjinv") from error

    index: dict[str, set[str]] = defaultdict(set)
    for path in sorted(inventory_dir.glob("*.inv")):
        inventory = sphobjinv.Inventory(str(path))
        for obj in inventory.objects:
            if obj.domain == "py":
                index[obj.name].add(obj.role)
    return dict(index)


def normalize_candidate_target(raw_target: str) -> str:
    """Remove display markup while retaining a leading-dot suffix request."""
    _, target = _split_explicit_title(raw_target)
    while target.startswith(("!", "~")):
        target = target[1:]
    return target.strip()


def _resolution_from_roles(
    candidate: Candidate,
    normalized_target: str,
    resolved_name: str | None,
    roles: set[str],
    forced_ambiguous: bool = False,
) -> ResolvedCandidate:
    entity_types = {INVENTORY_ROLE_TO_TYPE[role] for role in roles if role in INVENTORY_ROLE_TO_TYPE}
    if forced_ambiguous or len(entity_types) > 1:
        status = "ambiguous"
        entity_type = None
    elif len(entity_types) == 1:
        status = "resolved"
        entity_type = next(iter(entity_types))
    else:
        status = "unresolved"
        entity_type = None
    return ResolvedCandidate(
        candidate.char_start,
        candidate.char_end,
        candidate.role,
        candidate.raw_target,
        normalized_target,
        status,
        resolved_name,
        entity_type,
        tuple(sorted(roles)),
    )


def resolve_candidate(
    candidate: Candidate,
    current_module: str | None,
    inventory_index: dict[str, set[str]],
) -> ResolvedCandidate:
    """Resolve one candidate using exact, current-module, then unique suffix lookup."""
    target = normalize_candidate_target(candidate.raw_target)
    if not target:
        return _resolution_from_roles(candidate, target, None, set())

    if not target.startswith(".") and target in inventory_index:
        return _resolution_from_roles(candidate, target, target, inventory_index[target])

    if not target.startswith(".") and current_module:
        qualified = f"{current_module}.{target}"
        if qualified in inventory_index:
            return _resolution_from_roles(candidate, target, qualified, inventory_index[qualified])

    if target.startswith("."):
        suffix = target[1:]
        matching_names = sorted(
            name for name in inventory_index if name == suffix or name.endswith(f".{suffix}")
        )
        if len(matching_names) == 1:
            name = matching_names[0]
            return _resolution_from_roles(candidate, target, name, inventory_index[name])
        if len(matching_names) > 1:
            roles = set().union(*(inventory_index[name] for name in matching_names))
            return _resolution_from_roles(candidate, target, None, roles, forced_ambiguous=True)

    return _resolution_from_roles(candidate, target, None, set())


def resolve_paragraph(paragraph: Paragraph, inventory_index: dict[str, set[str]]) -> ResolvedParagraph:
    """Resolve every candidate and set later sentence-drop flags."""
    candidates = tuple(
        resolve_candidate(candidate, paragraph.current_module, inventory_index)
        for candidate in paragraph.candidates
    )
    return ResolvedParagraph(
        paragraph,
        candidates,
        any(candidate.status == "unresolved" for candidate in candidates),
        any(candidate.status == "ambiguous" for candidate in candidates),
    )


def split_sentences(paragraph: ResolvedParagraph) -> list[ResolvedSentence]:
    """Split rendered prose while masking entity dots and common abbreviations."""
    text = paragraph.paragraph.text
    masked = list(text)
    for candidate in paragraph.candidates:
        for index in range(candidate.char_start, candidate.char_end):
            if not masked[index].isspace():
                masked[index] = "X"
    masked_text = "".join(masked)
    for match in ABBREVIATION_RE.finditer(masked_text):
        for index in range(match.start(), match.end()):
            if masked[index] == ".":
                masked[index] = "~"
    masked_text = "".join(masked)

    ranges: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_BREAK_RE.finditer(masked_text):
        ranges.append((start, match.start()))
        start = match.end()
    ranges.append((start, len(text)))

    sentences: list[ResolvedSentence] = []
    for raw_start, raw_end in ranges:
        start = raw_start
        end = raw_end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        candidates: list[ResolvedCandidate] = []
        for candidate in paragraph.candidates:
            if candidate.char_start >= start and candidate.char_end <= end:
                candidates.append(
                    ResolvedCandidate(
                        candidate.char_start - start,
                        candidate.char_end - start,
                        candidate.role,
                        candidate.raw_target,
                        candidate.normalized_target,
                        candidate.status,
                        candidate.resolved_name,
                        candidate.entity_type,
                        candidate.inventory_roles,
                    )
                )
            elif candidate.char_start < end and candidate.char_end > start:
                raise ValueError("sentence boundary crossed a masked entity")
        literals = tuple(
            LiteralSpan(span.char_start - start, span.char_end - start, span.text)
            for span in paragraph.paragraph.literal_spans
            if span.char_start >= start and span.char_end <= end
        )
        sentences.append(
            ResolvedSentence(
                paragraph.paragraph.library,
                paragraph.paragraph.doc_path,
                paragraph.paragraph.source_type,
                paragraph.paragraph.current_module,
                text[start:end],
                tuple(candidates),
                literals,
            )
        )
    return sentences


def has_markup_residue(text: str) -> bool:
    """Return whether rendered text still contains obvious reST/MyST residue."""
    return bool(MARKUP_RESIDUE_RE.search(text)) or text.count("`") % 2 == 1 or ">>>" in text


def _candidate_token_span(
    candidate: ResolvedCandidate,
    token_offsets: Sequence[tuple[str, int, int]],
) -> tuple[int, int] | None:
    covered = [
        index
        for index, (_, start, end) in enumerate(token_offsets)
        if start >= candidate.char_start and end <= candidate.char_end
    ]
    if not covered:
        return None
    start_index = covered[0]
    end_index = covered[-1] + 1
    if token_offsets[start_index][1] != candidate.char_start:
        return None
    if token_offsets[end_index - 1][2] != candidate.char_end:
        return None
    return start_index, end_index


def _literal_is_api(
    literal: LiteralSpan,
    inventory_names: set[str],
    unique_last_components: set[str],
) -> bool:
    value = literal.text.strip()
    if value.endswith("()"):
        value = value[:-2]
    return value in inventory_names or value in unique_last_components


def clean_sentence(
    sentence: ResolvedSentence,
    inventory_names: set[str],
    unique_last_components: set[str],
) -> CleaningOutcome:
    """Apply R1-R7 in the mandated order and build IOB2 output when kept."""
    if has_markup_residue(sentence.text):
        return CleaningOutcome(sentence, "R1", None)
    if any(candidate.status == "unresolved" for candidate in sentence.candidates):
        return CleaningOutcome(sentence, "R2", None)
    if any(candidate.status == "ambiguous" for candidate in sentence.candidates):
        return CleaningOutcome(sentence, "R3", None)

    token_offsets = tokenize(sentence.text)
    span_records: list[dict[str, object]] = []
    occupied: set[int] = set()
    for candidate in sentence.candidates:
        token_span = _candidate_token_span(candidate, token_offsets)
        if token_span is None or candidate.entity_type is None or candidate.resolved_name is None:
            return CleaningOutcome(sentence, "R4", None)
        start, end = token_span
        if occupied.intersection(range(start, end)):
            return CleaningOutcome(sentence, "R4", None)
        occupied.update(range(start, end))
        span_records.append(
            {
                "start": start,
                "end": end,
                "type": candidate.entity_type,
                "target": candidate.resolved_name,
                "role": candidate.role,
            }
        )

    if any(
        _literal_is_api(literal, inventory_names, unique_last_components)
        for literal in sentence.literal_spans
    ):
        return CleaningOutcome(sentence, "R5", None)
    if len(token_offsets) < 5 or len(token_offsets) > 60:
        return CleaningOutcome(sentence, "R6", None)
    non_space = sum(not character.isspace() for character in sentence.text)
    alphabetic = sum(character.isalpha() for character in sentence.text)
    if non_space == 0 or alphabetic / non_space < 0.5:
        return CleaningOutcome(sentence, "R7", None)

    span_records.sort(key=lambda span: int(span["start"]))
    tags = spans_to_tags(len(token_offsets), span_records)
    cleaned = CleanedSentence(
        sentence.library,
        sentence.doc_path,
        sentence.source_type,
        sentence.text,
        tuple(token for token, _, _ in token_offsets),
        tuple(tags),
        tuple(span_records),
    )
    return CleaningOutcome(sentence, None, cleaned)


def collect_cleaning_outcomes() -> tuple[list[tuple[str, CleaningOutcome]], dict[str, set[str]]]:
    """Run extraction, resolution, splitting, and ordered cleaning in source order."""
    paragraphs, _ = extract_all()
    inventory_index = load_inventory_index()
    inventory_names = set(inventory_index)
    last_components = Counter(name.rsplit(".", 1)[-1] for name in inventory_index)
    unique_last_components = {name for name, count in last_components.items() if count == 1}
    sentence_numbers: Counter[tuple[str, str]] = Counter()
    outcomes: list[tuple[str, CleaningOutcome]] = []
    for paragraph in paragraphs:
        resolved = resolve_paragraph(paragraph, inventory_index)
        for sentence in split_sentences(resolved):
            key = (sentence.library, sentence.doc_path)
            number = sentence_numbers[key]
            sentence_numbers[key] += 1
            sentence_id = f"{sentence.library}/{sentence.doc_path}#s{number:05d}"
            outcomes.append(
                (
                    sentence_id,
                    clean_sentence(sentence, inventory_names, unique_last_components),
                )
            )
    return outcomes, inventory_index


def seeded_score(value: str, seed: int = 42) -> int:
    """Return a deterministic integer score independent of Python hash randomization."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest(), "big")


def assign_split(cleaned: CleanedSentence, seed: int = 42) -> str:
    """Assign whole documents to heldout or deterministic 80/10/10 splits."""
    if cleaned.library == "pytorch":
        return "heldout"
    maximum = 1 << 256
    score = seeded_score(f"{cleaned.library}:{cleaned.doc_path}", seed) / maximum
    if score < 0.8:
        return "train"
    if score < 0.9:
        return "dev"
    return "test"


def normalize_duplicate_text(text: str) -> str:
    """Normalize whitespace for exact cross-corpus duplicate detection."""
    return " ".join(text.split())


def deduplicate_records(records: Sequence[BuildRecord]) -> tuple[list[BuildRecord], Counter[str]]:
    """Apply R8 with heldout > test > dev > train priority."""
    priority = {"heldout": 3, "test": 2, "dev": 1, "train": 0}
    groups: dict[str, list[BuildRecord]] = defaultdict(list)
    for record in records:
        groups[normalize_duplicate_text(record.cleaned.text)].append(record)

    kept: list[BuildRecord] = []
    dropped_by_library: Counter[str] = Counter()
    for normalized in sorted(groups):
        group = groups[normalized]
        winner = min(group, key=lambda record: (-priority[record.split], record.id))
        kept.append(winner)
        for record in group:
            if record is not winner:
                dropped_by_library[record.cleaned.library] += 1
    return kept, dropped_by_library


def sample_zero_entity_records(
    records: Sequence[BuildRecord], seed: int = 42
) -> tuple[list[BuildRecord], Counter[str]]:
    """Keep zero-entity sentences at approximately 20% of each library final set."""
    by_library: dict[str, list[BuildRecord]] = defaultdict(list)
    for record in records:
        by_library[record.cleaned.library].append(record)

    selected: list[BuildRecord] = []
    dropped: Counter[str] = Counter()
    for library in sorted(by_library):
        entity_records = [record for record in by_library[library] if record.cleaned.spans]
        zero_records = [record for record in by_library[library] if not record.cleaned.spans]
        target_zero = min(len(zero_records), (len(entity_records) + 2) // 4)
        zero_records.sort(key=lambda record: (seeded_score(record.id, seed), record.id))
        selected.extend(entity_records)
        selected.extend(zero_records[:target_zero])
        dropped[library] = len(zero_records) - target_zero
    return selected, dropped


def make_test_noparen(record: BuildRecord) -> BuildRecord:
    """Remove terminal parentheses from FUNCTION/METHOD spans and realign tags."""
    tokens = list(record.cleaned.tokens)
    removed: set[int] = set()
    for span in record.cleaned.spans:
        start = int(span["start"])
        end = int(span["end"])
        if span["type"] in {"FUNCTION", "METHOD"} and end - start >= 2:
            if tokens[end - 2 : end] == ["(", ")"]:
                removed.update({end - 2, end - 1})

    new_tokens = [token for index, token in enumerate(tokens) if index not in removed]
    new_spans: list[dict[str, object]] = []
    for span in record.cleaned.spans:
        start = int(span["start"])
        end = int(span["end"])
        new_span = dict(span)
        new_span["start"] = sum(index not in removed for index in range(start))
        new_span["end"] = sum(index not in removed for index in range(end))
        new_spans.append(new_span)
    new_tags = spans_to_tags(len(new_tokens), new_spans)
    cleaned = CleanedSentence(
        record.cleaned.library,
        record.cleaned.doc_path,
        record.cleaned.source_type,
        record.cleaned.text,
        tuple(new_tokens),
        tuple(new_tags),
        tuple(new_spans),
    )
    return BuildRecord(record.id, "test_noparen", cleaned)


def prepare_records() -> tuple[list[BuildRecord], dict[str, object]]:
    """Create final records and diagnostics before serialization."""
    outcomes, _ = collect_cleaning_outcomes()
    drop_by_library: dict[str, Counter[str]] = defaultdict(Counter)
    records: list[BuildRecord] = []
    for sentence_id, outcome in outcomes:
        if outcome.drop_rule:
            drop_by_library[outcome.sentence.library][outcome.drop_rule] += 1
        elif outcome.cleaned is not None:
            records.append(BuildRecord(sentence_id, assign_split(outcome.cleaned), outcome.cleaned))

    records, r8_drops = deduplicate_records(records)
    for library, count in r8_drops.items():
        drop_by_library[library]["R8"] += count
    records, zero_drops = sample_zero_entity_records(records)
    records.sort(key=lambda record: (record.split, record.id))
    diagnostics: dict[str, object] = {
        "drop_by_library": drop_by_library,
        "zero_entity_unsampled": zero_drops,
    }
    return records, diagnostics


def count_records(records: Sequence[BuildRecord]) -> dict[str, int]:
    """Count sentences, tokens, and entities in a record group."""
    return {
        "sentences": len(records),
        "tokens": sum(len(record.cleaned.tokens) for record in records),
        "entities": sum(len(record.cleaned.spans) for record in records),
    }


def build_stats(records: Sequence[BuildRecord], diagnostics: dict[str, object]) -> dict[str, object]:
    """Build the traceable statistics artifact from final records."""
    base_records = list(records)
    test_records = [record for record in base_records if record.split == "test"]
    noparen_records = [make_test_noparen(record) for record in test_records]
    all_records = base_records + noparen_records

    split_stats = {
        split: count_records([record for record in all_records if record.split == split])
        for split in ("train", "dev", "test", "heldout", "test_noparen")
    }
    library_stats = {
        library: count_records([record for record in base_records if record.cleaned.library == library])
        for library in sorted({record.cleaned.library for record in base_records})
    }
    source_type_stats = {
        source_type: count_records(
            [record for record in base_records if record.cleaned.source_type == source_type]
        )
        for source_type in sorted({record.cleaned.source_type for record in base_records})
    }
    entity_types: Counter[str] = Counter()
    entity_lengths: Counter[int] = Counter()
    for record in base_records:
        for span in record.cleaned.spans:
            entity_types[str(span["type"])] += 1
            entity_lengths[int(span["end"]) - int(span["start"])] += 1

    train_targets = {
        str(span["target"])
        for record in base_records
        if record.split == "train"
        for span in record.cleaned.spans
    }
    test_targets = [
        str(span["target"])
        for record in base_records
        if record.split == "test"
        for span in record.cleaned.spans
    ]
    unseen_count = sum(target not in train_targets for target in test_targets)

    doc_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in base_records:
        doc_splits[(record.cleaned.library, record.cleaned.doc_path)].add(record.split)
    shared_doc_paths = sum(len(splits) > 1 for splits in doc_splits.values())

    drop_by_library = diagnostics["drop_by_library"]
    rules = [f"R{number}" for number in range(1, 9)]
    drop_table = {
        "by_library": {
            library: {rule: int(counts[rule]) for rule in rules}
            for library, counts in sorted(drop_by_library.items())
        },
        "by_rule": {
            rule: sum(int(counts[rule]) for counts in drop_by_library.values()) for rule in rules
        },
    }
    return {
        "schema_version": 1,
        "seed": 42,
        "splits": split_stats,
        "libraries": library_stats,
        "source_types": source_type_stats,
        "entity_types": dict(sorted(entity_types.items())),
        "entity_length_histogram": {
            str(length): count for length, count in sorted(entity_lengths.items())
        },
        "drop_table": drop_table,
        "zero_entity_unsampled": {
            library: int(count)
            for library, count in sorted(diagnostics["zero_entity_unsampled"].items())
        },
        "test_unseen_target_mentions": unseen_count,
        "test_target_mentions": len(test_targets),
        "test_unseen_target_share": unseen_count / len(test_targets) if test_targets else 0.0,
        "shared_doc_paths": shared_doc_paths,
    }


def iter_doc_files(source: dict[str, object]) -> Iterator[Path]:
    """Yield allowed source text files for one pinned library."""
    library_root = RAW_DIR / str(source["library"])
    docs_root = library_root / str(source["docs_subdir"])
    yield from sorted(
        (path for path in docs_root.rglob("*") if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES),
        key=lambda path: path.relative_to(library_root).as_posix(),
    )


def read_source_text(path: Path) -> str:
    """Decode official docs without silently introducing replacement glyphs."""
    return decode_source_bytes(path.read_bytes())


def decode_source_bytes(payload: bytes) -> str:
    """Decode source bytes as UTF-8, then the common legacy Windows encoding."""
    for encoding in ("utf-8", "cp1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def load_header_current_module(source: dict[str, object]) -> str | None:
    """Read the current module from a library's configured header string."""
    conf_path = source.get("conf_path")
    if not isinstance(conf_path, str):
        return None
    library = str(source["library"])
    conf_text = read_source_text(RAW_DIR / library / conf_path)
    header_match = HEADER_ASSIGNMENT_RE.search(conf_text)
    if not header_match:
        return None
    module_match = CURRENTMODULE_RE.search(header_match.group("body"))
    return module_match.group("module") if module_match else None


def extract_all() -> tuple[list[Paragraph], list[tuple[str, str, str]]]:
    """Extract every pinned documentation file in stable order."""
    paragraphs: list[Paragraph] = []
    mappings: list[tuple[str, str, str]] = []
    for source in load_sources():
        library = str(source["library"])
        default_role = source.get("default_role")
        library_root = RAW_DIR / library
        documents = [(path, read_source_text(path)) for path in iter_doc_files(source)]
        substitutions: dict[str, str] = {}
        for path, text in documents:
            for name, label in parse_badge_substitutions(text).items():
                previous = substitutions.get(name)
                if previous is not None and previous != label:
                    doc_path = path.relative_to(library_root).as_posix()
                    raise ValueError(f"conflicting badge substitution {name!r} in {library}/{doc_path}")
                substitutions[name] = label
        header_current_module = load_header_current_module(source)
        for path, text in documents:
            doc_path = path.relative_to(library_root).as_posix()
            source_type = classify_source_type(doc_path)
            mappings.append((library, doc_path, source_type))
            paragraphs.extend(
                extract_document(
                    library,
                    doc_path,
                    text,
                    default_role if isinstance(default_role, str) else None,
                    header_current_module=header_current_module,
                    substitutions=substitutions,
                )
            )
    return paragraphs, mappings


def print_mapping_review(mappings: Sequence[tuple[str, str, str]], show_all: bool) -> None:
    """Print path classification counts and representative mappings."""
    print("\nPath -> source_type mapping review")
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for library, path, source_type in mappings:
        grouped[(library, source_type)].append(path)
    for (library, source_type), paths in sorted(grouped.items()):
        print(f"{library:<15} {source_type:<10} {len(paths):>5} files")
        selected = paths if show_all else paths[:3]
        for path in selected:
            print(f"  {path} -> {source_type}")
        if not show_all and len(paths) > len(selected):
            print(f"  ... {len(paths) - len(selected)} more")


def run_extract(sample_size: int, seed: int, show_all_mappings: bool) -> None:
    """Run stage 1 and print its review outputs without writing intermediates."""
    paragraphs, mappings = extract_all()
    print_mapping_review(mappings, show_all_mappings)
    counts = Counter(paragraph.source_type for paragraph in paragraphs)
    candidate_count = sum(len(paragraph.candidates) for paragraph in paragraphs)
    print("\nExtract summary")
    print(f"files={len(mappings)} paragraphs={len(paragraphs)} candidates={candidate_count}")
    print("source_types=" + json.dumps(dict(sorted(counts.items())), sort_keys=True))

    rng = random.Random(seed)
    by_library: dict[str, list[Paragraph]] = defaultdict(list)
    for paragraph in paragraphs:
        by_library[paragraph.library].append(paragraph)
    sample: list[Paragraph] = []
    libraries = sorted(by_library)
    while len(sample) < min(sample_size, len(paragraphs)):
        progressed = False
        for library in libraries:
            choices = by_library[library]
            if choices and len(sample) < sample_size:
                sample.append(choices.pop(rng.randrange(len(choices))))
                progressed = True
        if not progressed:
            break
    print(f"\nSeeded paragraph sample (seed={seed}, n={len(sample)})")
    for paragraph in sample:
        print(json.dumps(paragraph.to_dict(), ensure_ascii=True, sort_keys=True))


def run_resolve() -> None:
    """Run stage 2 and print resolution diagnostics without writing intermediates."""
    paragraphs, _ = extract_all()
    inventory_index = load_inventory_index()
    resolved = [resolve_paragraph(paragraph, inventory_index) for paragraph in paragraphs]

    print(f"Inventory index: {len(inventory_index)} unique Python object names")
    by_library: dict[str, Counter[str]] = defaultdict(Counter)
    unresolved_targets: dict[str, Counter[str]] = defaultdict(Counter)
    disagreements: Counter[tuple[str, str, str, str]] = Counter()
    for item in resolved:
        library = item.paragraph.library
        for candidate in item.candidates:
            by_library[library][candidate.status] += 1
            if candidate.status == "unresolved":
                unresolved_targets[library][candidate.normalized_target] += 1
            expected_type = CANDIDATE_ROLE_TO_TYPE.get(candidate.role.split(":")[-1])
            if candidate.status == "resolved" and expected_type and expected_type != candidate.entity_type:
                disagreements[(library, candidate.role, expected_type, candidate.entity_type or "")] += 1

    print("\nResolution counts")
    print(f"{'library':<15} {'resolved':>10} {'unresolved':>12} {'ambiguous':>11}")
    for library in sorted(by_library):
        counts = by_library[library]
        print(
            f"{library:<15} {counts['resolved']:>10} {counts['unresolved']:>12} "
            f"{counts['ambiguous']:>11}"
        )

    print("\nRole vs inventory disagreements")
    if disagreements:
        for (library, role, expected, actual), count in disagreements.most_common():
            print(f"{library}: {role} -> {expected}, inventory -> {actual}: {count}")
    else:
        print("none")

    print("\nTop unresolved targets per library")
    for library in sorted(unresolved_targets):
        print(f"[{library}]")
        for target, count in unresolved_targets[library].most_common(20):
            print(f"  {count:>5}  {target}")


def run_clean() -> None:
    """Run stage 3 and print the ordered R1-R7 drop diagnostics."""
    paragraphs, _ = extract_all()
    inventory_index = load_inventory_index()
    inventory_names = set(inventory_index)
    last_components = Counter(name.rsplit(".", 1)[-1] for name in inventory_index)
    unique_last_components = {name for name, count in last_components.items() if count == 1}
    outcomes: list[CleaningOutcome] = []
    for paragraph in paragraphs:
        resolved = resolve_paragraph(paragraph, inventory_index)
        for sentence in split_sentences(resolved):
            outcomes.append(clean_sentence(sentence, inventory_names, unique_last_components))

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[CleaningOutcome]] = defaultdict(list)
    for outcome in outcomes:
        rule = outcome.drop_rule or "KEPT"
        counts[outcome.sentence.library][rule] += 1
        if outcome.drop_rule and len(examples[outcome.drop_rule]) < 10:
            examples[outcome.drop_rule].append(outcome)

    rules = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "KEPT"]
    print("Drop table (first matching rule only)")
    print(f"{'library':<15}" + "".join(f"{rule:>9}" for rule in rules))
    for library in sorted(counts):
        print(f"{library:<15}" + "".join(f"{counts[library][rule]:>9}" for rule in rules))
    print(f"{'TOTAL':<15}" + "".join(f"{sum(row[rule] for row in counts.values()):>9}" for rule in rules))

    for rule in rules[:-1]:
        print(f"\n{rule} examples ({len(examples[rule])})")
        for outcome in examples[rule]:
            sentence = outcome.sentence
            print(f"[{sentence.library}] {sentence.doc_path}: {sentence.text}")


def write_processed(output_dir: Path) -> dict[str, object]:
    """Build and validate all final JSONL splits plus stats.json."""
    records, diagnostics = prepare_records()
    by_split: dict[str, list[BuildRecord]] = defaultdict(list)
    for record in records:
        by_split[record.split].append(record)
    by_split["test_noparen"] = [make_test_noparen(record) for record in by_split["test"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test", "heldout", "test_noparen"):
        path = output_dir / f"{split}.jsonl"
        lines: list[str] = []
        seen_ids: set[str] = set()
        for record in sorted(by_split[split], key=lambda item: item.id):
            payload = record.to_json()
            validate(payload)
            if record.id in seen_ids:
                raise ValueError(f"duplicate id in {split}: {record.id}")
            seen_ids.add(record.id)
            lines.append(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")

    stats = build_stats(records, diagnostics)
    if stats["shared_doc_paths"] != 0:
        raise ValueError(f"document leakage detected: {stats['shared_doc_paths']} shared paths")
    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return stats


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one generated output file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_processed(expected_dir: Path) -> None:
    """Rebuild into a temporary directory and compare every output byte-for-byte."""
    if not expected_dir.is_dir():
        raise FileNotFoundError(f"processed dataset does not exist: {expected_dir}")
    with tempfile.TemporaryDirectory(prefix=".apilens-check-", dir=ROOT) as temporary:
        rebuilt_dir = Path(temporary) / "processed"
        write_processed(rebuilt_dir)
        expected_files = {
            path.name: file_sha256(path) for path in expected_dir.iterdir() if path.is_file()
        }
        rebuilt_files = {
            path.name: file_sha256(path) for path in rebuilt_dir.iterdir() if path.is_file()
        }
        if expected_files != rebuilt_files:
            missing = sorted(set(expected_files) - set(rebuilt_files))
            extra = sorted(set(rebuilt_files) - set(expected_files))
            changed = sorted(
                name
                for name in set(expected_files) & set(rebuilt_files)
                if expected_files[name] != rebuilt_files[name]
            )
            raise ValueError(
                f"determinism check failed: missing={missing}, extra={extra}, changed={changed}"
            )
    print(f"Determinism check passed for {len(expected_files)} files (byte-for-byte SHA-256 match).")


def run_write() -> None:
    """Write the complete processed dataset and show its authoritative statistics."""
    stats = write_processed(ROOT / "data" / "processed")
    print(json.dumps(stats, indent=2, sort_keys=True))


def main() -> None:
    """Run a selected deterministic dataset-building stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["extract", "resolve", "clean"])
    parser.add_argument("--check", action="store_true", help="rebuild and compare all processed outputs")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-all-mappings", action="store_true")
    args = parser.parse_args()
    if args.sample < 0:
        parser.error("--sample must be non-negative")
    if args.check and args.stage:
        parser.error("--check cannot be combined with --stage")
    if args.check:
        check_processed(ROOT / "data" / "processed")
    elif args.stage == "extract":
        run_extract(args.sample, args.seed, args.show_all_mappings)
    elif args.stage == "resolve":
        run_resolve()
    elif args.stage == "clean":
        run_clean()
    else:
        run_write()


if __name__ == "__main__":
    main()
