"""The engine constants with their comments and sources, for `config.json` (SCHEMA.md 2.9).

    parse(source, values)   sections of fields from the text of a Config class
    document(cfg)           the config.json object of engine/config.py and Config.default()

The values come from `Config.default().as_dict()`, so they cannot drift from the engine. The
comments come from the source text. A dataclass field that the parser does not find in the text
is an error. A field in the text that the dataclass does not have is an error too.
"""

import ast
import io
import os
import re
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(ROOT, "engine", "config.py")
CLASS_NAME = "Config"

# A section header line: "# -- title ----" (the layout of engine/config.py).
SECTION_PATTERN = re.compile(r"^#\s*--\s*(.*?)\s*-*\s*$")
# Source references in a comment (plan "Exporters", config_doc.py). Each pattern gives the
# source family and the text to keep.
PLAN_PATTERN = re.compile(r"\bPLAN(?:\.md)?\s*(?:§\s*)?(\d+(?:\.\d+)?(?:\s*,\s*\d+\.\d+)*)")
SOURCE_PATTERNS = (
    re.compile(r"\bP\d(?:-[A-Z])?\b"),          # task cards: P0-B, P1-A, P2
    re.compile(r"\bC\d+\b"),                    # improvement cycles: C9, C11
    re.compile(r"\bM\d+\b"),                    # research measurements: M13, M14
    re.compile(r"\bRESEARCH(?:-\d+)?\b"),       # research reports: RESEARCH, RESEARCH-2
    re.compile(r"\bR-F\d+\b"),                  # findings.md items: R-F9
    re.compile(r"\bLaw \d+\b"),                 # Laws of the Game
)
# A public forensics case is a frame number inside the public window (SCHEMA.md Section 0).
CASE_PATTERN = re.compile(r"\b1\d{4}\b")
CASE_RANGE = (11185, 18685)


class ConfigDocError(ValueError):
    """The source text and the dataclass disagree about the fields."""


def sources_of(comment):
    """Source references of a comment, in order of first appearance, without repeats."""
    found = []
    for match in PLAN_PATTERN.finditer(comment):
        for number in re.split(r"\s*,\s*", match.group(1)):
            found.append((match.start(), "PLAN " + number))
    for pattern in SOURCE_PATTERNS:
        for match in pattern.finditer(comment):
            found.append((match.start(), match.group(0)))
    for match in CASE_PATTERN.finditer(comment):
        if CASE_RANGE[0] <= int(match.group(0)) < CASE_RANGE[1]:
            found.append((match.start(), match.group(0)))
    out = []
    for _, text in sorted(found, key=lambda item: item[0]):
        if text not in out:
            out.append(text)
    return out


def _comments(source):
    """{line number: (comment text, full line)} of every comment token."""
    out = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            line = token.start[0]
            full = token.line.strip().startswith("#")
            out[line] = (token.string, full)
    return out


def _strip(comment):
    """Comment text without the leading '#' and one space."""
    text = comment[1:]
    return text[1:] if text.startswith(" ") else text


def _class_fields(source, class_name):
    """[(name, first line, last line)] of the annotated fields of `class_name`, in order."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            body = node
            break
    else:
        raise ConfigDocError("no class {} in the source".format(class_name))
    fields = []
    for node in body.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            fields.append((node.target.id, node.lineno, node.end_lineno))
    return fields, body.lineno


def _type_name(value):
    """JSON-facing type name of a config value."""
    return type(value).__name__


def _jsonable(value):
    """A tuple becomes a list. Other values pass unchanged."""
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def parse(source, values, class_name=CLASS_NAME):
    """Sections `[{title, fields}]` of the Config class in `source`.

    `values` is `{field: value}`, normally `Config.default().as_dict()`. A field gets the full-line
    comments above it since the previous field, plus its inline comment. A field with no comment
    block of its own, directly below another field, shares the block of that field. A blank line
    or a section header ends a block.
    """
    fields, class_line = _class_fields(source, class_name)
    names = [name for name, _, _ in fields]
    missing = [name for name in values if name not in names]
    if missing:
        raise ConfigDocError("fields the parser did not find: {}".format(missing))
    extra = [name for name in names if name not in values]
    if extra:
        raise ConfigDocError("fields in the source but not in the values: {}".format(extra))

    comments = _comments(source)
    lines = source.splitlines()
    starts = {first: (name, last) for name, first, last in fields}
    sections = [{"title": "", "fields": []}]
    pending, shared = [], []
    line = class_line + 1
    last_line = max(last for _, _, last in fields)
    while line <= last_line:
        text = lines[line - 1].strip()
        if line in starts:
            name, last = starts[line]
            if pending:
                shared, pending = pending, []
            inline = [_strip(comments[n][0]) for n in range(line, last + 1)
                      if n in comments and not comments[n][1]]
            comment = " ".join(part.strip() for part in shared + inline if part.strip())
            value = values[name]
            section = sections[-1]
            section["fields"].append({
                "name": name, "value": _jsonable(value), "type": _type_name(value),
                "comment": comment,
                "sources": sources_of(comment) or sources_of(section["title"])})
            line = last + 1
            continue
        if not text:
            pending, shared = [], []
        elif line in comments and comments[line][1]:
            header = SECTION_PATTERN.match(text)
            if header:
                sections.append({"title": header.group(1), "fields": []})
                pending, shared = [], []
            else:
                pending.append(_strip(comments[line][0]))
        line += 1
    return [section for section in sections if section["fields"]]


def document(cfg=None, path=CONFIG_PATH):
    """The config.json object: `{config_hash, n_fields, sections}` (SCHEMA.md 2.9)."""
    from engine.config import Config

    cfg = cfg or Config.default()
    with open(path, "r") as handle:
        source = handle.read()
    values = cfg.as_dict()
    sections = parse(source, values)
    n_fields = sum(len(section["fields"]) for section in sections)
    if n_fields != len(values):
        raise ConfigDocError("parsed {} fields, the dataclass has {}".format(n_fields, len(values)))
    return {"config_hash": cfg.config_hash(), "n_fields": n_fields, "sections": sections}
