"""The project history for `ladder.json` and `cycles.json` (SCHEMA.md 2.10).

    parse_experiments(text)     rows of the experiments.md table
    parse_current_best(text)    rows of the "Current best" table of docs/plan/PROGRESS.md
    parse_bootstrap(rows)       {version: {lo, hi}} from the "p5 bootstrap" ledger row
    ladder(experiments, best)   {rows, rejected}
    cycles(text)                headings and hypotheses of docs/plan/LEARNINGS.md

The PROGRESS table gives the order of the ladder. The ledger gives the numbers, the md5 and the
note. A number that the ledger leaves out ("—") comes from the PROGRESS table.
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENTS_PATH = os.path.join(ROOT, "experiments.md")
PROGRESS_PATH = os.path.join(ROOT, "docs", "plan", "PROGRESS.md")
LEARNINGS_PATH = os.path.join(ROOT, "docs", "plan", "LEARNINGS.md")

# Column keys of the experiments.md table, in the order of its header (experiments.md).
EXPERIMENT_COLUMNS = ("date", "name", "git", "config_hash", "spell_f1", "macro_f1",
                      "mean_latency", "n_pred_spells", "possessions", "d_share", "mean_dur_s",
                      "verdict", "note")
# Column keys of the PROGRESS "Current best" table (docs/plan/PROGRESS.md).
BEST_COLUMNS = ("label", "spell_f1", "macro_f1", "mean_latency", "possessions", "d_share",
                "d_segments", "date")
BEST_HEADING = "## Current best"
# Characters that the tables write for a minus sign and for "no value" (experiments.md).
MINUS_SIGNS = ("−", "–")
NO_VALUE = ("—", "-", "")
# A version name, with an optional comment in parentheses: "c4_restart_anchor (speed anchor)".
VERSION_PATTERN = re.compile(r"^([a-z0-9_]+)(?:\s+\(.*\))?$")
MD5_PATTERN = re.compile(r"\bmd5\s+([0-9a-f]{32})\b")
MD5_PREFIX_PATTERN = re.compile(r"\bmd5\s+([0-9a-f]{8,31})…")
VERDICT_PATTERN = re.compile(r"^(keep|tie|reject|info)\b\s*(?:\(([^)]*)\))?")
CYCLE_NAME_PATTERN = re.compile(r"^c(\d+)_")
BOOTSTRAP_ROW = "p5 bootstrap"
BOOTSTRAP_PATTERN = re.compile(r"\b([a-z0-9_]+) [0-9.]+ \[([0-9.]+), ([0-9.]+)\]")
LEARNING_HEADING = re.compile(r"^## (\S+) — (.*?) \(([^,]*), ([^)]*)\)\s*$")
HYPOTHESIS_LABEL = "**Hypothesis.**"
# A sentence ends at a full stop, a question mark or an exclamation mark before a space and a
# capital letter, or at the end of the text.
SENTENCE_END = re.compile(r"[.!?](?=\s+[A-Z(`*]|\s*$)")


def split_row(line):
    """Cells of one Markdown table row. A `\\|` inside a cell is a literal pipe."""
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    cells = re.split(r"(?<!\\)\|", text)
    return [cell.strip().replace("\\|", "|") for cell in cells]


def normalise(text):
    """`text` with the table minus signs replaced by '-'."""
    for sign in MINUS_SIGNS:
        text = text.replace(sign, "-")
    return text.strip()


def number(text):
    """The leading number of a cell as a float, or None. '49.8%' gives 0.498."""
    text = normalise(text).replace("**", "")
    if text in NO_VALUE:
        return None
    match = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*(%?)", text)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2):
        value = round(value / 100.0, 6)
    return value


def integer(text):
    """The leading number of a cell as an int, or None."""
    value = number(text)
    return None if value is None else int(round(value))


def _table_rows(lines, start):
    """Rows of the Markdown table whose header line is lines[start]."""
    rows = []
    for line in lines[start + 2:]:
        if not line.strip().startswith("|"):
            break
        rows.append(split_row(line))
    return rows


def parse_verdict(text):
    """(status, cycle) of a verdict cell: 'keep (C8)' gives ('keep', 'C8')."""
    match = VERDICT_PATTERN.match(normalise(text))
    if not match:
        return None, None
    return match.group(1), match.group(2)


def md5_of_note(note):
    """The first full md5 of a note, else the first md5 prefix, else None."""
    match = MD5_PATTERN.search(note) or MD5_PREFIX_PATTERN.search(note)
    return match.group(1) if match else None


def parse_experiments(text):
    """Rows of the experiments.md table as dicts, in file order."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("| date | name |"):
            break
    else:
        raise ValueError("experiments.md: no table header")
    out = []
    width = len(EXPERIMENT_COLUMNS)
    for cells in _table_rows(lines, index):
        if len(cells) > width:
            # A bare "|" in the note column, for example "|δ|", splits the note. The note is the
            # last column, so the extra cells join back into it.
            cells = cells[:width - 1] + [" | ".join(cells[width - 1:])]
        if len(cells) != width:
            raise ValueError("experiments.md: row with {} cells: {}".format(len(cells), cells[:2]))
        row = dict(zip(EXPERIMENT_COLUMNS, cells))
        status, cycle = parse_verdict(row["verdict"])
        row.update(status=status, cycle=cycle, md5=md5_of_note(row["note"]))
        out.append(row)
    return out


def parse_current_best(text):
    """Rows of the PROGRESS "Current best" table as dicts, in table order."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == BEST_HEADING:
            break
    else:
        raise ValueError("PROGRESS.md: no '{}' heading".format(BEST_HEADING))
    header = next(i for i in range(index + 1, len(lines)) if lines[i].strip().startswith("|"))
    out = []
    for cells in _table_rows(lines, header):
        row = dict(zip(BEST_COLUMNS, cells))
        label = row["label"].replace("**", "").strip()
        row["label"] = label
        row["version"] = re.match(r"^([a-z0-9_]+)", label).group(1)
        out.append(row)
    return out


def parse_bootstrap(rows):
    """{version: {lo, hi}} of spellF1 from the ledger row named BOOTSTRAP_ROW."""
    out = {}
    for row in rows:
        if row["name"].startswith(BOOTSTRAP_ROW):
            for match in BOOTSTRAP_PATTERN.finditer(row["note"]):
                name = match.group(1)
                if name not in out:
                    out[name] = {"lo": float(match.group(2)), "hi": float(match.group(3))}
    return out


def _text(cell):
    """A text cell, or None for an empty cell or a dash."""
    if cell is None or cell.strip() in NO_VALUE:
        return None
    return cell.strip()


def _pick(ledger_cell, best_cell, kind):
    """The ledger value, else the PROGRESS value."""
    convert = {"float": number, "int": integer}[kind]
    value = convert(ledger_cell) if ledger_cell is not None else None
    if value is None and best_cell is not None:
        value = convert(best_cell)
    return value


def _ladder_row(version, entry, best, status, bootstrap, md5_override=None):
    """One ladder row from a ledger row (or None) and a PROGRESS row (or None)."""
    entry = entry or {}
    best = best or {}
    cycle = entry.get("cycle")
    if status == "baseline":
        cycle = "baseline"
    elif cycle is None:
        match = CYCLE_NAME_PATTERN.match(version)
        cycle = "C{}".format(match.group(1)) if match else None
    return {
        "version": version, "cycle": cycle, "status": status,
        "label": best.get("label", version),
        "date": entry.get("date") or best.get("date"),
        "git": _text(entry.get("git")), "config_hash": _text(entry.get("config_hash")),
        "spell_f1": _pick(entry.get("spell_f1"), best.get("spell_f1"), "float"),
        "macro_f1": _pick(entry.get("macro_f1"), best.get("macro_f1"), "float"),
        "mean_latency": _pick(entry.get("mean_latency"), best.get("mean_latency"), "float"),
        "possessions": _pick(entry.get("possessions"), best.get("possessions"), "int"),
        "d_share": _pick(entry.get("d_share"), best.get("d_share"), "float"),
        "d_segments": _pick(None, best.get("d_segments"), "int"),
        "md5": md5_override or entry.get("md5"),
        "bootstrap": bootstrap.get(version),
        "note": entry.get("note", "")}


def ladder(experiments, best, md5_by_version=None):
    """`{rows, rejected}` of ladder.json (SCHEMA.md 2.10).

    `md5_by_version` gives md5 values measured on files, for example the committed baselines.
    It wins over the ledger note.
    """
    md5_by_version = md5_by_version or {}
    bootstrap = parse_bootstrap(experiments)
    rows = []
    for item in best:
        version = item["version"]
        kept = [r for r in experiments if r["name"] == version and r["status"] == "keep"]
        named = [r for r in experiments if r["name"] == version]
        entry = kept[-1] if kept else (named[-1] if named else None)
        status = "kept" if kept else "baseline"
        rows.append(_ladder_row(version, entry, item, status, bootstrap,
                                md5_by_version.get(version)))
    rejected = []
    for entry in experiments:
        match = VERSION_PATTERN.match(entry["name"])
        if entry["status"] == "reject" and match:
            row = _ladder_row(match.group(1), entry, None, "rejected", bootstrap)
            # External-only verdicts remain in the ledger, but have no public chart value.
            if row["spell_f1"] is not None:
                rejected.append(row)
    return {"rows": rows, "rejected": rejected}


def first_sentence(text):
    """The first sentence of a paragraph."""
    text = " ".join(text.split())
    match = SENTENCE_END.search(text)
    return text[:match.end()] if match else text


def cycles(text):
    """[{id, title, date, status, hypothesis}] of LEARNINGS.md, in file order.

    The template heading `C<n>` is skipped. `hypothesis` is the first sentence after the
    "Hypothesis." label of the cycle, or "" when the cycle has none.
    """
    out = []
    current = None
    paragraph = None
    for line in text.splitlines():
        heading = LEARNING_HEADING.match(line)
        if heading:
            current = None
            if heading.group(1).startswith("C<"):
                continue
            current = {"id": heading.group(1), "title": heading.group(2),
                       "date": heading.group(3), "status": heading.group(4), "hypothesis": ""}
            out.append(current)
            paragraph = None
            continue
        if current is None:
            continue
        if line.startswith(HYPOTHESIS_LABEL) and not current["hypothesis"]:
            paragraph = [line[len(HYPOTHESIS_LABEL):]]
            continue
        if paragraph is not None:
            if line.strip() and not line.startswith("**"):
                paragraph.append(line)
                continue
            current["hypothesis"] = first_sentence(" ".join(paragraph))
            paragraph = None
    if current is not None and paragraph is not None:
        current["hypothesis"] = first_sentence(" ".join(paragraph))
    return out


def read(path):
    """Text of a UTF-8 file.

    The public copy of the repository leaves out the two working notes (tools/export_public.py).
    A missing note raises FileNotFoundError with the reason, and build_site prints it.
    """
    if path in (PROGRESS_PATH, LEARNINGS_PATH) and not os.path.isfile(path):
        raise FileNotFoundError(
            "{} is not here. The public repository leaves out this working note and holds the "
            "built bundle in site/data/public/. Rebuild the bundle in the private "
            "repository.".format(os.path.relpath(path, ROOT)))
    with open(path, "rb") as handle:
        return handle.read().decode("utf-8")
