"""Export the public copy of this repository and check it.

The team develops in a private repository. The submission needs a public repository. This tool
writes the public copy: the files of one commit, without the team's documents and working notes
and without the git history.

    python -m tools.export_public --out /path/to/public-repo            # export and check
    python -m tools.export_public --out /path/to/public-repo --init     # also make the one commit
    python -m tools.export_public --check /path/to/public-repo          # check a folder again
    python -m tools.export_public --list                                # print the dropped paths
    python -m tools.export_public --engine-hash                         # print the engine source hash

The export is a filter. It copies each kept file of the commit byte for byte and changes no
file. The public text must be correct in this repository before the export. The check refuses
a copy whose public text points at a file, a commit, a tag or a branch that the copy does not
hold. It also refuses a copy that matches a term of the team's list in `docs/PUBLIC_TERMS.txt`.
The copy leaves that list out, so the check of a copy skips this step.

The copy holds nothing under `docs/`. The organisers ask for the code, the pinned dependencies,
the instructions and the command that writes the file. The technical document goes to them as a
separate PDF. The copy also leaves out the build of that PDF and the tests of dropped files.

The tool sorts the kept text into two tiers:

- Entry points: `README.md`, every other kept `README.md` and the demo site. A judge reads
  them. Every link and every repository path in them must resolve. `README.md` must name the
  repository address and the site address. The site must name the repository address.
- Code: `engine/`, `tools/` and `tests/`. A comment can name a dropped file. `engine/` cannot
  change, because the technical document quotes its source hash.

The tool reads no match data and no ground truth. It needs `git` and the standard library.
docs/PUBLIC_REPOSITORY.md gives the whole procedure, from the export to the submission form.
"""

import argparse
import fnmatch
import glob
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
from urllib.parse import unquote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -- what the public copy leaves out -------------------------------------------------------------
# A path that ends with "/" is a folder. The copy keeps every other tracked file. No kept file may
# need a dropped file to run or to pass its tests. A comment can still name a dropped file.
DROP = (
    "docs/",                        # the organisers' documents, our plan, records and guides,
                                    # and the technical document, which goes to the organisers
                                    # as a separate PDF
    "findings.md",                  # the review of the first plan
    "experiments.md",               # the ledger of every scored run, a record of the private work
    "requirements-docs.txt",        # the packages of the PDF build
    "tools/docs/",                  # the build of the technical document PDF
    "tools/analysis/provenance_inventory.py",   # checks the provenance record under docs/
    "tests/test_docs_build.py",     # tests the PDF build
    "tests/test_public_copy.py",    # tests the text of the technical document
    "tests/test_provenance_inventory.py",       # tests the provenance record under docs/
    "site/dev/",                    # the development harness of the site
)
# The top-level Markdown files with an upper-case name, except README.md: the team's working
# instructions in the private repository. The copy leaves them out. A future top-level file of
# this form is a working note too. Give a public file a lower-case name or put it in a folder.
DROP_TOP_LEVEL_RE = re.compile(r"[A-Z][A-Z0-9_-]*\.md\Z")
DROP_TOP_LEVEL_EXCEPT = ("README.md",)

# -- the tiers -----------------------------------------------------------------------------------
ORGANISERS = ("evaluation/", "starter/", "data/README.md")
ENTRY_FILES = ("README.md", "site/index.html")
ENTRY_GLOBS = ("*/README.md", "site/js/*.js", "site/js/*/*.js")
# The site bundle and the vendored libraries are data, not prose.
NOT_TEXT = ("site/data/public/", "site/vendor/")
# The entry points that must name the project addresses: (path or folder prefix, labels).
REQUIRED_URLS = (("README.md", ("repository",)), ("site/", ("repository",)))

# -- paths that a text can name although git does not track them ---------------------------------
# The engine and the tools write them, and .gitignore ignores them.
GENERATED = ("data/", "cache/", "outputs/", "evaluation/results/", "evaluation/reports/",
             "evaluation/scoreboard.csv", "site/data/local/", ".venv/")

# -- files that must never be in a public copy ---------------------------------------------------
# The same patterns as .gitignore. git does not track these files, so a match means that
# somebody copied a file into the folder by hand.
FORBIDDEN_GLOBS = ("*.mp4", "*.gzip", "*.pl3container", "*.pdf", "*.npz", "*.pkl",
                   "*_possession_gt.csv", "*_possession_gt_meta.json",
                   "*_possession_gt_reference.csv")
FORBIDDEN_PREFIXES = ("cache/", "site/data/local/", "evaluation/reports/", "outputs/", "demo/")
ALLOWED_UNDER_DATA = ("data/README.md",)
ALLOWED_RESULTS = "evaluation/results/baseline_*.csv"
# The weights that the default engine loads, and the small synthetic arrays of the learned-stage
# tests. They hold trained parameters and made-up numbers, no match data. Every other .npz file
# stays forbidden: the tracking caches are .npz files too. Exact paths only, never a pattern.
ALLOWED_NPZ = (
    "engine/models/learned_v7.npz",
    "engine/models/learned_pf_v7.npz",
    "tests/fixtures/learned_net_outputs.npz",
    "tests/fixtures/learned_net_weights.npz",
    "tests/fixtures/learned_pf_synthetic.npz",
)

# -- git object names ----------------------------------------------------------------------------
# A squashed repository holds one commit, so a commit hash of the private repository is a dead
# reference. The md5 of a file has 32 hex digits and stays valid. The engine source hash has 40
# and stays valid, because it depends only on the bytes of engine/*.py and games.py. The pattern
# skips a colour: a hex value after "#" or after the URL escape "%23".
HASH_RE = re.compile(r"(?<![0-9a-zA-Z_.%#-])(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}"
                     r"(?![0-9a-zA-Z_])")
MD5_LEN = 32
# Hashes that a public text can show. Each one names something outside this repository's history.
ALLOWED_HASHES = {
    "c073f29": "the commit of the organisers' starter repository that we built on",
}
# The only tag of the public copy. `--init` creates it on the one commit.
PUBLIC_TAG = "submission"
# Names of branches and tags of the private repository. The public copy holds none of them.
PRIVATE_REF_RE = re.compile(r"\badvisor/[A-Za-z0-9_-]+|\bexp-[a-z0-9-]+\b|\bfreeze-\d{4}-\d\d-\d\d\b"
                            r"|\bcycle-c\d+\b|\b[ck]\d+[a-z]?-(best|duel)\b|\bk2-owner-backdate\b"
                            r"|\blearned-stage(-r\d)?\b")
# The team's list of terms that the copy never shows: one regular expression per line, matched
# without case. Lines that start with "#" are comments. The copy leaves the file out (DROP).
TERMS_FILE = "docs/PUBLIC_TERMS.txt"
# A path on a team member's machine.
LOCAL_PATH_RE = re.compile(r"/Users/[A-Za-z0-9._-]+|/home/(?!user\b)[A-Za-z0-9._-]+/"
                           r"|-Users-[A-Za-z0-9._-]+")

# A dropped folder counts with its final "/", so that the plain word "docs" in a sentence does
# not count.
DROPPED_NAME_RE = re.compile("|".join(
    [r"(?<![A-Za-z0-9_-])" + re.escape(d) for d in DROP if d.endswith("/")]
    + [r"(?<![A-Za-z0-9_-])" + re.escape(os.path.basename(d)) + r"(?![A-Za-z0-9_])"
       for d in DROP if not d.endswith("/") and os.path.basename(d) != "README.md"]))

MD_LINK_RE = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML_LINK_RE = re.compile(r"""(?:href|src)\s*=\s*["']([^"'#?]+)""")
BLOB_RE = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?"
                     r"/(blob|tree)/([A-Za-z0-9_.-]+)/([A-Za-z0-9_./%-]+)")
REPO_URL_RE = re.compile(r"https://github\.com/([A-Za-z0-9_-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?"
                         r"(?=[/\s'\"`)<]|\.(?![A-Za-z0-9_-])|$)")
PAGES_URL_RE = re.compile(r"https://([A-Za-z0-9_-]+)\.github\.io/([A-Za-z0-9_.-]+)/?")
PATH_TOKEN_RE = re.compile(r"`([^`\n]+)`|<code>([^<\n]+)</code>")
ENGINE_HASH_LINE_RE = re.compile(r"engine source hash[^`]*`([0-9a-f]{40})`")
# A web address. The check of dropped names skips it: blob_links checks the links into this
# repository, and a link into another project names a path of that project.
URL_RE = re.compile(r"https?://[^\s'\"<>`)]+")


class ExportError(Exception):
    """The export cannot run: a bad reference, a folder that is not empty or a git failure."""


# -- the kept set --------------------------------------------------------------------------------

def is_dropped(path):
    """Return True if the public copy leaves out the tracked file `path`."""
    if ("/" not in path and DROP_TOP_LEVEL_RE.match(path)
            and path not in DROP_TOP_LEVEL_EXCEPT):
        return True
    for entry in DROP:
        if entry.endswith("/"):
            if path.startswith(entry):
                return True
        elif path == entry:
            return True
    return False


def kept_paths(paths):
    """Return the paths of `paths` that the public copy keeps, sorted."""
    return sorted(p for p in paths if not is_dropped(p))


def tier(path):
    """Return the tier of a kept file: 'entry', 'organisers', 'data' or 'code'."""
    if path.startswith(NOT_TEXT):
        return "data"
    if path.startswith(ORGANISERS):
        return "organisers"
    if path in ENTRY_FILES or any(fnmatch.fnmatch(path, g) for g in ENTRY_GLOBS):
        return "entry"
    if os.path.basename(path) == "README.md":
        return "entry"
    return "code"


# -- git -----------------------------------------------------------------------------------------

def git(args, cwd, binary=False):
    """Run git in `cwd` and return its standard output. Raise ExportError if git fails."""
    try:
        done = subprocess.run(["git"] + list(args), cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=False)
    except OSError as err:
        raise ExportError("git did not start: {}".format(err))
    if done.returncode != 0:
        raise ExportError("git {} failed: {}".format(
            " ".join(args), done.stderr.decode("utf-8", "replace").strip()))
    return done.stdout if binary else done.stdout.decode("utf-8")


def export(ref, out, repo=None):
    """Write the kept files of commit `ref` of `repo` into the new folder `out`.

    `repo` is this checkout if it is None.

    The function reads the files from git, not from the working tree. An uncommitted change
    and an ignored file never reach the copy. The function returns the kept paths.
    """
    if os.path.exists(out) and os.listdir(out):
        raise ExportError("the folder {} is not empty".format(out))
    archive = git(["archive", "--format=tar", ref], repo or ROOT, binary=True)
    os.makedirs(out, exist_ok=True)
    root = os.path.realpath(out)
    kept = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if not member.isfile():
                if member.issym() or member.islnk():
                    raise ExportError("{} is a link: the copy holds plain files only".format(
                        member.name))
                continue
            if is_dropped(member.name):
                continue
            target = os.path.realpath(os.path.join(root, member.name))
            if not target.startswith(root + os.sep):
                raise ExportError("{} leaves the output folder".format(member.name))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as handle:
                handle.write(tar.extractfile(member).read())
            os.chmod(target, 0o755 if member.mode & 0o100 else 0o644)
            kept.append(member.name)
    return sorted(kept)


def init(out, message, tag=PUBLIC_TAG):
    """Make `out` a git repository with one commit on `main` and the tag `tag`.

    git takes the author from its own configuration. Set `user.name` and `user.email` first.
    """
    if os.path.exists(os.path.join(out, ".git")):
        raise ExportError("{} is a git repository already".format(out))
    git(["init", "-q", "-b", "main"], out)
    git(["add", "-A"], out)
    git(["commit", "-q", "-m", message], out)
    git(["tag", tag], out)
    return git(["rev-parse", "HEAD"], out).strip()


# -- the check -----------------------------------------------------------------------------------

def walk(root):
    """Return the files of the copy under `root` as sorted paths with '/' separators.

    If `root` is a git repository, the files are the ones that git tracks or can add: the
    ignored files, for example a virtual environment or `cache/`, are not part of the copy.
    If not, the files are all the files under `root`.
    """
    if os.path.exists(os.path.join(root, ".git")):
        listed = git(["ls-files", "--cached", "--others", "--exclude-standard"], root)
        return sorted(p for p in listed.split("\n") if p and os.path.isfile(os.path.join(root, p)))
    out = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        for name in files:
            out.append(os.path.relpath(os.path.join(folder, name), root).replace(os.sep, "/"))
    return sorted(out)


def read_text(root, path):
    """Return the text of a file of the copy, or None if the file is not UTF-8 text."""
    try:
        with open(os.path.join(root, path), encoding="utf-8") as handle:
            return handle.read()
    except (UnicodeDecodeError, OSError):
        return None


def engine_source_hash(root):
    """sha1 of engine/*.py and games.py under `root`.

    The same value as tools.analysis.external_sweep.engine_source_hash(). The function is here
    again so that the check can hash a folder other than this checkout.
    """
    digest = hashlib.sha1()
    paths = sorted(glob.glob(os.path.join(root, "engine", "*.py")))
    for path in paths + [os.path.join(root, "games.py")]:
        digest.update(os.path.relpath(path, root).encode("utf-8"))
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def forbidden(paths):
    """Return one problem line for each file that a public repository must not hold."""
    out = []
    for path in paths:
        name = os.path.basename(path)
        if is_dropped(path):
            out.append("{}: a dropped path is in the copy".format(path))
        elif path in ALLOWED_NPZ:
            continue
        elif any(fnmatch.fnmatch(name, g) for g in FORBIDDEN_GLOBS):
            out.append("{}: a file of a kind that is never public".format(path))
        elif path.startswith(FORBIDDEN_PREFIXES):
            out.append("{}: a generated folder that is never public".format(path))
        elif path.startswith("data/") and path not in ALLOWED_UNDER_DATA:
            out.append("{}: match data".format(path))
        elif (path.startswith("evaluation/results/") and path.endswith(".csv")
              and not fnmatch.fnmatch(path, ALLOWED_RESULTS)):
            out.append("{}: a full label file of the match".format(path))
    return out


def _exists(target, files, folders):
    """Return True if `target` is a file or a folder of the copy, or a generated path."""
    target = target.rstrip("/")
    if target in ("", "."):
        return True
    if target in files or target in folders:
        return True
    return any(target == g.rstrip("/") or target.startswith(g) for g in GENERATED)


def _folders(paths):
    """Return every folder that holds a file of `paths`."""
    out = set()
    for path in paths:
        parent = os.path.dirname(path)
        while parent:
            out.add(parent)
            parent = os.path.dirname(parent)
    return out


def _line(text, pos):
    """Return the 1-based line number of the character at `pos`."""
    return text.count("\n", 0, pos) + 1


def links(path, text, files, folders):
    """Return one problem line for each relative link of `text` that does not resolve."""
    out = []
    base = os.path.dirname(path)
    pattern = MD_LINK_RE if path.endswith(".md") else HTML_LINK_RE
    if not path.endswith((".md", ".html")):
        return out
    for m in pattern.finditer(text):
        target = unquote(m.group(1).split("#")[0].split("?")[0])
        if not target or re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("//"):
            continue
        resolved = os.path.normpath(os.path.join(base, target)).replace(os.sep, "/")
        if resolved.startswith(".."):
            out.append("{}:{}: the link '{}' leaves the repository".format(
                path, _line(text, m.start()), target))
        elif not _exists(resolved, files, folders):
            out.append("{}:{}: the link '{}' points at a file that the copy does not hold".format(
                path, _line(text, m.start()), target))
    return out


def blob_links(path, text, files, folders, owner_repo):
    """Return one problem line for each GitHub file link into this repository that is dead."""
    out = []
    for m in BLOB_RE.finditer(text):
        owner, repo, _kind, branch, target = m.groups()
        target = unquote(target.rstrip(".,;:)"))
        if owner_repo is not None and (owner, repo) != owner_repo:
            continue
        if branch != "main":
            out.append("{}:{}: the link names the branch '{}': the copy has only main".format(
                path, _line(text, m.start()), branch))
        if not _exists(target, files, folders):
            out.append("{}:{}: the GitHub link points at '{}', which the copy does not hold".format(
                path, _line(text, m.start()), target))
    return out


def named_paths(path, text, files, folders, tops):
    """Return one problem line for each repository path in code marks that does not exist.

    A token counts as a repository path if it starts with a top-level folder of the private
    repository or is a top-level file of it. A token with a placeholder or a wildcard is skipped.
    """
    out = []
    for m in PATH_TOKEN_RE.finditer(text):
        token = (m.group(1) or m.group(2)).strip()
        if _exists(token, files, folders):
            continue
        token = re.sub(r"^(python|bash|cat|open|source|cd|ls)\s+", "", token)
        token = token.split(" ")[0].split(":")[0].rstrip(".,;")
        if not token or re.search(r"[<>*{}$|=]|\.\.\.", token):
            continue
        head = token.split("/")[0]
        if "/" in token:
            if head not in tops["folders"]:
                continue
        elif token not in tops["files"]:
            continue
        if not _exists(token, files, folders):
            out.append("{}:{}: the text names `{}`, which the copy does not hold".format(
                path, _line(text, m.start()), token))
    return out


def git_names(path, text, engine_hash):
    """Return one problem line for each commit hash or private branch or tag name in `text`."""
    out = []
    for m in HASH_RE.finditer(text):
        token = m.group(0)
        if len(token) == MD5_LEN or token in ALLOWED_HASHES or token == engine_hash:
            continue
        out.append("{}:{}: '{}' reads as a commit hash: a squashed repository does not hold "
                   "it".format(path, _line(text, m.start()), token))
    for m in PRIVATE_REF_RE.finditer(text):
        out.append("{}:{}: '{}' is a branch or a tag of the private repository".format(
            path, _line(text, m.start()), m.group(0)))
    return out


def dropped_names(path, text):
    """Return one problem line for each dropped file or folder that `text` names.

    The check looks for the path of each dropped folder and for the file name of each dropped
    file. `README.md` is left out, because kept folders have a file of that name. The check skips
    web addresses (URL_RE).
    """
    out = []
    plain = URL_RE.sub(lambda m: " " * len(m.group(0)), text)
    for m in DROPPED_NAME_RE.finditer(plain):
        out.append("{}:{}: the text names '{}', which the copy does not hold".format(
            path, _line(text, m.start()), m.group(0)))
    return out


def project_urls(texts):
    """Return ({(owner, repo): [where]}, {(owner, repo): [where]}) for the repository and Pages URLs.

    Only a repository URL whose name equals the name of a Pages URL or of another clone URL counts.
    The entry points also link to other projects, for example the organisers' repository.
    """
    clones, pages = {}, {}
    for path, text in texts:
        for m in PAGES_URL_RE.finditer(text):
            pages.setdefault((m.group(1), m.group(2)), []).append(
                "{}:{}".format(path, _line(text, m.start())))
        for m in REPO_URL_RE.finditer(text):
            clones.setdefault((m.group(1), m.group(2)), []).append(
                "{}:{}".format(path, _line(text, m.start())))
    names = {repo for _owner, repo in pages}
    ours = {key: where for key, where in clones.items() if key[1] in names}
    return ours, pages


def urls(texts, repo_url=None, pages_url=None):
    """Return the problem lines of the project URLs and the (owner, repo) of the repository.

    The entry points must name one repository URL and one Pages URL. If `repo_url` or `pages_url`
    is given, the text must name that URL.
    """
    out = []
    ours, pages = project_urls(texts)
    for label, found in (("repository", ours), ("Pages", pages)):
        if len(found) > 1:
            out.append("the text names {} {} URLs: {}".format(len(found), label, "; ".join(
                "{}/{} at {}".format(o, r, where[0]) for (o, r), where in sorted(found.items()))))
    owner_repo = sorted(ours)[0] if ours else None
    if repo_url is not None:
        m = REPO_URL_RE.match(repo_url.rstrip("/") + "/")
        want = (m.group(1), m.group(2)) if m else None
        if want is None:
            out.append("--repo-url '{}' is not a GitHub repository URL".format(repo_url))
        elif set(ours) != {want}:
            out.append("the text names the repository {} and not {}".format(
                ", ".join("{}/{}".format(o, r) for o, r in sorted(ours)) or "(none)",
                "{}/{}".format(*want)))
    if pages_url is not None:
        m = PAGES_URL_RE.match(pages_url)
        want = (m.group(1), m.group(2)) if m else None
        if want is None:
            out.append("--pages-url '{}' is not a GitHub Pages URL".format(pages_url))
        elif set(pages) != {want}:
            out.append("the text names the Pages site {} and not {}".format(
                ", ".join("{}.github.io/{}".format(o, r) for o, r in sorted(pages)) or "(none)",
                "{}.github.io/{}".format(*want)))
    return out, owner_repo


def required_urls(texts):
    """Return one problem line for each entry point of REQUIRED_URLS that misses a project URL.

    `texts` is a list of (path, text). A prefix that ends with "/" stands for every entry point
    under it, and one of them must name the URL. An entry point that the copy does not hold is
    not checked here.
    """
    out = []
    names = {repo for _owner, repo in project_urls(texts)[1]}
    for where, labels in REQUIRED_URLS:
        group = [(p, t) for p, t in texts
                 if (p.startswith(where) if where.endswith("/") else p == where)]
        if not group:
            continue
        pages = project_urls(group)[1]
        clones = {(m.group(1), m.group(2)) for _p, t in group for m in REPO_URL_RE.finditer(t)}
        clones = {key for key in clones if key[1] in names}
        found = {"repository": clones, "Pages": pages}
        for label in labels:
            if not found[label]:
                out.append("{} names no {} URL".format(where, label))
    return out


def load_terms(root=ROOT):
    """Return the compiled terms of TERMS_FILE under `root`, or an empty list without the file."""
    path = os.path.join(root, TERMS_FILE)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    return compile_terms(line for line in lines if line and not line.startswith("#"))


def compile_terms(patterns):
    """Return the regular expressions of `patterns`, matched without case."""
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def terms_found(path, text, terms):
    """Return one problem line for each term that the path or the text (None: binary) matches."""
    out = []
    for term in terms:
        if term.search(path):
            out.append("{}: the path matches a term of {}".format(path, TERMS_FILE))
        found = term.search(text) if text is not None else None
        if found:
            out.append("{}:{}: the text matches a term of {}: '{}'".format(
                path, _line(text, found.start()), TERMS_FILE, found.group(0)))
    return out


def check(root, tops=None, repo_url=None, pages_url=None, paths=None, terms=None):
    """Check the public copy in the folder `root`. Return (problems, notes), two lists of lines.

    A problem stops the export. A note is a fact for the person who publishes the copy.
    `tops` gives the top-level folders and files of the private repository. The default takes
    them from this checkout. `paths` gives the files of the copy. The default is every file of
    `root`. A test passes the kept files of the private checkout, to check the text before an
    export. `terms` gives the compiled terms that no path and no text can match. The default
    reads TERMS_FILE of this checkout. The check skips the organisers' files and site/vendor/.
    """
    if tops is None:
        tops = top_level(ROOT)
    if terms is None:
        terms = load_terms()
    if paths is None:
        paths = walk(root)
    paths = sorted(paths)
    files, folders = set(paths), _folders(paths)
    problems = forbidden(paths)
    notes = []

    engine_hash = None
    if os.path.isdir(os.path.join(root, "engine")) and os.path.isfile(os.path.join(root, "games.py")):
        engine_hash = engine_source_hash(root)
        notes.append("engine source hash: {}".format(engine_hash))

    entry = [(p, read_text(root, p)) for p in paths if tier(p) == "entry"]
    entry = [(p, t) for p, t in entry if t is not None]
    url_problems, owner_repo = urls(entry, repo_url, pages_url)
    problems += url_problems
    problems += required_urls(entry)
    if owner_repo is not None:
        notes.append("repository URL in the text: https://github.com/{}/{}".format(*owner_repo))
    for path, text in entry:
        problems += links(path, text, files, folders)
        problems += blob_links(path, text, files, folders, owner_repo)
        problems += named_paths(path, text, files, folders, tops)
        problems += git_names(path, text, engine_hash)
        problems += dropped_names(path, text)
        for m in LOCAL_PATH_RE.finditer(text):
            problems.append("{}:{}: '{}' is a path on a team member's machine".format(
                path, _line(text, m.start()), m.group(0)))
        for m in ENGINE_HASH_LINE_RE.finditer(text):
            if engine_hash is not None and m.group(1) != engine_hash:
                problems.append("{}:{}: the text gives the engine source hash {} and the copy "
                                "has {}".format(path, _line(text, m.start()), m.group(1),
                                                engine_hash))

    # Every path and every text of the copy, except the organisers' files and the vendored code.
    for path in paths:
        if terms and tier(path) != "organisers" and not path.startswith("site/vendor/"):
            problems += terms_found(path, read_text(root, path), terms)

    # Code: count what a reader can meet, and stop nothing.
    local, dangling = {}, {}
    for path in paths:
        kind = tier(path)
        if kind != "code":
            continue
        text = read_text(root, path)
        if text is None:
            continue
        if LOCAL_PATH_RE.search(text):
            local.setdefault(kind, []).append(path)
        if dropped_names(path, text):
            dangling.setdefault(kind, []).append(path)
    for kind in sorted(local):
        notes.append("{} {} files hold a path on a team member's machine: {}".format(
            len(local[kind]), kind, ", ".join(local[kind][:6])
            + (" ..." if len(local[kind]) > 6 else "")))
    for kind in sorted(dangling):
        notes.append("{} {} files name a dropped file in a comment".format(
            len(dangling[kind]), kind))
    notes.append("{} files, {:.1f} MB".format(
        len(paths), sum(os.path.getsize(os.path.join(root, p)) for p in paths) / 1e6))
    return list(dict.fromkeys(problems)), notes


def top_level(repo):
    """Return {'folders': set, 'files': set} of the tracked top-level names of `repo`."""
    tracked = git(["ls-files"], repo).split("\n")
    folders = {p.split("/")[0] for p in tracked if "/" in p}
    files = {p for p in tracked if p and "/" not in p}
    return {"folders": folders, "files": files}


# -- command line --------------------------------------------------------------------------------

def main(argv=None):
    """Run the command line. Return 0 if the copy passes the check, 1 if not, 2 on an error."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", help="new or empty folder for the public copy")
    parser.add_argument("--ref", default="HEAD", help="commit to export (default: HEAD)")
    parser.add_argument("--check", metavar="DIR", help="check a folder and export nothing")
    parser.add_argument("--list", action="store_true", help="print the dropped tracked files")
    parser.add_argument("--engine-hash", action="store_true",
                        help="print the engine source hash of this checkout")
    parser.add_argument("--init", action="store_true",
                        help="make one commit on main and the tag '{}'".format(PUBLIC_TAG))
    parser.add_argument("--message", default="The Outliers: submission for More than a Hack 2027",
                        help="commit message for --init")
    parser.add_argument("--repo-url", help="the URL that the text must name for the repository")
    parser.add_argument("--pages-url", help="the URL that the text must name for the site")
    args = parser.parse_args(argv)

    try:
        if args.engine_hash:
            print(engine_source_hash(ROOT))
            return 0
        if args.list:
            tracked = [p for p in git(["ls-tree", "-r", "--name-only", args.ref], ROOT).split("\n")
                       if p]
            dropped = [p for p in tracked if is_dropped(p)]
            print("\n".join(dropped))
            print("{} of {} tracked files dropped".format(len(dropped), len(tracked)),
                  file=sys.stderr)
            return 0
        if not args.out and not args.check:
            parser.error("give --out, --check, --list or --engine-hash")
        root = args.check or args.out
        tops = None
        if args.out:
            kept = export(args.ref, args.out)
            source = git(["rev-parse", args.ref + "^{commit}"], ROOT).strip()
            print("exported {} files of {} to {}".format(len(kept), source[:12], args.out))
            tracked = [p for p in git(["ls-tree", "-r", "--name-only", args.ref], ROOT).split("\n")
                       if p]
            tops = {"folders": {p.split("/")[0] for p in tracked if "/" in p},
                    "files": {p for p in tracked if "/" not in p}}
        problems, notes = check(root, tops, args.repo_url, args.pages_url)
    except ExportError as err:
        print("error: {}".format(err), file=sys.stderr)
        return 2
    for note in notes:
        print("note: {}".format(note))
    for problem in problems:
        print("PROBLEM: {}".format(problem))
    if problems:
        print("FAIL: {} problems. Fix the text in the private repository, commit, and export "
              "again.".format(len(problems)))
        return 1
    print("PASS: the copy has no dead reference and no forbidden file.")
    if args.init and args.out:
        try:
            commit = init(args.out, args.message)
        except ExportError as err:
            print("error: {}".format(err), file=sys.stderr)
            return 2
        print("one commit {} on main, tagged '{}'".format(commit[:12], PUBLIC_TAG))
    return 0


if __name__ == "__main__":
    sys.exit(main())
