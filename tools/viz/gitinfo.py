"""Git identity of a bundle (SCHEMA.md 2.1, `git`). It never reads the wall clock.

The commit is the newest commit that changed a file outside the bundle folders. A commit that
adds only a rebuilt bundle therefore does not change the commit that the next build records, and
a rebuild on that commit gives the same bytes.
"""

import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Folders that the exporters write. A change inside them does not make the code dirty and does
# not move the recorded commit (SCHEMA.md Section 1: byte-identical rebuilds).
BUNDLE_PATHS = ("site/data/public", "site/data/local")


class GitError(RuntimeError):
    """Git is missing, or the folder is not a git work tree."""


def run_git(args, root=ROOT):
    """stdout of `git <args>` in `root`, stripped. Raise GitError if git fails or is missing."""
    try:
        result = subprocess.run(["git"] + list(args), cwd=root, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
    except OSError as exc:
        raise GitError("git is not available: {}".format(exc))
    if result.returncode != 0:
        raise GitError("git {} failed: {}".format(" ".join(args), result.stderr.strip()))
    return result.stdout.strip()


def _excludes():
    """Pathspecs that leave out the bundle folders."""
    return ["--", "."] + [":(exclude){}".format(path) for path in BUNDLE_PATHS]


def head(root=ROOT):
    """`{commit, commit_date, tag, dirty}` of the code that a bundle comes from."""
    commit = run_git(["log", "-1", "--format=%H"] + _excludes(), root)
    if not commit:
        raise GitError("no commit outside the bundle folders")
    date = run_git(["log", "-1", "--format=%cI", commit], root)
    tags = sorted(run_git(["tag", "--points-at", commit], root).split())
    status = run_git(["status", "--porcelain"] + _excludes(), root)
    return {"commit": commit, "commit_date": date, "tag": tags[0] if tags else None,
            "dirty": bool(status)}
