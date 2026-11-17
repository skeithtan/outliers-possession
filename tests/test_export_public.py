"""Tests of tools/export_public.py on a synthetic repository.

Each test builds a small git repository in a temporary folder. No test reads match data, and no
test exports this repository. The last test checks the drop list against the tracked files of
this checkout.
"""

import os
import subprocess

import pytest

from tools import export_public as ep

ENGINE_PY = "X = 1\n"
GAMES_PY = "GAMES = {}\n"


def git(repo, *args):
    """Run git in `repo` with a fixed identity and return its output."""
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    return subprocess.run(["git"] + list(args), cwd=str(repo), env=env, check=True,
                          stdout=subprocess.PIPE).stdout.decode("utf-8")


def make_repo(tmp_path, files):
    """Create a git repository under `tmp_path` with one commit that holds `files`."""
    repo = tmp_path / "private"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "first")
    return repo


def base_files():
    """Return the files of a small repository whose public copy passes the check."""
    return {
        "README.md": "# Entry\n\nSee [the tools](tools/) and `engine/run.py`.\n"
                     "The site is at https://team.github.io/entry/ and the code is at\n"
                     "https://github.com/team/entry.git. The organisers publish\n"
                     "https://github.com/metrica-sports/more-than-a-hack-2027.\n",
        "NOTES.md": "rules\n",
        "findings.md": "review\n",
        "experiments.md": "| run |\n",
        "games.py": GAMES_PY,
        "engine/run.py": ENGINE_PY,
        "docs/WRITING.md": "guide\n",
        "docs/plan/PLAN.md": "plan, see PROGRESS.md and commit abc1234\n",
        "docs/plan/PROGRESS.md": "status\n",
        "docs/challenge/RULES.md": "rules\n",
        "docs/technical/TECHNICAL_DOCUMENT.md": "document\n",
        "tools/docs/build_pdf.py": "# build\n",
        "tools/run.py": "# reads docs/plan/PLAN.md\n",
        "requirements-docs.txt": "markdown==3.9\n",
        "site/index.html": '<a href="css/a.css">x</a>'
                           '<a href="https://github.com/team/entry/blob/main/tools/run.py">p</a>\n',
        "site/css/a.css": "a{}\n",
        "site/dev/harness.js": "dev\n",
        "site/data/public/manifest.json": "{}\n",
        "data/README.md": "no data here\n",
    }


def export_and_check(tmp_path, files, **kwargs):
    """Export the synthetic repository and return (kept paths, problems, notes)."""
    repo = make_repo(tmp_path, files)
    out = tmp_path / "public"
    kept = ep.export("HEAD", str(out), repo=str(repo))
    problems, notes = ep.check(str(out), ep.top_level(str(repo)), **kwargs)
    return kept, problems, notes


def test_is_dropped_follows_the_lists():
    assert ep.is_dropped("NOTES.md")
    assert not ep.is_dropped("README.md")
    assert not ep.is_dropped("notes.md")
    assert ep.is_dropped("docs/NOTES.md")
    assert ep.is_dropped("docs/plan/tasks/P0-A-data-layer.md")
    assert ep.is_dropped("docs/plan/PLAN.md")
    assert ep.is_dropped("docs/challenge/RULES.md")
    assert ep.is_dropped("docs/technical/The Outliers.pdf")
    assert ep.is_dropped("tools/docs/build_pdf.py")
    assert ep.is_dropped("experiments.md")
    assert ep.is_dropped("requirements-docs.txt")
    assert not ep.is_dropped("docsite/index.html")
    assert not ep.is_dropped("tools/analysis/provenance_recheck.py")
    assert not ep.is_dropped("engine/run.py")
    assert not ep.is_dropped("NOTES.md.bak")


def test_tiers():
    assert ep.tier("README.md") == "entry"
    assert ep.tier("tools/analysis/README.md") == "entry"
    assert ep.tier("site/index.html") == "entry"
    assert ep.tier("site/js/sections/repro.js") == "entry"
    assert ep.tier("engine/config.py") == "code"
    assert ep.tier("evaluation/run_evaluation.py") == "organisers"
    assert ep.tier("site/data/public/config.json") == "data"


def test_export_keeps_and_drops(tmp_path):
    kept, problems, notes = export_and_check(tmp_path, base_files())
    assert problems == []
    assert "engine/run.py" in kept and "tools/run.py" in kept
    assert not [p for p in kept if p.startswith("docs/")]
    assert not (tmp_path / "public" / "docs").exists()
    for gone in ("NOTES.md", "findings.md", "experiments.md", "docs/WRITING.md",
                 "docs/plan/PLAN.md", "docs/challenge/RULES.md",
                 "docs/technical/TECHNICAL_DOCUMENT.md", "tools/docs/build_pdf.py",
                 "requirements-docs.txt", "site/dev/harness.js"):
        assert gone not in kept
        assert not (tmp_path / "public" / gone).exists()
    assert any(note.startswith("engine source hash: ") for note in notes)
    assert "repository URL in the text: https://github.com/team/entry" in notes


def test_export_copies_bytes_and_reads_the_commit_not_the_working_tree(tmp_path):
    repo = make_repo(tmp_path, base_files())
    (repo / "engine" / "run.py").write_text("X = 2\n", encoding="utf-8")
    (repo / "data" / "match.mp4").write_bytes(b"\x00video")
    out = tmp_path / "public"
    ep.export("HEAD", str(out), repo=str(repo))
    assert (out / "engine" / "run.py").read_text(encoding="utf-8") == ENGINE_PY
    assert not (out / "data" / "match.mp4").exists()
    assert ep.engine_source_hash(str(out)) != ep.engine_source_hash(str(repo))


def test_export_refuses_a_folder_that_is_not_empty(tmp_path):
    repo = make_repo(tmp_path, base_files())
    out = tmp_path / "public"
    out.mkdir()
    (out / "old.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ep.ExportError):
        ep.export("HEAD", str(out), repo=str(repo))


def test_export_refuses_an_unknown_ref(tmp_path):
    repo = make_repo(tmp_path, base_files())
    with pytest.raises(ep.ExportError):
        ep.export("no-such-ref", str(tmp_path / "public"), repo=str(repo))


def test_two_exports_are_identical(tmp_path):
    repo = make_repo(tmp_path, base_files())
    first, second = tmp_path / "a", tmp_path / "b"
    assert ep.export("HEAD", str(first), repo=str(repo)) == ep.export("HEAD", str(second),
                                                                     repo=str(repo))
    for path in ep.walk(str(first)):
        assert (first / path).read_bytes() == (second / path).read_bytes()


def test_engine_source_hash_equals_the_sweep_tool():
    from tools.analysis import external_sweep
    assert ep.engine_source_hash(ep.ROOT) == external_sweep.engine_source_hash()


@pytest.mark.parametrize("text, part", [
    ("See [the rules](NOTES.md).\n", "the link 'NOTES.md'"),
    ("See [the cards](docs/plan/tasks/).\n", "the link 'docs/plan/tasks/'"),
    ("See [the plan](docs/plan/PLAN.md).\n", "the link 'docs/plan/PLAN.md'"),
    ("Read the [rules](docs/challenge/RULES.md).\n", "the link 'docs/challenge/RULES.md'"),
    ("The document is `docs/technical/The Outliers.pdf`.\n", "`docs/technical/The"),
    ("The records are under docs/plan/calibration.\n", "names 'docs/'"),
    ("The ledger is experiments.md.\n", "names 'experiments.md'"),
    ("Build the PDF with tools/docs/build_pdf.py.\n", "names 'tools/docs/'"),
    ("See https://github.com/team/entry/tree/main/docs/challenge.\n",
     "points at 'docs/challenge'"),
    ("The status is in `docs/plan/PROGRESS.md`.\n", "`docs/plan/PROGRESS.md`"),
    ("The review is in findings.md.\n", "names 'findings.md'"),
    ("The notes are under docs/plan/research.\n", "names 'docs/'"),
    ("The engine last changed at `bc68ca6`.\n", "'bc68ca6' reads as a commit hash"),
    ("The code is at the tag `exp-learned-stage-r1`.\n", "a branch or a tag"),
    ("The record is in `advisor/ASK-012`.\n", "a branch or a tag"),
    ("Output in /Us" + "ers/someone/out.\n", "a path on a team member's machine"),
    ("See https://github.com/team/entry/blob/main/docs/plan/HANDOFF.md.\n",
     "points at 'docs/plan/HANDOFF.md'"),
    ("See https://github.com/team/entry/blob/dev/experiments.md.\n", "the branch 'dev'"),
    ("See [x](../outside.md).\n", "leaves the repository"),
])
def test_check_reports_a_dead_reference(tmp_path, text, part):
    files = base_files()
    files["README.md"] += text
    _kept, problems, _notes = export_and_check(tmp_path, files)
    assert len(problems) >= 1
    assert any(part in line and line.startswith("README.md:") for line in problems), problems


@pytest.mark.parametrize("text", [
    "The md5 is `7844cad862e38d55cf9a3fc29b18ae85`.\n",
    "We built on the organisers' commit `c073f29`.\n",
    "The engine writes `cache/` and `evaluation/results/<name>.csv`.\n",
    "The organisers' docs say more.\n",
    "Read [RULES.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md).\n",
    "Their docs are at https://example.org/docs/challenge/RULES.md.\n",
    "Run `python -m engine.run --out file.csv`.\n",
    "The colour is #0b0e14ff and the icon holds fill='%230b0e14'.\n",
    "See [the organisers](https://github.com/metrica-sports/more-than-a-hack-2027/blob/x/y.md).\n",
    "Set `MTAH_DATA_DIR=/path/to/mtah2027`.\n",
])
def test_check_accepts_a_valid_reference(tmp_path, text):
    files = base_files()
    files["README.md"] += text
    _kept, problems, _notes = export_and_check(tmp_path, files)
    assert problems == []


def test_check_accepts_the_engine_source_hash_and_refuses_another(tmp_path):
    files = base_files()
    repo = make_repo(tmp_path, files)
    digest = ep.engine_source_hash(str(repo))
    out = tmp_path / "public"
    ep.export("HEAD", str(out), repo=str(repo))
    readme = out / "README.md"
    original = readme.read_text(encoding="utf-8")
    readme.write_text(original + "The engine source hash is `%s`.\n" % digest, encoding="utf-8")
    assert ep.check(str(out), ep.top_level(str(repo)))[0] == []
    wrong = "0" * 39 + "a"
    readme.write_text(original + "The engine source hash is `%s`.\n" % wrong, encoding="utf-8")
    problems = ep.check(str(out), ep.top_level(str(repo)))[0]
    assert any("the copy has " + digest in line for line in problems)


def test_a_record_and_a_comment_can_name_a_dropped_file(tmp_path):
    files = base_files()
    files["engine/run.py"] = "# findings.md note, PLAN.md 3.8, commit abc1234\n" + ENGINE_PY
    _kept, problems, notes = export_and_check(tmp_path, files)
    assert problems == []
    assert "2 code files name a dropped file in a comment" in notes


def test_only_the_listed_npz_files_pass(tmp_path):
    """The weights and the synthetic fixtures pass by exact path; any other .npz is refused."""
    assert ep.forbidden(list(ep.ALLOWED_NPZ)) == []
    others = ["engine/models/learned_v8.npz", "engine/models/sub/learned_v7.npz",
              "cache/tracking.npz", "tests/fixtures/tracking.npz", "learned_v7.npz"]
    flagged = {line.split(":")[0] for line in ep.forbidden(others)}
    assert flagged == set(others)
    assert all("*" not in path and "?" not in path for path in ep.ALLOWED_NPZ)


def test_check_refuses_forbidden_files(tmp_path):
    repo = make_repo(tmp_path, base_files())
    out = tmp_path / "public"
    ep.export("HEAD", str(out), repo=str(repo))
    for name in ("data/BAR-ATH_tactical.mp4", "cache/tracking.npz", "The Outliers.pdf",
                 "evaluation/results/c23_backdate.csv", "site/data/local/replay.bin",
                 "NOTES.md", "docs/plan/tasks/P0-A.md"):
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (out / "evaluation" / "results" / "baseline_smooth50.csv").write_bytes(b"x")
    problems = ep.check(str(out), ep.top_level(str(repo)))[0]
    flagged = {line.split(":")[0] for line in problems}
    assert flagged == {"data/BAR-ATH_tactical.mp4", "cache/tracking.npz", "The Outliers.pdf",
                       "evaluation/results/c23_backdate.csv", "site/data/local/replay.bin",
                       "NOTES.md", "docs/plan/tasks/P0-A.md"}


def test_the_copy_holds_no_pdf_and_nothing_under_docs(tmp_path):
    """The technical document goes to the organisers as a separate PDF, never in the copy."""
    files = base_files()
    files["docs/technical/The Outliers.pdf"] = "%PDF-1.7\n"
    kept, problems, _notes = export_and_check(tmp_path, files)
    assert "docs/technical/The Outliers.pdf" not in kept
    assert problems == []
    for name in ("The Outliers.pdf", "docs/technical/The Outliers.pdf", "docs/notes.txt"):
        path = tmp_path / "public" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.7\n")
    problems = ep.check(str(tmp_path / "public"), ep.top_level(str(tmp_path / "private")))[0]
    assert sorted(line.split(":")[0] for line in problems) == [
        "The Outliers.pdf", "docs/notes.txt", "docs/technical/The Outliers.pdf"]


def test_readme_and_site_must_name_the_project_urls(tmp_path):
    """README.md names the repository. The site names the repository."""
    files = base_files()
    files["README.md"] = files["README.md"].replace(
        "https://github.com/team/entry.git", "our repository")
    files["site/index.html"] = '<a href="css/a.css">x</a>\n'
    files["site/js/team.js"] = "const PAGES = 'https://team.github.io/entry/';\n"
    _kept, problems, _notes = export_and_check(tmp_path, files)
    assert "README.md names no repository URL" in problems
    assert "site/ names no repository URL" in problems
    assert not any("Pages URL" in line for line in problems)


def test_readme_does_not_have_to_name_the_site(tmp_path):
    """The README names the site only after the site is deployed."""
    files = base_files()
    files["README.md"] = files["README.md"].replace(
        "The site is at https://team.github.io/entry/ and the code is at", "The code is at")
    assert "github.io" not in files["README.md"]
    # The site names its own address, as the site of this repository does.
    files["site/js/team.js"] = "const PAGES = 'https://team.github.io/entry/';\n"
    _kept, problems, _notes = export_and_check(tmp_path, files)
    assert problems == []


def test_check_of_a_git_repository_leaves_out_the_ignored_files(tmp_path):
    files = base_files()
    files[".gitignore"] = ".venv/\ncache/\n"
    repo = make_repo(tmp_path, files)
    out = tmp_path / "public"
    ep.export("HEAD", str(out), repo=str(repo))
    git(out, "init", "-q", "-b", "main")
    for name in (".venv/lib/data.npz", "cache/tracking.npz"):
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(b"x")
    assert ep.check(str(out), ep.top_level(str(repo)))[0] == []
    (out / "match.mp4").write_bytes(b"x")
    problems = ep.check(str(out), ep.top_level(str(repo)))[0]
    assert [line.split(":")[0] for line in problems] == ["match.mp4"]


def test_check_refuses_two_repository_urls(tmp_path):
    files = base_files()
    files["site/index.html"] += '<a href="https://github.com/other/entry">old</a>\n'
    _kept, problems, _notes = export_and_check(tmp_path, files)
    assert any("2 repository URLs" in line for line in problems)


def test_check_compares_the_urls_with_the_arguments(tmp_path):
    _kept, problems, _notes = export_and_check(
        tmp_path, base_files(), repo_url="https://github.com/team/entry",
        pages_url="https://team.github.io/entry/")
    assert problems == []
    problems = ep.check(str(tmp_path / "public"), ep.top_level(str(tmp_path / "private")),
                        repo_url="https://github.com/team/new-name",
                        pages_url="https://team.github.io/new-name/")[0]
    assert any("and not team/new-name" in line for line in problems)
    assert any("and not team.github.io/new-name" in line for line in problems)


def test_init_makes_one_commit_and_the_tag(tmp_path, monkeypatch):
    for key, value in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.invalid"),
                       ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@example.invalid")):
        monkeypatch.setenv(key, value)
    repo = make_repo(tmp_path, base_files())
    git(repo, "commit", "-q", "--allow-empty", "-m", "second")
    out = tmp_path / "public"
    ep.export("HEAD", str(out), repo=str(repo))
    commit = ep.init(str(out), "one commit")
    assert git(out, "rev-list", "--count", "HEAD").strip() == "1"
    assert git(out, "rev-parse", ep.PUBLIC_TAG + "^{commit}").strip() == commit
    assert git(out, "status", "--porcelain") == ""
    assert git(out, "branch", "--show-current").strip() == "main"
    with pytest.raises(ep.ExportError):
        ep.init(str(out), "again")


def test_init_writes_only_the_message_and_takes_the_identity_from_git(tmp_path, monkeypatch):
    """The one commit holds the message and nothing more. git gives the author and the committer.

    The tag is a lightweight tag: it has no message and no identity of its own.
    """
    for key, value in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.invalid"),
                       ("GIT_COMMITTER_NAME", "c"), ("GIT_COMMITTER_EMAIL", "c@example.invalid")):
        monkeypatch.setenv(key, value)
    repo = make_repo(tmp_path, base_files())
    out = tmp_path / "public"
    ep.export("HEAD", str(out), repo=str(repo))
    ep.init(str(out), "one commit")
    assert git(out, "log", "-1", "--format=%an <%ae>|%cn <%ce>|%B").strip() == \
        "t <t@example.invalid>|c <c@example.invalid>|one commit"
    assert git(out, "cat-file", "-t", ep.PUBLIC_TAG).strip() == "commit"


def test_main_exports_checks_and_returns_the_status(tmp_path, monkeypatch, capsys):
    repo = make_repo(tmp_path, base_files())
    monkeypatch.setattr(ep, "ROOT", str(repo))
    out = tmp_path / "public"
    assert ep.main(["--out", str(out)]) == 0
    assert "PASS" in capsys.readouterr().out
    assert ep.main(["--check", str(out)]) == 0
    (out / "README.md").write_text("See [the review](findings.md).\n", encoding="utf-8")
    assert ep.main(["--check", str(out)]) == 1
    # The link, the name of the dropped file, and the two missing repository URLs.
    assert "FAIL: 4 problems" in capsys.readouterr().out
    assert ep.main(["--out", str(out)]) == 2
    assert ep.main(["--list"]) == 0
    listed = capsys.readouterr().out.split("\n")
    assert "NOTES.md" in listed and "docs/plan/PLAN.md" in listed
    assert "engine/run.py" not in listed
    assert ep.main(["--engine-hash"]) == 0
    assert capsys.readouterr().out.strip() == ep.engine_source_hash(str(repo))


def test_load_terms_reads_the_list_and_skips_comments(tmp_path):
    """Without the file there is no term. Comments and empty lines are not terms."""
    assert ep.load_terms(str(tmp_path)) == []
    path = tmp_path / ep.TERMS_FILE
    path.parent.mkdir(parents=True)
    path.write_text("# a comment\n\nzqxwv\nfoo[0-9]+ bar\n", encoding="utf-8")
    terms = ep.load_terms(str(tmp_path))
    assert [t.pattern for t in terms] == ["zqxwv", "foo[0-9]+ bar"]
    assert terms[0].search("ZQXWV") and terms[1].search("Foo42 Bar")


def test_check_refuses_a_term_in_a_text_or_a_path(tmp_path):
    """The terms apply to code, records, entry points and paths. Synthetic terms only."""
    files = base_files()
    files["tools/note.py"] = "# a rule from the zqxwv notes\n"
    files["tools/zqxwv_run.md"] = "record\n"
    files["README.md"] += "Thanks to Zqxwv.\n"
    _kept, problems, _notes = export_and_check(tmp_path, files,
                                               terms=ep.compile_terms(["zqxwv"]))
    assert "tools/note.py:1: the text matches a term of {}: 'zqxwv'".format(
        ep.TERMS_FILE) in problems
    assert "tools/zqxwv_run.md: the path matches a term of {}".format(
        ep.TERMS_FILE) in problems
    assert "README.md:7: the text matches a term of {}: 'Zqxwv'".format(ep.TERMS_FILE) in problems


def test_the_terms_skip_the_organisers_files_and_an_empty_list_checks_nothing(tmp_path):
    files = base_files()
    files["evaluation/RULES.md"] = "zqxwv\n"
    files["site/vendor/lib.js"] = "// zqxwv\n"
    (tmp_path / "a").mkdir()
    _kept, problems, _notes = export_and_check(tmp_path / "a", files,
                                               terms=ep.compile_terms(["zqxwv"]))
    assert problems == []
    files["tools/note.py"] = "zqxwv\n"
    (tmp_path / "b").mkdir()
    _kept, problems, _notes = export_and_check(tmp_path / "b", files, terms=[])
    assert problems == []


def test_the_terms_apply_to_tests_and_to_the_site_bundle(tmp_path):
    """The site bundle is data, not prose, but its text reaches the public. Synthetic terms only."""
    files = base_files()
    files["tests/test_note.py"] = "# zqxwv\n"
    files["site/data/public/ladder.json"] = '{"note": "checked by zqxwv"}\n'
    _kept, problems, _notes = export_and_check(tmp_path, files,
                                               terms=ep.compile_terms(["zqxwv"]))
    flagged = {line.split(":")[0] for line in problems}
    assert flagged == {"tests/test_note.py", "site/data/public/ladder.json"}


def term_examples(root=ep.ROOT):
    """Return (refused, accepted): the example texts of TERMS_FILE under `root`, or None.

    A line "#+ text" gives a text that a term must refuse. A line "#- text" gives a text that no
    term may refuse. The examples stay in the list, which the copy leaves out.
    """
    path = os.path.join(root, ep.TERMS_FILE)
    if not os.path.isfile(path):
        return None
    refused, accepted = [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("#+ "):
                refused.append(line[3:])
            elif line.startswith("#- "):
                accepted.append(line[3:])
    return refused, accepted


def test_the_terms_of_this_checkout_refuse_and_accept_their_examples():
    """Each "#+" example matches a term. No "#-" example matches one."""
    examples = term_examples()
    if examples is None:
        pytest.skip("this is a public copy: the list of terms is not here")
    refused, accepted = examples
    assert len(refused) >= 20 and len(accepted) >= 10
    terms = ep.load_terms()
    for text in refused:
        assert any(t.search(text) for t in terms), text
    for text in accepted:
        assert not [t.pattern for t in terms if t.search(text)], text


def test_term_examples_reads_the_two_kinds_of_line(tmp_path):
    assert term_examples(str(tmp_path)) is None
    path = tmp_path / ep.TERMS_FILE
    path.parent.mkdir(parents=True)
    path.write_text("# comment\n#+ refuse me\n#- keep me\nzqxwv\n", encoding="utf-8")
    assert term_examples(str(tmp_path)) == (["refuse me"], ["keep me"])
    assert [t.pattern for t in ep.load_terms(str(tmp_path))] == ["zqxwv"]


def test_the_kept_files_of_this_checkout_pass_the_check():
    """The text of this checkout is valid for the public copy: an export of it prints PASS.

    The test reads the working tree, so it fails on a dead reference before the commit.
    """
    try:
        tracked = [p for p in ep.git(["ls-files"], ep.ROOT).split("\n") if p]
    except ep.ExportError:
        pytest.skip("this folder is not a git repository")
    kept = [p for p in ep.kept_paths(tracked) if os.path.isfile(os.path.join(ep.ROOT, p))]
    problems, _notes = ep.check(ep.ROOT, paths=kept)
    assert problems == []


def test_the_lists_match_the_tracked_files_of_this_checkout():
    """Each DROP entry names a tracked path, so no entry is stale."""
    try:
        tracked = [p for p in ep.git(["ls-files"], ep.ROOT).split("\n") if p]
    except ep.ExportError:
        pytest.skip("this folder is not a git repository")
    if ep.TERMS_FILE not in tracked:
        pytest.skip("this is a public copy: the dropped files are not here")
    assert any(ep.is_dropped(p) for p in tracked if "/" not in p and p not in ep.DROP)
    for entry in ep.DROP:
        assert any(p == entry or (entry.endswith("/") and p.startswith(entry)) for p in tracked), \
            entry
