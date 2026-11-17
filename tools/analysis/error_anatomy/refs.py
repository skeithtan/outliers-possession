"""Prepare the eight OPEN references (md5-checked by external_sweep.prepare_reference) into $ERROR_ANATOMY_WORK/refs."""
import os, sys, json
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (REPO, os.path.join(REPO, "tools"), os.path.join(REPO, "tools", "analysis"), os.path.join(REPO, "evaluation")):
    sys.path.insert(0, p)
import external_sweep as es
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("ERROR_ANATOMY_WORK", os.path.join("/tmp", "error_anatomy"))  # outputs, never the repo
OPEN = es.OPEN_GAMES

def ref_paths():
    refs_root = os.path.expanduser(es.DEFAULT_REFS_ROOT)
    sample_root = os.path.expanduser(es.DEFAULT_SAMPLE_ROOT)
    out = {}
    for g in OPEN:
        assert g not in ("BAR-ATH", "DFL-J03WQQ", "DFL-J03WR9")
        path, md5 = es.prepare_reference(g, refs_root, sample_root, WORK)
        out[g] = (path, md5)
    return out

if __name__ == "__main__":
    for g, (p, m) in ref_paths().items():
        print(g, p, m)
