"""The fresh-clone reproduction script exists, can run and prints its usage (card P5).

The test does not run the clone. The real run needs the data bundle and takes about 45 s.
"""

import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "repro_fresh_clone.sh")


def test_repro_script_is_executable_and_prints_usage():
    assert os.path.isfile(SCRIPT)
    assert os.access(SCRIPT, os.X_OK)
    result = subprocess.run([SCRIPT], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    assert "Usage: tools/repro_fresh_clone.sh EXPECTED_MD5" in result.stdout
    assert result.returncode == 2  # a call without arguments is a usage error
