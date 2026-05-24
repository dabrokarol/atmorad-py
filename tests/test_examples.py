import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
EXAMPLE_FILES = list(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Skipping heavy examples execution in CI environment"
)
@pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_readme_examples(example_path):
    result = subprocess.run([sys.executable, str(example_path)], capture_output=True, text=True)

    assert result.returncode == 0
