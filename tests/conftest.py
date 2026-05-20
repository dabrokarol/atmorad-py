from pathlib import Path

import pytest

from atmorad.builder import build_context


@pytest.fixture
def sim_context(request):
    filename = request.param
    config_path = Path(__file__).parent / filename
    return build_context(config_path)
