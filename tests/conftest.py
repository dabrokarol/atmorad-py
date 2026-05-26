from pathlib import Path

import pytest

from atmorad.builder import build_context_list


@pytest.fixture
def sim_context_list(request):
    filename = request.param
    config_path = Path(__file__).parent / filename
    return build_context_list(config_path)
