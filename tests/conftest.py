from pathlib import Path

import pytest

from atmorad.config.parser import parse_config


@pytest.fixture
def sim_context(request):
    filename = request.param
    config_path = Path(__file__).parent / filename
    return parse_config(config_path)
