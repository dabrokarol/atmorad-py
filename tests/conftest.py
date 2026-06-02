from pathlib import Path

import pytest

from atmorad.config import load_scenarios


@pytest.fixture
def config_list(request):
    filename = request.param
    config_path = Path(__file__).parent / filename
    return load_scenarios(config_path)
