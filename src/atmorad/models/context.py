from dataclasses import dataclass
from typing import Any

from atmorad.config import SimConfig

@dataclass
class SimContext:
    config: SimConfig
    scene: Any