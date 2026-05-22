from atmorad.config import SimConfig
from atmorad.registry import DETECTORS

from .fate import FateDetector


def build_detectors_from_config(config: SimConfig):
    detectors = []

    detectors.append(FateDetector())
    for det_name in config.detectors.active:
        DetectorClass = DETECTORS[det_name]
        detectors.append(DetectorClass())
    return detectors
