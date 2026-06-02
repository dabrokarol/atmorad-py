__all__ = ["BaseDetector", "DETECTORS"]

from .base import BaseDetector
from .fate import FateDetector
from .paths import PathTrackingDetector
from .plane_flux import PlaneFluxDetector
from .surface_absorption import SurfaceAbsorptionDetector
from .vertical_absorption import AbsorptionProfileDetector
from .vertical_flux import VerticalFluxDetector

DETECTORS = {
    "fate": FateDetector,
    "vertical_flux": VerticalFluxDetector,
    "path_tracking": PathTrackingDetector,
    "plane_flux": PlaneFluxDetector,
    "surface_absorption": SurfaceAbsorptionDetector,
    "vertical_absorption": AbsorptionProfileDetector,
}
