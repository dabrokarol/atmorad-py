__all__ = ["BaseDetector", "DETECTORS"]

from .absorption_profile import AbsorptionProfileDetector
from .base import BaseDetector
from .energy_budget import EnergyBudgetDetector
from .flux_maps import FluxMapsDetector
from .flux_profile import VerticalFluxDetector
from .surface_absorption_map import SurfaceAbsorptionDetector
from .trajectories import PathTrackingDetector

DETECTORS: dict[str, type[BaseDetector]] = {
    "energy_budget": EnergyBudgetDetector,
    "flux_profile": VerticalFluxDetector,
    "trajectories": PathTrackingDetector,
    "flux_maps": FluxMapsDetector,
    "surface_absorption_map": SurfaceAbsorptionDetector,
    "absorption_profile": AbsorptionProfileDetector,
}
