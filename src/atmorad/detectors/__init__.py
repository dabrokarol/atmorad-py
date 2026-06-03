__all__ = ["BaseDetector", "DETECTORS"]

from .base import BaseDetector
from .energy_budget import EnergyBudgetDetector
from .flux_maps import FluxMapsDetector
from .surface_absorption import SurfaceAbsorptionDetector
from .trajectories import PathTrackingDetector
from .vertical_absorption import AbsorptionProfileDetector
from .vertical_flux import VerticalFluxDetector

DETECTORS = {
    "energy_budget": EnergyBudgetDetector,
    "flux_profile": VerticalFluxDetector,
    "trajectories": PathTrackingDetector,
    "flux_maps": FluxMapsDetector,
    "surface_absorption_map": SurfaceAbsorptionDetector,
    "absorption_profile": AbsorptionProfileDetector,
}
