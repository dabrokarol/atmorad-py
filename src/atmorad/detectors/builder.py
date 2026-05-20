from atmorad.config import SimConfig
from .boundary_flux import BoundaryAbsorptionDetector
from .fate import FateDetector
from .flux import VerticalFluxDetector
from .heating import AtmosphericHeatingRateDetector
from .paths import PathTrackingDetector
from .plane_flux import IncidentFluxMapDetector


def build_detectors_from_config(config: SimConfig):
    detectors = []

    detectors.append(FateDetector())

    if config.detectors.num_full_paths > 0:
        detectors.append(PathTrackingDetector())

    if config.output.save_vertical_profile:
        detectors.append(VerticalFluxDetector())

    if config.output.save_absorption_maps:
        detectors.append(BoundaryAbsorptionDetector())

    if config.output.save_incident_flux_maps:
        detectors.append(IncidentFluxMapDetector())

    if config.output.save_heating_rates:
        detectors.append(AtmosphericHeatingRateDetector())

    return detectors
