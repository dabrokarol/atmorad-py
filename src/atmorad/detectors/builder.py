from atmorad.detectors.heating import AtmosphericHeatingRateDetector
from atmorad.detectors.flux import VerticalFluxDetector
from atmorad.detectors.boundary_flux import BoundaryAbsorptionDetector
from atmorad.detectors.plane_flux import IncidentFluxMapDetector
from atmorad.detectors.paths import PathTrackingDetector
from atmorad.detectors.fate import FateDetector
from atmorad.config import SimConfig

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