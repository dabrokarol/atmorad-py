from dataclasses import dataclass

from atmorad.environment.scene import Scene 

@dataclass
class MetadataConfig:
    experiment_name: str
    description: str
    
@dataclass
class DetectorConfig:
    vertical_flux_resolution_km: float
    map2d_resolution_km: float
    num_full_paths: int
    incident_flux_heights_km: list[float]
    
@dataclass
class OutputConfig:
    save_absorption_maps: bool
    save_incident_flux_maps: bool
    save_vertical_profile: bool
    save_heating_rates: bool
    save_photon_paths: bool
    overwrite: bool
    save_plots: bool
    path: str
    
@dataclass
class EngineConfig:
    num_photons: int
    batch_size: int
    random_seed: int
    cpu_cores: int

@dataclass
class SourceConfig:
    theta_sun_deg: float
    phi_sun_deg: float
    wavelength_nm: float

@dataclass
class GeometryConfig:
    domain_size_x_km: float
    domain_size_y_km: float
    boundary_condition: str
    
@dataclass
class SimConfig:
    engine: EngineConfig
    source: SourceConfig
    geometry: GeometryConfig
    output: OutputConfig
    metadata: MetadataConfig
    detectors: DetectorConfig
    
@dataclass
class SimContext:
    config: SimConfig
    scene: Scene