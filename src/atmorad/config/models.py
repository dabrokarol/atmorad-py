from dataclasses import dataclass


@dataclass
class MetadataConfig:
    experiment_name: str
    description: str


@dataclass
class DetectorConfig:
    vertical_profiles_resolution_km: float
    horizontal_maps_resolution_km: float
    num_full_paths: int
    incident_flux_heights_km: list[float]


@dataclass
class OutputConfig:
    save_absorption_maps: bool
    save_incident_flux_maps: bool
    save_vertical_profiles: bool
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
    resume_from_checkpoint: bool


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
class EnvironmentConfig:
    atmosphere_materials: dict
    layers: list[dict]
    surface: dict
    surface_materials: dict
    geometry: GeometryConfig


@dataclass
class SimConfig:
    engine: EngineConfig
    source: SourceConfig
    output: OutputConfig
    metadata: MetadataConfig
    detectors: DetectorConfig
    environment: EnvironmentConfig
