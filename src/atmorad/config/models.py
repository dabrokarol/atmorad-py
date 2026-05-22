from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class MetadataConfig(BaseModel):
    experiment_name: str
    description: str = ""


class DetectorConfig(BaseModel):
    vertical_profiles_resolution_km: float = Field(gt=0.0)
    horizontal_maps_resolution_km: float = Field(gt=0.0)
    num_full_paths: int = Field(ge=0)
    incident_flux_heights_km: list[float] = Field(default_factory=list)


class OutputConfig(BaseModel):
    save_absorption_maps: bool = False
    save_incident_flux_maps: bool = False
    save_vertical_profiles: bool = False
    save_photon_paths: bool = False
    overwrite: bool = False
    save_plots: bool = False
    path: str


class EngineConfig(BaseModel):
    num_photons: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    random_seed: int
    cpu_cores: int = Field(ge=1)
    resume_from_checkpoint: bool = False


class SourceConfig(BaseModel):
    theta_sun_deg: float = Field(ge=0.0, le=180.0)
    phi_sun_deg: float = Field(ge=0.0, le=360.0)
    wavelength_nm: float = Field(gt=0.0)


class GeometryConfig(BaseModel):
    domain_size_x_km: float = Field(gt=0.0)
    domain_size_y_km: float = Field(gt=0.0)
    boundary_condition: Literal["periodic", "open"]


class EnvironmentConfig(BaseModel):
    atmosphere_materials: dict[str, Any]
    layers: list[dict[str, Any]] = Field(min_length=1)
    surface: dict[str, Any]
    surface_materials: dict[str, Any]
    geometry: GeometryConfig


class SimConfig(BaseModel):
    engine: EngineConfig
    source: SourceConfig
    output: OutputConfig
    metadata: MetadataConfig
    detectors: DetectorConfig
    environment: EnvironmentConfig
    config_path: Path | None = Field(default=None, exclude=True)

    def is_compatible_for_resume(self, checkpoint_config: "SimConfig") -> bool:
        ignored_engine_fields = {
            "num_photons", 
            "batch_size", 
            "cpu_cores", 
            "resume_from_checkpoint"
        }

        current_dict = self.model_dump(exclude={"engine": ignored_engine_fields})
        checkpoint_dict = checkpoint_config.model_dump(exclude={"engine": ignored_engine_fields})

        return current_dict == checkpoint_dict