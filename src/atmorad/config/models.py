import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atmorad.constants import TIMESTAMP_FORMAT
from atmorad.registry import (
    DETECTORS,
    REFLECTION_MODELS,
    SCATTERING_MODELS,
    SURFACE_MAPS,
)

# ------- Environment Config models: ----------


class SurfaceMaterialConfig(BaseModel):
    albedo: float = Field(ge=0.0, le=1.0)
    reflection: dict[str, Any]

    @model_validator(mode="after")
    def validate_reflection(self) -> "SurfaceMaterialConfig":
        if "type" not in self.reflection:
            raise ValueError("Surface reflection configuration must contain a 'type' key.")
        if self.reflection["type"] not in REFLECTION_MODELS:
            raise ValueError(
                f"Surface reflection model not found: {self.reflection['type']}. "
                f"Available: {list(REFLECTION_MODELS.keys())}"
            )
        return self


class AtmosphereMaterialConfig(BaseModel):
    extinction_coeff_per_km: float = Field(ge=0.0)
    ssa: float = Field(ge=0.0, le=1.0)
    scattering: dict[str, Any]

    @model_validator(mode="after")
    def validate_scattering(self) -> "AtmosphereMaterialConfig":
        if "type" not in self.scattering:
            raise ValueError("Atmosphere scattering configuration must contain a 'type' key.")
        if self.scattering["type"] not in SCATTERING_MODELS:
            raise ValueError(
                f"Scattering model not found: {self.scattering['type']}. "
                f"Available: {list(SCATTERING_MODELS.keys())}"
            )
        return self


class LayerMaterialConfig(BaseModel):
    material: str
    concentration: float = Field(ge=0.0, le=1.0)


class LayerConfig(BaseModel):
    thickness_km: float = Field(gt=0.0)
    components: list[LayerMaterialConfig] = Field(min_length=1)


# -----------------------------------------------------------


def get_engine_version() -> str:
    try:
        return version("atmorad-py")
    except PackageNotFoundError:
        return "unknown-dev"


def generate_timestamp() -> str:
    return datetime.datetime.now().strftime(TIMESTAMP_FORMAT)


class MetadataConfig(BaseModel):
    experiment_name: str = "experiment"
    scenario_name: str = "baseline"
    description: str = ""
    config_version: str = "1.2"
    software_version: str = Field(default_factory=get_engine_version)
    run_timestamp: str = Field(default_factory=generate_timestamp)


class DetectorConfig(BaseModel):
    active: list[str] = Field(
        default=["fate"], description="List of active detectors to use during the simulation."
    )

    vertical_profiles_resolution_km: float = Field(gt=0.0, default=1.0)
    horizontal_maps_resolution_km: float = Field(gt=0.0, default=1.0)
    num_full_paths: int = Field(ge=0, default=0)
    flux_maps_z_levels_km: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_detectors(self) -> "DetectorConfig":
        for det in self.active:
            if det not in DETECTORS.keys():
                raise ValueError(f"Unknown detector: '{det}'. Available: {list(DETECTORS.keys())}")
        return self


class OutputConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    overwrite: bool = False
    save_plots: bool = Field(
        default=True,
        description="If true, generates and saves standard PNG plots for all active default detectors.",
    )
    base_dir: Path = Path("results")
    fig_dir: Path = Path("plots")


class EngineConfig(BaseModel):
    num_photons: int = Field(gt=0, default=100_000)
    batch_size: int = Field(gt=0, default=100_000)
    random_seed: int = 42
    cpu_cores: int = Field(ge=1, default=4)
    resume_from_checkpoint: bool = False
    photon_weight_threshold: float = Field(ge=0, default=1e-4)
    photon_survival_chance: float = Field(default=0.1, ge=0.0, le=1.0)


class SourceConfig(BaseModel):
    theta_sun_deg: float = Field(ge=0.0, le=180.0, default=0)
    phi_sun_deg: float = Field(ge=0.0, le=360.0, default=0)
    wavelength_nm: float = Field(ge=0.0, default=0)


class GeometryConfig(BaseModel):
    domain_size_x_km: float = Field(gt=0.0, default=100)
    domain_size_y_km: float = Field(gt=0.0, default=100)
    boundary_condition: Literal["periodic", "open"] = "periodic"


class EnvironmentConfig(BaseModel):
    layers: list[LayerConfig] = Field(min_length=1)
    surface: dict[str, Any]
    geometry: GeometryConfig
    atmosphere_materials: dict[str, AtmosphereMaterialConfig] = Field(default_factory=dict)
    surface_materials: dict[str, SurfaceMaterialConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_environment(self) -> "EnvironmentConfig":
        map_name = self.surface.get("name")
        if not map_name or map_name not in SURFACE_MAPS:
            raise ValueError(
                f"Valid surface 'name' required. Available: {list(SURFACE_MAPS.keys())}"
            )

        for key in SURFACE_MAPS[map_name]["material_keys"]:
            if key not in self.surface:
                raise ValueError(f"Map '{map_name}' requires key '{key}' in [surface].")
            if self.surface[key] not in self.surface_materials:
                raise ValueError(
                    f"Material '{self.surface[key]}' not defined in [surface_materials]."
                )

        for i, layer in enumerate(self.layers):
            for comp in layer.components:
                if comp.material not in self.atmosphere_materials:
                    raise ValueError(
                        f"Layer {i + 1} references undefined atmosphere material: '{comp.material}'. "
                        f"Available materials: {list(self.atmosphere_materials.keys())}"
                    )

        return self


class SimConfig(BaseModel):
    engine: EngineConfig = Field(default_factory=EngineConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    detectors: DetectorConfig = Field(default_factory=DetectorConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    environment: EnvironmentConfig
    config_path: Path | None = Field(default=None, exclude=True)

    def is_compatible_for_resume(self, checkpoint_config: "SimConfig") -> bool:
        excluded_fields = {"metadata", "output"}

        current_dict = self.model_dump(exclude=excluded_fields)
        checkpoint_dict = checkpoint_config.model_dump(exclude=excluded_fields)

        return current_dict == checkpoint_dict

    def get_experiment_attributes(self) -> dict[str, float | int | str]:
        """Returns attributes that will be saved to at root in netCDF file."""
        return {
            "experiment_name": self.metadata.experiment_name,
            "scenario_name": self.metadata.scenario_name,
            "domain_size_x_km": self.environment.geometry.domain_size_x_km,
            "domain_size_y_km": self.environment.geometry.domain_size_y_km,
            "boundary_condition": self.environment.geometry.boundary_condition,
            "vertical_resolution_km": self.detectors.vertical_profiles_resolution_km,
            "source_theta_sun_deg": self.source.theta_sun_deg,
            "source_phi_sun_deg": self.source.phi_sun_deg,
        }
