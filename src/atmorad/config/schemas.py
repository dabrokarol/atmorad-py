import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atmorad.constants import TIMESTAMP_FORMAT


def get_engine_version() -> str:
    try:
        return version("atmorad-py")
    except PackageNotFoundError:
        return "unknown-dev"


def generate_timestamp() -> str:
    return datetime.datetime.now().strftime(TIMESTAMP_FORMAT)


class MetadataConfig(BaseModel):
    experiment_name: str = "simulation"
    scenario_name: str = "baseline"
    description: str = ""
    config_version: str = "2.0"
    software_version: str = Field(default_factory=get_engine_version)
    run_timestamp: str = Field(default_factory=generate_timestamp)


class EngineConfig(BaseModel):
    random_seed: int = -1
    num_photons: int = Field(gt=0, default=100_000)
    batch_size: int = Field(gt=0, default=100_000)
    num_threads: int = Field(ge=1, default=4)
    resume_from_checkpoint: bool = False
    roulette_weight_threshold: float = Field(ge=0, default=1e-4)
    roulette_survival_probability: float = Field(default=0.1, ge=0.0, le=1.0)


class SourceConfig(BaseModel):
    zenith_angle_deg: float = Field(ge=0.0, le=180.0, default=0)
    azimuth_angle_deg: float = Field(ge=0.0, le=360.0, default=0)
    wavelength_nm: float = Field(ge=0.0, default=0)


class DomainConfig(BaseModel):
    size_x_km: float = Field(gt=0.0, default=100)
    size_y_km: float = Field(gt=0.0, default=100)
    boundary_condition: Literal["periodic", "open"] = "periodic"


class OutputConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    overwrite: bool = False
    save_plots: bool = True
    base_dir: Path = Path("results")
    fig_dir: Path = Path("plots")


# --- detectors ---


class TrajectoriesConfig(BaseModel):
    max_tracked_paths: int = Field(ge=0, default=200)


class FluxProfileConfig(BaseModel):
    vertical_resolution_km: float = Field(gt=0.0, default=1.0)


class AbsorptionProfileConfig(BaseModel):
    vertical_resolution_km: float = Field(gt=0.0, default=1.0)


class FluxMapsConfig(BaseModel):
    horizontal_resolution_km: float = Field(gt=0.0, default=1.0)
    z_levels_km: list[float] = Field(default_factory=list)


class SurfaceAbsorptionMapConfig(BaseModel):
    horizontal_resolution_km: float = Field(gt=0.0, default=1.0)


class DetectorsConfig(BaseModel):
    energy_budget: Any | None = None
    trajectories: TrajectoriesConfig | None = None
    flux_profile: FluxProfileConfig | None = None
    absorption_profile: AbsorptionProfileConfig | None = None
    flux_maps: FluxMapsConfig | None = None
    surface_absorption_map: SurfaceAbsorptionMapConfig | None = None
    active: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def determine_active_detectors(self) -> "DetectorsConfig":
        self.active = [
            field_name
            for field_name in self.__dict__.keys()
            if field_name != "active" and getattr(self, field_name) is not None
        ]
        return self


# --- Environment Materials & Layers ---


class SurfaceMaterialConfig(BaseModel):
    albedo: float = Field(ge=0.0, le=1.0)
    brdf: dict[str, Any]

    @model_validator(mode="after")
    def validate_brdf(self) -> "SurfaceMaterialConfig":
        if "type" not in self.brdf:
            raise ValueError("Surface BRDF configuration must contain a 'type' key.")
        return self


class AtmosphereMaterialConfig(BaseModel):
    extinction_coeff_per_km: float = Field(ge=0.0)
    ssa: float = Field(ge=0.0, le=1.0)
    phase_function: dict[str, Any]

    @model_validator(mode="after")
    def validate_scattering(self) -> "AtmosphereMaterialConfig":
        if "type" not in self.phase_function:
            raise ValueError("Phase function configuration must contain a 'type' key.")
        return self


class LayerConfig(BaseModel):
    thickness_km: float = Field(gt=0.0)
    components: dict[str, float]


class SimConfig(BaseModel):
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    engine: EngineConfig
    source: SourceConfig = Field(default_factory=SourceConfig)
    domain: DomainConfig
    detectors: DetectorsConfig = Field(default_factory=DetectorsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    atmosphere_materials: dict[str, AtmosphereMaterialConfig] = Field(default_factory=dict)
    surface_materials: dict[str, SurfaceMaterialConfig] = Field(default_factory=dict)
    layers: list[LayerConfig] = Field(default_factory=list)
    surface: dict[str, Any]

    config_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def handle_aliases(cls, data: Any) -> Any:
        """Mapuje klucz 'layer' z pliku TOML na pole 'layers' w modelu Pydantic."""
        if isinstance(data, dict):
            if "layer" in data and "layers" not in data:
                data["layers"] = data.pop("layer")
        return data

    @model_validator(mode="after")
    def validate_relations(self) -> "SimConfig":
        if "type" not in self.surface:
            raise ValueError("Surface configuration must contain a 'type' key.")

        for i, layer in enumerate(self.layers):
            for mat_name in layer.components.keys():
                if mat_name not in self.atmosphere_materials:
                    raise ValueError(
                        f"Layer {i + 1} references undefined atmosphere material: '{mat_name}'. "
                        f"Available: {list(self.atmosphere_materials.keys())}"
                    )
        return self

    def is_compatible_for_resume(self, checkpoint_config: "SimConfig") -> bool:
        currend_dict = self.model_dump(exclude={"metadata", "output"})
        old_dict = checkpoint_config.model_dump(exclude={"metadata", "output"})

        current_seed = currend_dict["engine"].pop("random_seed")
        old_seed = old_dict["engine"].pop("random_seed")

        return currend_dict == old_dict and (current_seed == -1 or current_seed == old_seed)

    def get_experiment_attributes(self) -> dict[str, float | int | str]:
        return {
            "experiment_name": self.metadata.experiment_name,
            "scenario_name": self.metadata.scenario_name,
            "domain_size_x_km": self.domain.size_x_km,
            "domain_size_y_km": self.domain.size_y_km,
            "boundary_condition": self.domain.boundary_condition,
            "source_zenith_angle_deg": self.source.zenith_angle_deg,
            "source_azimuth_angle_deg": self.source.azimuth_angle_deg,
        }
