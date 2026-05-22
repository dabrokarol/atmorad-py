from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from atmorad.registry import SURFACE_MAPS

# ------- Environment Config models: ----------

class SurfaceMaterialConfig(BaseModel):
    albedo: float = Field(ge=0.0, le=1.0)
    reflection: dict[str, Any]

    @model_validator(mode="after")
    def validate_reflection(self) -> "SurfaceMaterialConfig":
        if "type" not in self.reflection:
            raise ValueError("Surface reflection configuration must contain a 'type' key.")
        return self


class AtmosphereMaterialConfig(BaseModel):
    extinction_coeff_per_km: float = Field(ge=0.0)
    ssa: float = Field(ge=0.0, le=1.0)
    scattering: dict[str, Any]

    @model_validator(mode="after")
    def validate_scattering(self) -> "AtmosphereMaterialConfig":
        if "type" not in self.scattering:
            raise ValueError("Atmosphere scattering configuration must contain a 'type' key.")
        return self


class LayerMaterialConfig(BaseModel):
    type: str
    weight: float = Field(ge=0.0)


class LayerConfig(BaseModel):
    thickness_km: float = Field(gt=0.0)
    materials: list[LayerMaterialConfig] = Field(min_length=1)
    
# -----------------------------------------------------------
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
    atmosphere_materials: dict[str, AtmosphereMaterialConfig]
    layers: list[LayerConfig] = Field(min_length=1) 
    surface: dict[str, Any]
    surface_materials: dict[str, SurfaceMaterialConfig]
    geometry: GeometryConfig

    @model_validator(mode="after")
    def validate_environment(self) -> "EnvironmentConfig":
        map_name = self.surface.get("name")
        if not map_name or map_name not in SURFACE_MAPS:
            raise ValueError(f"Valid surface 'name' required. Available: {list(SURFACE_MAPS.keys())}")

        for key in SURFACE_MAPS[map_name]["material_keys"]:
            if key not in self.surface:
                raise ValueError(f"Map '{map_name}' requires key '{key}' in [surface].")
            if self.surface[key] not in self.surface_materials:
                raise ValueError(f"Material '{self.surface[key]}' not defined in [surface_materials].")

        for i, layer in enumerate(self.layers):
            for mat in layer.materials:
                if mat.type not in self.atmosphere_materials:
                    raise ValueError(
                        f"Layer {i + 1} references undefined atmosphere material: '{mat.type}'. "
                        f"Available materials: {list(self.atmosphere_materials.keys())}"
                    )
                
        return self

class SimConfig(BaseModel):
    engine: EngineConfig
    source: SourceConfig
    output: OutputConfig
    metadata: MetadataConfig
    detectors: DetectorConfig
    environment: EnvironmentConfig
    config_path: Path | None = Field(default=None, exclude=True)

    def is_compatible_for_resume(self, checkpoint_config: "SimConfig") -> bool:
        excluded_fields = {
            "engine": {"num_photons", "batch_size", "cpu_cores", "resume_from_checkpoint"}
        }
        
        
        current_dict = self.model_dump(exclude=excluded_fields)
        checkpoint_dict = checkpoint_config.model_dump(exclude=excluded_fields)
        
	 return current_dict == checkpoint_dict