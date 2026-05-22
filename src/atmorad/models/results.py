from typing import Self, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def merge(self, other: Self) -> Self:
        raise NotImplementedError


class EngineResult(BaseResult):
    cpu_time_s: float = Field(default=0.0, ge=0.0)
    simulation_time_s: float = Field(default=0.0, ge=0.0)

    def merge(self, other: Self) -> Self:
        return self.__class__(
            cpu_time_s=self.cpu_time_s + other.cpu_time_s,
            simulation_time_s=self.simulation_time_s + other.simulation_time_s,
        )


# --- DETECTOR RESULTS ---


class FateResult(BaseResult):
    photons_absorbed_surface: int = 0
    photons_absorbed_atmosphere: int = 0
    photons_escaped_toa: int = 0
    cpu_time_s: float = 0.0

    def merge(self, other: Self) -> Self:
        return self.__class__(
            photons_absorbed_surface=self.photons_absorbed_surface + other.photons_absorbed_surface,
            photons_absorbed_atmosphere=self.photons_absorbed_atmosphere
            + other.photons_absorbed_atmosphere,
            photons_escaped_toa=self.photons_escaped_toa + other.photons_escaped_toa,
            cpu_time_s=self.cpu_time_s + other.cpu_time_s,
        )


class VerticalFluxResult(BaseResult):
    measure_z: np.ndarray
    flux_up: np.ndarray
    flux_down: np.ndarray

    def merge(self, other: Self) -> Self:
        return self.__class__(
            measure_z=self.measure_z,
            flux_up=self.flux_up + other.flux_up,
            flux_down=self.flux_down + other.flux_down,
        )


class AbsorptionProfileResult(BaseResult):
    measure_z: np.ndarray
    absorption_profile_1d: np.ndarray

    def merge(self, other: Self) -> Self:
        return self.__class__(
            measure_z=self.measure_z,
            absorption_profile_1d=self.absorption_profile_1d + other.absorption_profile_1d,
        )


class IncidentFluxMapResult(BaseResult):
    x_edges: np.ndarray
    y_edges: np.ndarray
    flux_maps_z_levels_km: np.ndarray
    incident_flux_down_maps_2d: dict[float, np.ndarray]
    incident_flux_up_maps_2d: dict[float, np.ndarray]

    def merge(self, other: Self) -> Self:
        merged_down = {
            z: self.incident_flux_down_maps_2d[z] + other.incident_flux_down_maps_2d[z]
            for z in self.incident_flux_down_maps_2d
        }
        merged_up = {
            z: self.incident_flux_up_maps_2d[z] + other.incident_flux_up_maps_2d[z]
            for z in self.incident_flux_up_maps_2d
        }
        return self.__class__(
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            flux_maps_z_levels_km=self.flux_maps_z_levels_km,
            incident_flux_down_maps_2d=merged_down,
            incident_flux_up_maps_2d=merged_up,
        )


class BoundaryAbsorptionResult(BaseResult):
    x_edges: np.ndarray
    y_edges: np.ndarray
    surface_absorption_map_2d: np.ndarray
    toa_flux_map_2d: np.ndarray

    def merge(self, other: Self) -> Self:
        return self.__class__(
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            surface_absorption_map_2d=self.surface_absorption_map_2d
            + other.surface_absorption_map_2d,
            toa_flux_map_2d=self.toa_flux_map_2d + other.toa_flux_map_2d,
        )


class PathTrackingResult(BaseResult):
    sample_paths: dict[int, list[np.ndarray]]
    sample_escaped_toa: dict[int, bool]
    sample_absorbed_atmosphere: dict[int, bool]
    sample_absorbed_surface: dict[int, bool]
    toa_z: float

    def merge(self, other: Self) -> Self:
        return self.__class__(
            sample_paths={**self.sample_paths, **other.sample_paths},
            sample_escaped_toa={**self.sample_escaped_toa, **other.sample_escaped_toa},
            sample_absorbed_atmosphere={
                **self.sample_absorbed_atmosphere,
                **other.sample_absorbed_atmosphere,
            },
            sample_absorbed_surface={
                **self.sample_absorbed_surface,
                **other.sample_absorbed_surface,
            },
            toa_z=self.toa_z,
        )


# --- MASTER SIMULATION MODEL ---

AnyDetectorResult = Union[
    FateResult,
    VerticalFluxResult,
    AbsorptionProfileResult,
    IncidentFluxMapResult,
    BoundaryAbsorptionResult,
    PathTrackingResult,
]


class SimulationResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine: EngineResult = EngineResult()
    detectors: dict[str, AnyDetectorResult] = {}

    @model_validator(mode="before")
    @classmethod
    def _parse_detectors(cls, data):
        """Correctly maps the raw dictionary back into the right Pydantic classes when loading from NetCDF."""
        if isinstance(data, dict) and "detectors" in data:
            dets = data["detectors"]
            if isinstance(dets, dict):
                parsed_dets = {}
                for k, v in dets.items():
                    if not isinstance(v, dict):
                        parsed_dets[k] = v
                        continue

                    # Explicitly match the key to the specific class
                    if k in ("fate", "FateDetector"):
                        parsed_dets[k] = FateResult(**v)
                    elif k in ("vertical_flux", "VerticalFluxDetector"):
                        parsed_dets[k] = VerticalFluxResult(**v)
                    elif k in ("absorption_vertical", "AbsorptionProfileDetector"):
                        parsed_dets[k] = AbsorptionProfileResult(**v)
                    elif k in ("plane_flux", "IncidentFluxMapDetector"):
                        parsed_dets[k] = IncidentFluxMapResult(**v)
                    elif k in ("boundary_flux", "BoundaryAbsorptionDetector"):
                        parsed_dets[k] = BoundaryAbsorptionResult(**v)
                    elif k in ("path_tracking", "PathTrackingDetector"):
                        parsed_dets[k] = PathTrackingResult(**v)
                    else:
                        parsed_dets[k] = v
                data["detectors"] = parsed_dets
        return data

    def merge(self, other: Self) -> Self:
        merged_detectors = {}
        all_keys = set(self.detectors.keys()).union(other.detectors.keys())

        for key in all_keys:
            if key in self.detectors and key in other.detectors:
                merged_detectors[key] = self.detectors[key].merge(other.detectors[key])
            elif key in self.detectors:
                merged_detectors[key] = self.detectors[key]
            else:
                merged_detectors[key] = other.detectors[key]

        return self.__class__(engine=self.engine.merge(other.engine), detectors=merged_detectors)
