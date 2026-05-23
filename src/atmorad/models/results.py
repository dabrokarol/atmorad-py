from dataclasses import dataclass, field
from typing import Self, Union

import numpy as np


@dataclass(slots=True)
class EngineResult:
    cpu_time_s: float = 0
    simulation_time_s: float = 0

    def merge(self, other: Self):
        return EngineResult(
            cpu_time_s=self.cpu_time_s + other.cpu_time_s,
            simulation_time_s=self.simulation_time_s + other.simulation_time_s,
        )


@dataclass(slots=True)
class FateResult:
    energy_absorbed_surface: float = 0.0
    energy_absorbed_atmosphere: float = 0.0
    energy_escaped_toa: float = 0.0
    cpu_time_s: float = 0.0

    def merge(self, other: Self):
        return FateResult(
            energy_absorbed_surface=self.energy_absorbed_surface + other.energy_absorbed_surface,
            energy_absorbed_atmosphere=self.energy_absorbed_atmosphere
            + other.energy_absorbed_atmosphere,
            energy_escaped_toa=self.energy_escaped_toa + other.energy_escaped_toa,
            cpu_time_s=self.cpu_time_s + other.cpu_time_s,
        )


@dataclass(slots=True)
class VerticalFluxResult:
    measure_z: np.ndarray
    flux_up: np.ndarray
    flux_down: np.ndarray

    def merge(self, other: Self):
        return VerticalFluxResult(
            measure_z=self.measure_z,
            flux_up=self.flux_up + other.flux_up,
            flux_down=self.flux_down + other.flux_down,
        )


@dataclass(slots=True)
class AbsorptionProfileResult:
    z_centers: np.ndarray
    absorption_profile_1d: np.ndarray

    def merge(self, other: Self):
        return AbsorptionProfileResult(
            z_centers=self.z_centers,
            absorption_profile_1d=self.absorption_profile_1d + other.absorption_profile_1d,
        )


@dataclass(slots=True)
class IncidentFluxMapResult:
    x_centers: np.ndarray
    y_centers: np.ndarray
    measure_z: np.ndarray
    incident_flux_down_3d: np.ndarray
    incident_flux_up_3d: np.ndarray

    def merge(self, other: Self):
        return IncidentFluxMapResult(
            x_centers=self.x_centers,
            y_centers=self.y_centers,
            measure_z=self.measure_z,
            incident_flux_down_3d=self.incident_flux_down_3d + other.incident_flux_down_3d,
            incident_flux_up_3d=self.incident_flux_up_3d + other.incident_flux_up_3d,
        )


@dataclass(slots=True)
class SurfaceAbsorptionResult:
    x_centers: np.ndarray
    y_centers: np.ndarray
    surface_absorption_map_2d: np.ndarray

    def merge(self, other: Self):
        return SurfaceAbsorptionResult(
            x_centers=self.x_centers,
            y_centers=self.y_centers,
            surface_absorption_map_2d=self.surface_absorption_map_2d
            + other.surface_absorption_map_2d,
        )


@dataclass(slots=True)
class PathTrackingResult:
    sample_paths_3d: np.ndarray
    sample_weights_2d: np.ndarray
    sample_escaped_toa: np.ndarray
    sample_absorbed_atmosphere: np.ndarray
    sample_absorbed_surface: np.ndarray
    toa_z: float

    def merge(self, other: Self):  # returns result for only one non-empty batch
        if len(self.sample_paths_3d) > 0:
            return self
        return other


# --- MASTER SIMULATION MODEL ---

AnyDetectorResult = Union[
    FateResult,
    VerticalFluxResult,
    AbsorptionProfileResult,
    IncidentFluxMapResult,
    SurfaceAbsorptionResult,
    PathTrackingResult,
]


@dataclass(slots=True)
class SimulationResults:
    engine: EngineResult = field(default_factory=EngineResult)
    detector_results: dict[str, AnyDetectorResult] = field(default_factory=dict)
    num_photons: int = 0

    def merge(self, other: Self):
        merged_detectors = {}
        all_keys = set(self.detector_results.keys()).union(other.detector_results.keys())

        for key in all_keys:
            if key in self.detector_results and key in other.detector_results:
                merged_detectors[key] = self.detector_results[key].merge(
                    other.detector_results[key]
                )
            elif key in self.detector_results:
                merged_detectors[key] = self.detector_results[key]
            else:
                merged_detectors[key] = other.detector_results[key]

        return SimulationResults(
            engine=self.engine.merge(other.engine),
            detector_results=merged_detectors,
            num_photons=self.num_photons + other.num_photons,
        )
