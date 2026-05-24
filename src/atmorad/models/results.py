# Refactored using a large language model.

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import xarray as xr

from atmorad.registry import DETECTOR_RESULTS


class BaseResult(ABC):
    """
    Abstract base class for all detector results.
    Enforces the ability to merge results (for multiprocessing)
    and serialize/deserialize to/from xarray Datasets (for storage).
    """

    @abstractmethod
    def merge(self, other: Any) -> Any:
        pass

    @abstractmethod
    def to_dataset(self, prefix: str, n: int = 1, val_unit: str = "photons") -> xr.Dataset:
        """Packs the result data into an xarray Dataset."""
        pass

    @classmethod
    @abstractmethod
    def from_dataset(cls, ds: xr.Dataset, prefix: str) -> Self:
        """Reconstructs the result object from an xarray Dataset."""
        pass


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
class FateResult(BaseResult):
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

    def to_dataset(self, prefix: str, n: int = 1, val_unit: str = "photons") -> xr.Dataset:
        return xr.Dataset(
            attrs={
                f"{prefix}_energy_absorbed_surface": self.energy_absorbed_surface / n,
                f"{prefix}_energy_absorbed_atmosphere": self.energy_absorbed_atmosphere / n,
                f"{prefix}_energy_escaped_toa": self.energy_escaped_toa / n,
                f"{prefix}_cpu_time_s": self.cpu_time_s,
                f"{prefix}_energy_units": val_unit,
            }
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, prefix: str) -> Self:
        return cls(
            energy_absorbed_surface=float(ds.attrs.get(f"{prefix}_energy_absorbed_surface", 0.0)),
            energy_absorbed_atmosphere=float(
                ds.attrs.get(f"{prefix}_energy_absorbed_atmosphere", 0.0)
            ),
            energy_escaped_toa=float(ds.attrs.get(f"{prefix}_energy_escaped_toa", 0.0)),
            cpu_time_s=float(ds.attrs.get(f"{prefix}_cpu_time_s", 0.0)),
        )


@dataclass(slots=True)
class VerticalFluxResult(BaseResult):
    measure_z: np.ndarray
    flux_up: np.ndarray
    flux_down: np.ndarray

    def merge(self, other: Self):
        return VerticalFluxResult(
            measure_z=self.measure_z,
            flux_up=self.flux_up + other.flux_up,
            flux_down=self.flux_down + other.flux_down,
        )

    def to_dataset(self, prefix: str, n: int = 1, val_unit: str = "photons") -> xr.Dataset:
        dim_z = f"{prefix}_z"
        return xr.Dataset(
            coords={dim_z: (dim_z, self.measure_z, {"units": "km", "long_name": "Altitude"})},
            data_vars={
                f"{prefix}_flux_up": (
                    [dim_z],
                    self.flux_up / n,
                    {"units": val_unit, "long_name": "Upward Flux"},
                ),
                f"{prefix}_flux_down": (
                    [dim_z],
                    self.flux_down / n,
                    {"units": val_unit, "long_name": "Downward Flux"},
                ),
            },
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, prefix: str) -> Self:
        return cls(
            measure_z=ds.coords[f"{prefix}_z"].values,
            flux_up=ds[f"{prefix}_flux_up"].values,
            flux_down=ds[f"{prefix}_flux_down"].values,
        )


@dataclass(slots=True)
class AbsorptionProfileResult(BaseResult):
    z_centers: np.ndarray
    absorption_profile_1d: np.ndarray

    def merge(self, other: Self):
        return AbsorptionProfileResult(
            z_centers=self.z_centers,
            absorption_profile_1d=self.absorption_profile_1d + other.absorption_profile_1d,
        )

    def to_dataset(self, prefix: str, n: int = 1, val_unit: str = "photons") -> xr.Dataset:
        dim_z = f"{prefix}_center_z"
        return xr.Dataset(
            coords={dim_z: (dim_z, self.z_centers, {"units": "km", "long_name": "Altitude"})},
            data_vars={
                f"{prefix}_absorption_profile_1d": (
                    [dim_z],
                    self.absorption_profile_1d / n,
                    {"units": val_unit, "long_name": "Absorption Profile"},
                ),
            },
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, prefix: str) -> Self:
        return cls(
            z_centers=ds.coords[f"{prefix}_center_z"].values,
            absorption_profile_1d=ds[f"{prefix}_absorption_profile_1d"].values,
        )


@dataclass(slots=True)
class IncidentFluxMapResult(BaseResult):
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

    def to_dataset(self, prefix: str, n: int = 1, val_unit: str = "photons") -> xr.Dataset:
        dim_x, dim_y, dim_z = f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"
        return xr.Dataset(
            coords={
                dim_x: (dim_x, self.x_centers, {"units": "km", "long_name": "X Coordinate"}),
                dim_y: (dim_y, self.y_centers, {"units": "km", "long_name": "Y Coordinate"}),
                dim_z: (dim_z, self.measure_z, {"units": "km", "long_name": "Altitude"}),
            },
            data_vars={
                f"{prefix}_incident_flux_down_3d": (
                    [dim_z, dim_x, dim_y],
                    self.incident_flux_down_3d / n,
                    {"units": val_unit, "long_name": "Incident Downward Flux"},
                ),
                f"{prefix}_incident_flux_up_3d": (
                    [dim_z, dim_x, dim_y],
                    self.incident_flux_up_3d / n,
                    {"units": val_unit, "long_name": "Incident Upward Flux"},
                ),
            },
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, prefix: str) -> Self:
        return cls(
            x_centers=ds.coords[f"{prefix}_x"].values,
            y_centers=ds.coords[f"{prefix}_y"].values,
            measure_z=ds.coords[f"{prefix}_z"].values,
            incident_flux_down_3d=ds[f"{prefix}_incident_flux_down_3d"].values,
            incident_flux_up_3d=ds[f"{prefix}_incident_flux_up_3d"].values,
        )


@dataclass(slots=True)
class SurfaceAbsorptionResult(BaseResult):
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

    def to_dataset(self, prefix: str, n: int = 1, val_unit: str = "photons") -> xr.Dataset:
        dim_x, dim_y = f"{prefix}_x", f"{prefix}_y"
        return xr.Dataset(
            coords={
                dim_x: (dim_x, self.x_centers, {"units": "km", "long_name": "X Coordinate"}),
                dim_y: (dim_y, self.y_centers, {"units": "km", "long_name": "Y Coordinate"}),
            },
            data_vars={
                f"{prefix}_surface_absorption_map_2d": (
                    [dim_x, dim_y],
                    self.surface_absorption_map_2d / n,
                    {"units": val_unit, "long_name": "Surface Absorption"},
                ),
            },
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, prefix: str) -> Self:
        return cls(
            x_centers=ds.coords[f"{prefix}_x"].values,
            y_centers=ds.coords[f"{prefix}_y"].values,
            surface_absorption_map_2d=ds[f"{prefix}_surface_absorption_map_2d"].values,
        )


@dataclass(slots=True)
class PathTrackingResult(BaseResult):
    sample_paths_3d: np.ndarray
    sample_weights_2d: np.ndarray
    sample_escaped_toa: np.ndarray
    sample_absorbed_atmosphere: np.ndarray
    sample_absorbed_surface: np.ndarray
    toa_z: float

    def merge(self, other: Self):
        if len(self.sample_paths_3d) > 0:
            return self
        return other

    def to_dataset(self, prefix: str, n: int = 1, val_unit: str = "photons") -> xr.Dataset:
        attrs = {
            f"{prefix}_toa_z": self.toa_z,
            f"{prefix}_toa_z_units": "km",
        }

        if len(self.sample_paths_3d) == 0:
            attrs[f"{prefix}_empty_paths"] = 1
            return xr.Dataset(attrs=attrs)

        dim_p, dim_s, dim_c = f"{prefix}_photon", f"{prefix}_step", f"{prefix}_coord"
        return xr.Dataset(
            attrs=attrs,
            coords={
                dim_c: (dim_c, np.array(["x", "y", "z"]), {"long_name": "Spatial Dimension"}),
            },
            data_vars={
                f"{prefix}_sample_paths_3d": (
                    [dim_p, dim_s, dim_c],
                    self.sample_paths_3d,
                    {"units": "km", "long_name": "Photon Path Coordinates"},
                ),
                f"{prefix}_sample_weights_2d": (
                    [dim_p, dim_s],
                    self.sample_weights_2d,
                    {"units": "1", "long_name": "Photon Weight"},
                ),
                f"{prefix}_sample_escaped_toa": (
                    [dim_p],
                    self.sample_escaped_toa,
                    {"units": "boolean", "long_name": "Escaped TOA Flag"},
                ),
                f"{prefix}_sample_absorbed_atmosphere": (
                    [dim_p],
                    self.sample_absorbed_atmosphere,
                    {"units": "boolean", "long_name": "Absorbed in Atmosphere Flag"},
                ),
                f"{prefix}_sample_absorbed_surface": (
                    [dim_p],
                    self.sample_absorbed_surface,
                    {"units": "boolean", "long_name": "Absorbed at Surface Flag"},
                ),
            },
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, prefix: str) -> Self:
        toa_z = float(ds.attrs.get(f"{prefix}_toa_z", 0.0))

        if ds.attrs.get(f"{prefix}_empty_paths", False):
            return cls(
                sample_paths_3d=np.array([]),
                sample_weights_2d=np.array([]),
                sample_escaped_toa=np.array([]),
                sample_absorbed_atmosphere=np.array([]),
                sample_absorbed_surface=np.array([]),
                toa_z=toa_z,
            )

        return cls(
            sample_paths_3d=ds[f"{prefix}_sample_paths_3d"].values,
            sample_weights_2d=ds[f"{prefix}_sample_weights_2d"].values,
            sample_escaped_toa=ds[f"{prefix}_sample_escaped_toa"].values,
            sample_absorbed_atmosphere=ds[f"{prefix}_sample_absorbed_atmosphere"].values,
            sample_absorbed_surface=ds[f"{prefix}_sample_absorbed_surface"].values,
            toa_z=toa_z,
        )


# --- MASTER SIMULATION MODEL ---


@dataclass(slots=True)
class SimulationResults:
    engine: EngineResult = field(default_factory=EngineResult)
    detector_results: dict[str, BaseResult] = field(default_factory=dict)
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

    def to_dataset(self, normalize: bool = False) -> xr.Dataset:
        """Delegates dataset creation to all registered detector results and merges them."""
        datasets_to_merge = []
        n = self.num_photons if (normalize and self.num_photons > 0) else 1
        val_unit = "1" if normalize else "photons"

        base_ds = xr.Dataset(
            attrs={
                "engine_cpu_time_s": self.engine.cpu_time_s,
                "engine_simulation_time_s": self.engine.simulation_time_s,
            }
        )
        datasets_to_merge.append(base_ds)

        det_types = {}
        for det_id, det_res in self.detector_results.items():
            class_name = type(det_res).__name__
            det_types[det_id] = class_name

            det_ds = det_res.to_dataset(prefix=det_id, n=n, val_unit=val_unit)
            datasets_to_merge.append(det_ds)

        final_ds = xr.merge(datasets_to_merge)
        final_ds.attrs["_detector_types"] = json.dumps(det_types)

        return final_ds

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Reconstructs the full SimulationResults object dynamically using the registry."""
        engine = EngineResult(
            cpu_time_s=float(ds.attrs.get("engine_cpu_time_s", 0.0)),
            simulation_time_s=float(ds.attrs.get("engine_simulation_time_s", 0.0)),
        )

        detector_results = {}
        det_types = json.loads(str(ds.attrs.get("_detector_types", "{}")))

        for det_id, class_name in det_types.items():
            if class_name in DETECTOR_RESULTS:
                result_cls = DETECTOR_RESULTS[class_name]
                detector_results[det_id] = result_cls.from_dataset(ds, prefix=det_id)

        return cls(engine=engine, detector_results=detector_results)
