# refactored using LLMs

import json
from abc import ABC
from dataclasses import dataclass, field, fields
from typing import Self

import numpy as np
import xarray as xr

from atmorad.registry import DETECTOR_RESULTS


def coord_field(nc_name: str, units: str, long_name: str, on_merge: str | None = None):
    """Creates a coordinate field for simulation results.

    Args:
        nc_name: Name of the coordinate in the NetCDF file.
        units: Physical units of the coordinate (e.g., 'km').
        long_name: Descriptive name for the NetCDF attribute.
        on_merge: Merge strategy ('keep', 'sum', 'concat'). Defaults to 'keep' for coordinates.
    """
    metadata = {"role": "coord", "nc_name": nc_name, "units": units, "long_name": long_name}
    if on_merge is not None:
        metadata["on_merge"] = on_merge

    return field(metadata=metadata)


def data_field(
    dims: list[str],
    normalize: bool = False,
    long_name: str = "",
    units: str | None = None,
    on_merge: str | None = None,
):
    """Creates a data field (e.g., a results array) for simulation results.

    Args:
        dims: List of dimension names matching the coordinate nc_names (e.g., ['z', 'x']).
        normalize: Whether to divide the results by the number of simulated photons.
        long_name: Descriptive name for the NetCDF attribute.
        units: Physical units. Overrides default normalization units if provided.
        on_merge: Merge strategy ('keep', 'sum', 'concat'). Defaults to 'sum' for data.
    """
    metadata = {"role": "data", "dims": dims, "normalize": normalize, "long_name": long_name}
    if units is not None:
        metadata["units"] = units
    if on_merge is not None:
        metadata["on_merge"] = on_merge

    return field(metadata=metadata)


def attr_field(normalize: bool = False, units: str | None = None):
    """Creates an attribute field (e.g., scalar values, summaries).

    Args:
        normalize: Whether to divide the attribute by the number of simulated photons.
        units: Physical units of the attribute.
    """
    metadata = {"role": "attr", "normalize": normalize}
    if units is not None:
        metadata["units"] = units

    return field(metadata=metadata)


@dataclass
class BaseResult(ABC):
    """
    Automated base class for Monte Carlo simulation results (atmorad).
    Serialization, deserialization, and merging are driven by @dataclass metadata.
    """

    def merge(self, other: "BaseResult") -> Self:
        """
        Dynamically merges two results based on the 'on_merge' metadata.
        Available operations: 'sum' (default for 'data'), 'keep' (default for 'coord' and 'attr'), 'concat'.
        """
        assert type(self) is type(other), f"Type mismatch: {type(self)} != {type(other)}"

        kwargs = {}
        for f in fields(self):
            v1, v2 = getattr(self, f.name), getattr(other, f.name)

            role = f.metadata.get("role", "attr")
            # By default, sum data variables, and keep the first value for coordinates and attributes
            op = f.metadata.get("on_merge", "keep" if role in ("coord", "attr") else "sum")

            if op == "sum":
                kwargs[f.name] = v1 + v2

            elif op == "concat":
                if isinstance(v1, np.ndarray) and v1.size == 0:
                    kwargs[f.name] = v2
                else:
                    kwargs[f.name] = np.concatenate((v1, v2))

            else:  # "keep"
                kwargs[f.name] = v1

        return type(self)(**kwargs)

    def to_dataset(self, prefix: str, n_photons: int = 1, val_unit: str = "fraction") -> xr.Dataset:
        """Converts dataclass fields into an xarray.Dataset object using metadata."""
        attrs = {"energy_units": val_unit}
        coords = {}
        data_vars = {}

        for f in fields(self):
            meta = f.metadata
            if not meta:
                continue

            nc_name = meta.get("nc_name", f.name)
            role = meta.get("role", "attr")
            val = getattr(self, f.name)

            normalize = meta.get("normalize", False)
            units = meta.get("units", val_unit if normalize else "1")
            long_name = meta.get("long_name", nc_name)

            if normalize and n_photons > 1:
                val = val / n_photons

            if role == "attr":
                attrs[f"{prefix}_{nc_name}"] = val
            elif role == "coord":
                coords[f"{prefix}_{nc_name}"] = (
                    f"{prefix}_{nc_name}",
                    val,
                    {"units": units, "long_name": long_name},
                )
            elif role == "data":
                dims = [f"{prefix}_{d}" for d in meta.get("dims", [])]
                data_vars[f"{prefix}_{nc_name}"] = (
                    dims,
                    val,
                    {"units": units, "long_name": long_name},
                )

        return xr.Dataset(attrs=attrs, coords=coords, data_vars=data_vars)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, prefix: str) -> Self:
        """Reconstructs the result object from a loaded xarray.Dataset."""
        kwargs = {}
        for f in fields(cls):
            if not f.metadata:
                continue

            nc_name = f.metadata.get("nc_name", f.name)
            full_name = f"{prefix}_{nc_name}"

            if full_name in ds.data_vars:
                kwargs[f.name] = ds[full_name].values
            elif full_name in ds.coords:
                kwargs[f.name] = ds.coords[full_name].values
            elif full_name in ds.attrs:
                kwargs[f.name] = ds.attrs[full_name]

        return cls(**kwargs)


@dataclass(slots=True)
class EngineResult:
    cpu_time_s: float = 0.0
    simulation_time_s: float = 0.0

    def merge(self, other: Self) -> "EngineResult":
        return EngineResult(
            cpu_time_s=self.cpu_time_s + other.cpu_time_s,
            simulation_time_s=self.simulation_time_s + other.simulation_time_s,
        )


@dataclass(slots=True)
class FateResult(BaseResult):
    energy_absorbed_surface: float = attr_field(normalize=True)
    energy_absorbed_atmosphere: float = attr_field(normalize=True)
    energy_reflected_toa: float = attr_field(normalize=True)


@dataclass(slots=True)
class VerticalFluxResult(BaseResult):
    measure_z: np.ndarray = coord_field(nc_name="z", units="km", long_name="Altitude")
    flux_up: np.ndarray = data_field(dims=["z"], normalize=True, long_name="Upward Flux")
    flux_down: np.ndarray = data_field(dims=["z"], normalize=True, long_name="Downward Flux")


@dataclass(slots=True)
class AbsorptionProfileResult(BaseResult):
    z_centers: np.ndarray = coord_field(nc_name="center_z", units="km", long_name="Altitude")
    absorption_profile_1d: np.ndarray = data_field(
        dims=["center_z"], normalize=True, long_name="Absorption Profile"
    )


@dataclass(slots=True)
class IncidentFluxMapResult(BaseResult):
    x_centers: np.ndarray = coord_field(nc_name="x", units="km", long_name="X Coordinate")
    y_centers: np.ndarray = coord_field(nc_name="y", units="km", long_name="Y Coordinate")
    measure_z: np.ndarray = coord_field(nc_name="z", units="km", long_name="Altitude")

    incident_flux_down_3d: np.ndarray = data_field(
        dims=["z", "x", "y"], normalize=True, long_name="Incident Downward Flux"
    )
    incident_flux_up_3d: np.ndarray = data_field(
        dims=["z", "x", "y"], normalize=True, long_name="Incident Upward Flux"
    )


@dataclass(slots=True)
class SurfaceAbsorptionResult(BaseResult):
    x_centers: np.ndarray = coord_field(nc_name="x", units="km", long_name="X Coordinate")
    y_centers: np.ndarray = coord_field(nc_name="y", units="km", long_name="Y Coordinate")
    surface_absorption_map_2d: np.ndarray = data_field(
        dims=["x", "y"], normalize=True, long_name="Surface Absorption"
    )


@dataclass(slots=True)
class PathTrackingResult(BaseResult):
    sample_paths_3d: np.ndarray = data_field(
        dims=["photon", "step", "coord"],
        units="km",
        on_merge="concat",
        long_name="Photon Path Coordinates",
    )
    sample_weights_2d: np.ndarray = data_field(
        dims=["photon", "step"], units="1", on_merge="concat", long_name="Photon Weight"
    )
    sample_reflected_toa: np.ndarray = data_field(
        dims=["photon"], units="boolean", on_merge="concat", long_name="Escaped TOA Flag"
    )
    sample_absorbed_atmosphere: np.ndarray = data_field(
        dims=["photon"], units="boolean", on_merge="concat", long_name="Absorbed in Atmosphere Flag"
    )
    sample_absorbed_surface: np.ndarray = data_field(
        dims=["photon"], units="boolean", on_merge="concat", long_name="Absorbed at Surface Flag"
    )
    toa_z: float = attr_field(units="km")

    def to_dataset(self, prefix: str, n_photons: int = 1, val_unit: str = "photons") -> xr.Dataset:
        ds = super(PathTrackingResult, self).to_dataset(prefix, n_photons, val_unit)
        if len(self.sample_paths_3d) > 0:
            dim_c = f"{prefix}_coord"
            ds = ds.assign_coords(
                {dim_c: (dim_c, np.array(["x", "y", "z"]), {"long_name": "Spatial Dimension"})}
            )

        return ds


# --- MASTER SIMULATION MODEL ---


@dataclass(slots=True)
class SimulationResults:
    engine: EngineResult = field(default_factory=EngineResult)
    detector_results: dict[str, BaseResult] = field(default_factory=dict)
    num_photons: int = 0

    def merge(self, other: Self) -> "SimulationResults":
        merged_detectors = {}

        for key in self.detector_results.keys() | other.detector_results.keys():
            res_self = self.detector_results.get(key)
            res_other = other.detector_results.get(key)

            if res_self and res_other:
                merged_detectors[key] = res_self.merge(res_other)
            else:
                merged_detectors[key] = res_self or res_other

        return SimulationResults(
            engine=self.engine.merge(other.engine),
            detector_results=merged_detectors,
            num_photons=self.num_photons + other.num_photons,
        )

    def to_dataset(self, normalize: bool = False) -> xr.Dataset:
        """Delegates dataset creation to all registered detector results and merges them."""
        n = self.num_photons if (normalize and self.num_photons > 0) else 1
        val_unit = "fraction" if normalize else "photons"

        datasets = [
            xr.Dataset(
                attrs={
                    "engine_cpu_time_s": self.engine.cpu_time_s,
                    "engine_simulation_time_s": self.engine.simulation_time_s,
                    "num_photons": self.num_photons,
                }
            )
        ] + [
            det_res.to_dataset(prefix=det_id, n_photons=n, val_unit=val_unit)
            for det_id, det_res in self.detector_results.items()
        ]

        with xr.set_options(keep_attrs=True):
            final_ds = xr.merge(datasets, combine_attrs="no_conflicts")

        det_types = {
            det_id: getattr(det_res, "_registry_id", type(det_res).__name__)
            for det_id, det_res in self.detector_results.items()
        }
        final_ds.attrs["_detector_types"] = json.dumps(det_types)

        return final_ds

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Reconstructs the full SimulationResults object dynamically using the registry."""
        engine = EngineResult(
            cpu_time_s=float(ds.attrs.get("engine_cpu_time_s", 0.0)),
            simulation_time_s=float(ds.attrs.get("engine_simulation_time_s", 0.0)),
        )

        det_types = json.loads(str(ds.attrs.get("_detector_types", "{}")))

        detector_results = {
            det_id: DETECTOR_RESULTS[class_name].from_dataset(ds, prefix=det_id)
            for det_id, class_name in det_types.items()
            if class_name in DETECTOR_RESULTS
        }

        num_photons = int(ds.attrs.get("num_photons", 0))

        return cls(engine=engine, detector_results=detector_results, num_photons=num_photons)
