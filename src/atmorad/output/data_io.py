import datetime
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from atmorad.config import SimConfig
from atmorad.models.results import (
    AbsorptionProfileResult,
    EngineResult,
    FateResult,
    IncidentFluxMapResult,
    PathTrackingResult,
    SimulationResults,
    SurfaceAbsorptionResult,
    VerticalFluxResult,
)


class DataIO:
    RESULTS_FILE = "data.nc"
    METADATA_FILE = "metadata.json"
    CONFIG_FILE = "runtime_config.toml"
    CHECKPOINT_FILE = "checkpoint.nc"

    def __init__(self, config: SimConfig) -> None:
        self.config = config

        output_dir = Path(config.output.path)
        exp_name = config.metadata.experiment_name.replace(" ", "-")
        resume = config.engine.resume_from_checkpoint
        overwrite = config.output.overwrite

        if resume:
            latest_checkpoint_dir = self._find_latest_checkpoint_dir(output_dir, exp_name)
            if latest_checkpoint_dir:
                self.base_dir = latest_checkpoint_dir
                logging.info(f"Resuming from the most recent directory: {self.base_dir}")
                return

            logging.warning(
                f"Resume requested for '{exp_name}', but no checkpoint found. Starting fresh."
            )

        if overwrite:
            self.base_dir = output_dir / f"{exp_name}"
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.base_dir = output_dir / f"{exp_name}-{timestamp}"

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def output_summary(self) -> str:
        lines = [f"Outputs saved to: {self.base_dir}/"]
        files = [self.METADATA_FILE, self.RESULTS_FILE, self.CONFIG_FILE]
        for i, filename in enumerate(files):
            if i == len(files) - 1:
                lines.append(f"  └─ {filename}")
            else:
                lines.append(f"  ├─ {filename}")

        return "\n".join(lines)

    def _find_latest_checkpoint_dir(self, output_dir: Path, exp_name: str) -> Path | None:
        valid_dirs = [
            d
            for d in output_dir.glob(f"{exp_name}*")
            if d.is_dir() and (d / self.CHECKPOINT_FILE).exists()
        ]
        return max(valid_dirs, key=lambda p: p.stat().st_mtime) if valid_dirs else None

    def save_config_file(self, config_file_path: Path):
        if not config_file_path.exists():
            logging.error(f"Cannot find original config at {config_file_path.resolve()}")
            return

        destination_path = self.base_dir / self.CONFIG_FILE
        shutil.copy2(config_file_path, destination_path)

    def save_simulation_run(self, results: SimulationResults):
        self.save_metadata(results)
        self.save_results(results)
        if self.config.config_path:
            self.save_config_file(self.config.config_path)

    def save_metadata(self, results: SimulationResults) -> None:
        metadata = self.config.model_dump(mode="json")
        metadata["cpu_time_s"] = results.engine.cpu_time_s
        metadata["simulation_time_s"] = results.engine.simulation_time_s

        metadata_path = self.base_dir / self.METADATA_FILE

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def save_figure(self, fig: Figure, relative_path: str, dpi: int = 300) -> None:
        full_path = self.base_dir / relative_path.lstrip("/")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_path, dpi=dpi, bbox_inches="tight")

    @property
    def checkpoint_path(self):
        return self.base_dir / self.CHECKPOINT_FILE

    def save_results(self, results: SimulationResults) -> None:
        results_path = self.base_dir / self.RESULTS_FILE
        ds = self._results_to_dataset(results)
        ds.to_netcdf(results_path, engine="netcdf4")

    def save_checkpoint(self, simulated_photons: int, results: SimulationResults):
        tmp_path = self.checkpoint_path.with_suffix(".nc.tmp")

        ds = self._results_to_dataset(results)
        ds.attrs["_simulated_photons"] = simulated_photons
        ds.attrs["_config_json"] = self.config.model_dump_json()

        ds.to_netcdf(tmp_path, engine="netcdf4")
        shutil.move(tmp_path, self.checkpoint_path)

    def load_checkpoint(self):
        if not self.checkpoint_path.exists():
            return 0, SimulationResults(), None

        try:
            with xr.open_dataset(self.checkpoint_path, engine="netcdf4") as ds:
                ds.load()
                simulated_photons = int(ds.attrs.get("_simulated_photons", 0))

                from atmorad.config import SimConfig

                config_json = str(ds.attrs.get("_config_json", "{}"))
                config = SimConfig.model_validate_json(config_json)

                results = self._dataset_to_results(ds)

                return simulated_photons, results, config

        except Exception as e:
            logging.error(f"Failed to load checkpoint file: {e}")
            return 0, SimulationResults(), None

    def delete_checkpoint(self):
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    @classmethod
    def load_simulation_data(cls, directory: str | Path):
        dir_path = Path(directory)
        results_path = dir_path / cls.RESULTS_FILE
        config_path = dir_path / cls.CONFIG_FILE

        if not results_path.exists():
            raise FileNotFoundError(f"Could not find results at {results_path.resolve()}")

        with xr.open_dataset(results_path, engine="netcdf4") as ds:
            ds.load()
            results = cls._dataset_to_results(ds)

        config = None
        if config_path.exists():
            from atmorad.config import load_config

            config = load_config(config_path)

        return config, results

    @classmethod
    def _results_to_dataset(cls, results: SimulationResults) -> xr.Dataset:
        data_vars = {}
        coords = {}

        attrs = {
            "engine_cpu_time_s": results.engine.cpu_time_s,
            "engine_simulation_time_s": results.engine.simulation_time_s,
        }

        det_types = {}

        for det_id, det_res in results.detector_results.items():
            det_types[det_id] = type(det_res).__name__
            pfx = det_id

            if isinstance(det_res, FateResult):
                attrs[f"{pfx}_photons_absorbed_surface"] = det_res.photons_absorbed_surface
                attrs[f"{pfx}_photons_absorbed_atmosphere"] = det_res.photons_absorbed_atmosphere
                attrs[f"{pfx}_photons_escaped_toa"] = det_res.photons_escaped_toa
                attrs[f"{pfx}_cpu_time_s"] = det_res.cpu_time_s

            elif isinstance(det_res, VerticalFluxResult):
                dim_z = f"{pfx}_z"
                coords[dim_z] = (dim_z, det_res.measure_z)
                data_vars[f"{pfx}_flux_up"] = ([dim_z], det_res.flux_up)
                data_vars[f"{pfx}_flux_down"] = ([dim_z], det_res.flux_down)

            elif isinstance(det_res, AbsorptionProfileResult):
                dim_z = f"{pfx}_center_z"
                coords[dim_z] = (dim_z, det_res.z_centers)
                data_vars[f"{pfx}_absorption_profile_1d"] = ([dim_z], det_res.absorption_profile_1d)

            elif isinstance(det_res, IncidentFluxMapResult):
                dim_x, dim_y, dim_z = f"{pfx}_x", f"{pfx}_y", f"{pfx}_z"

                coords[dim_x] = (dim_x, det_res.x_centers)
                coords[dim_y] = (dim_y, det_res.y_centers)
                coords[dim_z] = (dim_z, det_res.measure_z)

                data_vars[f"{pfx}_incident_flux_down_3d"] = (
                    [dim_z, dim_x, dim_y],
                    det_res.incident_flux_down_3d,
                )
                data_vars[f"{pfx}_incident_flux_up_3d"] = (
                    [dim_z, dim_x, dim_y],
                    det_res.incident_flux_up_3d,
                )

            elif isinstance(det_res, SurfaceAbsorptionResult):
                dim_x, dim_y = f"{pfx}_x", f"{pfx}_y"

                coords[dim_x] = (dim_x, det_res.x_centers)
                coords[dim_y] = (dim_y, det_res.y_centers)

                data_vars[f"{pfx}_surface_absorption_map_2d"] = (
                    [dim_x, dim_y],
                    det_res.surface_absorption_map_2d,
                )
                data_vars[f"{pfx}_surface_absorption_map_2d"] = (
                    [dim_x, dim_y],
                    det_res.surface_absorption_map_2d,
                )
            elif isinstance(det_res, PathTrackingResult):
                attrs[f"{pfx}_toa_z"] = det_res.toa_z
                if len(det_res.sample_paths_3d) > 0:
                    dim_p, dim_s, dim_c = f"{pfx}_photon", f"{pfx}_step", f"{pfx}_coord"

                    coords[dim_c] = (dim_c, np.array(["x", "y", "z"]))

                    data_vars[f"{pfx}_sample_paths_3d"] = (
                        [dim_p, dim_s, dim_c],
                        det_res.sample_paths_3d,
                    )
                    data_vars[f"{pfx}_sample_weights_2d"] = (
                        [dim_p, dim_s],
                        det_res.sample_weights_2d,
                    )
                    data_vars[f"{pfx}_sample_escaped_toa"] = ([dim_p], det_res.sample_escaped_toa)
                    data_vars[f"{pfx}_sample_absorbed_atmosphere"] = (
                        [dim_p],
                        det_res.sample_absorbed_atmosphere,
                    )
                    data_vars[f"{pfx}_sample_absorbed_surface"] = (
                        [dim_p],
                        det_res.sample_absorbed_surface,
                    )
                else:
                    attrs[f"{pfx}_empty_paths"] = 1

        attrs["_detector_types"] = json.dumps(det_types)
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    @classmethod
    def _dataset_to_results(cls, ds: xr.Dataset) -> SimulationResults:
        """Odbudowuje Pythonowe struktury z xarray.Dataset."""
        engine = EngineResult(
            cpu_time_s=float(ds.attrs.get("engine_cpu_time_s", 0.0)),
            simulation_time_s=float(ds.attrs.get("engine_simulation_time_s", 0.0)),
        )

        detector_results = {}
        det_types = json.loads(ds.attrs.get("_detector_types", "{}"))

        for det_id, class_name in det_types.items():
            pfx = det_id

            if class_name == "FateResult":
                detector_results[det_id] = FateResult(
                    photons_absorbed_surface=int(
                        ds.attrs.get(f"{pfx}_photons_absorbed_surface", 0)
                    ),
                    photons_absorbed_atmosphere=int(
                        ds.attrs.get(f"{pfx}_photons_absorbed_atmosphere", 0)
                    ),
                    photons_escaped_toa=int(ds.attrs.get(f"{pfx}_photons_escaped_toa", 0)),
                    cpu_time_s=float(ds.attrs.get(f"{pfx}_cpu_time_s", 0.0)),
                )

            elif class_name == "VerticalFluxResult":
                detector_results[det_id] = VerticalFluxResult(
                    measure_z=ds.coords[f"{pfx}_z"].values,
                    flux_up=ds[f"{pfx}_flux_up"].values,
                    flux_down=ds[f"{pfx}_flux_down"].values,
                )

            elif class_name == "AbsorptionProfileResult":
                detector_results[det_id] = AbsorptionProfileResult(
                    z_centers=ds.coords[f"{pfx}_center_z"].values,
                    absorption_profile_1d=ds[f"{pfx}_absorption_profile_1d"].values,
                )

            elif class_name == "IncidentFluxMapResult":
                detector_results[det_id] = IncidentFluxMapResult(
                    x_centers=ds.coords[f"{pfx}_x"].values,
                    y_centers=ds.coords[f"{pfx}_y"].values,
                    measure_z=ds.coords[f"{pfx}_z"].values,
                    incident_flux_down_3d=ds[f"{pfx}_incident_flux_down_3d"].values,
                    incident_flux_up_3d=ds[f"{pfx}_incident_flux_up_3d"].values,
                )

            elif class_name == "SurfaceAbsorptionResult":
                detector_results[det_id] = SurfaceAbsorptionResult(
                    x_centers=ds.coords[f"{pfx}_x"].values,
                    y_centers=ds.coords[f"{pfx}_y"].values,
                    surface_absorption_map_2d=ds[f"{pfx}_surface_absorption_map_2d"].values,
                )

            elif class_name == "PathTrackingResult":
                if ds.attrs.get(f"{pfx}_empty_paths", False):
                    detector_results[det_id] = PathTrackingResult(
                        sample_paths_3d=np.array([]),
                        sample_weights_2d=np.array([]),
                        sample_escaped_toa=np.array([]),
                        sample_absorbed_atmosphere=np.array([]),
                        sample_absorbed_surface=np.array([]),
                        toa_z=float(ds.attrs.get(f"{pfx}_toa_z", 0.0)),
                    )
                else:
                    detector_results[det_id] = PathTrackingResult(
                        sample_paths_3d=ds[f"{pfx}_sample_paths_3d"].values,
                        sample_weights_2d=ds[f"{pfx}_sample_weights_2d"].values,
                        sample_escaped_toa=ds[f"{pfx}_sample_escaped_toa"].values,
                        sample_absorbed_atmosphere=ds[f"{pfx}_sample_absorbed_atmosphere"].values,
                        sample_absorbed_surface=ds[f"{pfx}_sample_absorbed_surface"].values,
                        toa_z=float(ds.attrs.get(f"{pfx}_toa_z", 0.0)),
                    )

        return SimulationResults(engine=engine, detector_results=detector_results)
