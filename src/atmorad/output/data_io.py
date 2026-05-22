import datetime
import json
import logging
import shutil
from pathlib import Path

import netCDF4 as nc
import numpy as np
from matplotlib.figure import Figure

from atmorad.config import SimConfig


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

    def save_simulation_run(self, results_dict: dict):
        self.save_metadata(results_dict)
        self.save_results(results_dict)
        if self.config.config_path:
            self.save_config_file(self.config.config_path)

    def save_metadata(self, results_dict: dict) -> None:
        metadata = self.config.model_dump(mode="json")

        for key in ["cpu_time_s", "simulation_time_s"]:
            if key in results_dict:
                metadata[key] = results_dict[key]

        metadata_path = self.base_dir / self.METADATA_FILE

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def save_figure(self, fig: Figure, relative_path: str, dpi: int = 300) -> None:
        full_path = self.base_dir / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_path, dpi=dpi, bbox_inches="tight")

    @property
    def checkpoint_path(self):
        return self.base_dir / self.CHECKPOINT_FILE

    def save_results(self, results_dict: dict) -> None:
        results_path = self.base_dir / self.RESULTS_FILE
        with nc.Dataset(results_path, "w", format="NETCDF4") as ncfile:
            self._save_dict_to_group(ncfile, results_dict)

    @classmethod
    def load_simulation_data(cls, directory: str | Path):
        """
        Loads the results and config from a completed simulation.
        Returns: (config, results_dict)
        """
        dir_path = Path(directory)
        results_path = dir_path / cls.RESULTS_FILE
        config_path = dir_path / cls.CONFIG_FILE

        if not results_path.exists():
            raise FileNotFoundError(f"Could not find results at {results_path.resolve()}")

        with nc.Dataset(results_path, "r") as ncfile:
            results = cls._load_group_to_dict(ncfile)

        config = None
        if config_path.exists():
            from atmorad.config import load_config

            config = load_config(config_path)

        return config, results

    def save_checkpoint(self, simulated_photons: int, results: dict):
        tmp_path = self.checkpoint_path.with_suffix(".nc.tmp")

        with nc.Dataset(tmp_path, "w", format="NETCDF4") as ncfile:
            ncfile.setncattr("simulated_photons", simulated_photons)
            ncfile.setncattr("config_json", self.config.model_dump_json())

            res_grp = ncfile.createGroup("res")
            self._save_dict_to_group(res_grp, results)

        shutil.move(tmp_path, self.checkpoint_path)

    def load_checkpoint(self):
        if not self.checkpoint_path.exists():
            return 0, {}, None

        try:
            with nc.Dataset(self.checkpoint_path, "r") as ncfile:
                simulated_photons = int(ncfile.getncattr("simulated_photons"))

                from atmorad.config import SimConfig

                config_json = str(ncfile.getncattr("config_json"))
                config = SimConfig.model_validate_json(config_json)

                results = {}
                if "res" in ncfile.groups:
                    results = self._load_group_to_dict(ncfile.groups["res"])

                return simulated_photons, results, config

        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return 0, {}, None

    def delete_checkpoint(self):
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    @classmethod
    def _save_dict_to_group(cls, group, d: dict):
        """Recursively saves a dictionary into NetCDF groups and variables."""
        for k, v in d.items():
            if isinstance(v, dict):
                subgroup = group.createGroup(k)
                cls._save_dict_to_group(subgroup, v)
            elif isinstance(v, np.ndarray):
                dims = []
                for i, dim_size in enumerate(v.shape):
                    dim_name = f"{k}_dim_{i}"
                    if dim_name not in group.dimensions:
                        group.createDimension(dim_name, dim_size)
                    dims.append(dim_name)

                var = group.createVariable(k, v.dtype, tuple(dims))
                var[:] = v
            elif isinstance(v, np.generic):
                group.setncattr(k, v.item())
            else:
                group.setncattr(k, v)

    @classmethod
    def _load_group_to_dict(cls, group) -> dict:
        """Recursively reconstructs a dictionary from NetCDF groups."""
        res = {}

        for attr_name in group.ncattrs():
            res[attr_name] = group.getncattr(attr_name)

        for var_name, var in group.variables.items():
            res[var_name] = np.array(var[:])

        for grp_name, grp in group.groups.items():
            res[grp_name] = cls._load_group_to_dict(grp)

        return res
