import datetime
import json
import logging
import shutil
from pathlib import Path

import xarray as xr
from matplotlib.figure import Figure

from atmorad.config import SimConfig
from atmorad.models.results import SimulationResults


class DataIO:
    """Handles all file system operations: saving results, config, and checkpoints."""

    RESULTS_FILE = "data.nc"
    METADATA_FILE = "metadata.json"
    CONFIG_FILE = "runtime_config.toml"
    CHECKPOINT_FILE = "checkpoint.nc"
    NETCDF_ENGINE = "h5netcdf"

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

        results_path = self.base_dir / self.RESULTS_FILE

        ds = results.to_dataset(normalize=True)

        ds.to_netcdf(results_path, engine=self.NETCDF_ENGINE)

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

    def save_checkpoint(self, simulated_photons: int, results: SimulationResults):
        tmp_path = self.checkpoint_path.with_suffix(".nc.tmp")

        ds = results.to_dataset(normalize=False)

        ds.attrs["_simulated_photons"] = simulated_photons
        ds.attrs["_config_json"] = self.config.model_dump_json()

        ds.to_netcdf(tmp_path, engine=self.NETCDF_ENGINE)
        shutil.move(tmp_path, self.checkpoint_path)

    def load_checkpoint(self):
        if not self.checkpoint_path.exists():
            return 0, SimulationResults(), None

        try:
            with xr.open_dataset(self.checkpoint_path, engine=self.NETCDF_ENGINE) as ds:
                ds.load()
                simulated_photons = int(ds.attrs.get("_simulated_photons", 0))

                from atmorad.config import SimConfig

                config_json = str(ds.attrs.get("_config_json", "{}"))
                config = SimConfig.model_validate_json(config_json)

                results = SimulationResults.from_dataset(ds)

                return simulated_photons, results, config

        except (OSError, ValueError):
            logging.exception("Failed to load checkpoint file")
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

        with xr.open_dataset(results_path, engine=cls.NETCDF_ENGINE) as ds:
            ds.load()
            results = SimulationResults.from_dataset(ds)

        config = None
        if config_path.exists():
            from atmorad.config import load_config

            config = load_config(config_path)

        return config, results
