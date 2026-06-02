import logging
import re
import shutil
from pathlib import Path

import tomli_w
import xarray as xr
from matplotlib.figure import Figure
from pydantic import ValidationError

from atmorad.config import SimConfig
from atmorad.models.results import SimResults


class DataIO:
    """Handles all file system operations: saving results, config, and checkpoints."""

    NETCDF_ENGINE = "h5netcdf"

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        meta = config.metadata

        self.exp_name = meta.experiment_name.replace(" ", "-")
        self.scen_name = meta.scenario_name
        self.base_name = f"atmorad_{self.exp_name}_{self.scen_name}"
        self.results_filename = f"{self.base_name}.nc"
        self.checkpoint_filename = f"{self.base_name}_checkpoint.nc"

        self.output_dir = config.output.base_dir
        self.fig_dir_base = config.output.fig_dir

        self.base_dir: Path | None = None
        self.fig_dir: Path | None = None

        if config.engine.resume_from_checkpoint:
            checkpoint_dir, self.checkpoint_config = self.find_checkpoint()
            self._initialize_directories(checkpoint_dir)
        else:
            self._initialize_directories()

    def output_summary(self) -> str:
        lines = [f"Outputs saved to: {self.base_dir}/"]
        files = [self.results_filename]
        for i, filename in enumerate(files):
            lines.append(f"  {'└─' if i == len(files) - 1 else '├─'} {filename}")
        return "\n".join(lines)

    def _generate_candidate_dirs(self):
        """Yields valid directories sorted from newest to oldest."""
        timestamp_pattern = re.compile(r"^\d{8}-\d{6}$")
        valid_dirs = []

        for candidate in self.output_dir.glob(f"{self.exp_name}-*"):
            if not candidate.is_dir():
                continue

            suffix = candidate.name.removeprefix(f"{self.exp_name}-")
            if not timestamp_pattern.fullmatch(suffix):
                continue

            if (candidate / self.checkpoint_filename).is_file() or (
                candidate / self.results_filename
            ).is_file():
                valid_dirs.append(candidate)

        base_dir = self.output_dir / self.exp_name
        if base_dir.is_dir():
            if (base_dir / self.checkpoint_filename).is_file() or (
                base_dir / self.results_filename
            ).is_file():
                valid_dirs.append(base_dir)

        valid_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        yield from valid_dirs

    def _iter_checkpoint_configs(self):
        """Yields (candidate_dir, config) for all readable checkpoints."""
        for candidate in self._generate_candidate_dirs():
            target = candidate / self.results_filename
            if not target.exists():
                target = candidate / self.checkpoint_filename
                if not target.exists():
                    continue

            try:
                with xr.open_dataset(target, engine=self.NETCDF_ENGINE) as ds:
                    config_str = ds.attrs.get("_simulation_config")
                    if config_str:
                        yield candidate, SimConfig.model_validate_json(config_str)
            except Exception as e:
                logging.debug(f"Failed to load config from {target.name}: {e}")

    def find_checkpoint(self) -> tuple[Path, SimConfig] | tuple[None, None]:
        """Looks for compatible checkpoint. Remembers yielded last checkpoint directory."""
        for candidate_dir, old_config in self._iter_checkpoint_configs():
            if self.config.is_compatible_for_resume(old_config):
                logging.info(f"Found compatible checkpoint in: {candidate_dir.name}")
                return candidate_dir, old_config

            logging.debug(f"Skipped {candidate_dir.name}: Configuration mismatch.")
        return None, None

    def _initialize_directories(self, checkpoint_dir: Path | None = None) -> None:
        """Setups working directory"""
        if checkpoint_dir:
            self.base_dir = checkpoint_dir
            self.fig_dir = self.fig_dir_base / checkpoint_dir.name
        else:
            overwrite = self.config.output.overwrite
            timestamp = self.config.metadata.run_timestamp
            run_folder_name = self.exp_name if overwrite else f"{self.exp_name}-{timestamp}"
            self.base_dir = self.output_dir / run_folder_name
            self.fig_dir = self.fig_dir_base / run_folder_name

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory initialized at: {self.base_dir}")

    def _save_nc_file(self, results: SimResults, target_path: Path, normalize: bool) -> None:
        tmp_path = target_path.with_name(target_path.name + ".tmp")
        ds = results.to_dataset(normalize=normalize)
        ds.to_netcdf(tmp_path, engine=self.NETCDF_ENGINE)
        shutil.move(tmp_path, target_path)

    def save_simulation_run(self, results: SimResults) -> None:
        assert self.base_dir is not None
        self._save_nc_file(results, self.base_dir / self.results_filename, normalize=True)

    def save_checkpoint(self, results: SimResults) -> None:
        self._save_nc_file(results, self.checkpoint_path, normalize=False)

    def save_figure(self, fig: Figure, plot_name: str, dpi: int = 300) -> None:
        assert self.fig_dir is not None
        filename = f"{plot_name}_{self.exp_name}_{self.scen_name}.png"
        full_path = self.fig_dir / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_path, dpi=dpi, bbox_inches="tight")

    @property
    def checkpoint_path(self) -> Path:
        assert self.base_dir is not None
        return self.base_dir / self.checkpoint_filename

    def load_checkpoint(self) -> SimResults | None:
        """Loads simulation state from a completed results file or a checkpoint."""
        assert self.base_dir is not None
        finished_path = self.base_dir / self.results_filename
        if finished_path.exists():
            return self._load_nc_file(finished_path)

        if self.checkpoint_path.exists():
            return self._load_nc_file(self.checkpoint_path)

        return None

    def delete_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    @classmethod
    def _load_nc_file(cls, path: Path) -> SimResults | None:
        if not path.exists():
            return None
        try:
            with xr.open_dataset(path, engine=cls.NETCDF_ENGINE) as ds:
                ds.load()
                return SimResults.from_dataset(ds)
        except (OSError, ValueError, TypeError) as e:
            logging.exception(f"Failed to load data file: {path}. Error: {e}")

    @classmethod
    def load_simulation_results(cls, directory: str | Path) -> SimResults:
        dir_path = Path(directory)
        nc_files = [f for f in dir_path.glob("*.nc") if "checkpoint" not in f.name]

        if not nc_files:
            raise FileNotFoundError(f"Could not find any result .nc files at {dir_path.resolve()}")
        elif len(nc_files) > 1:
            logging.warning(
                f"Multiple .nc files found in {dir_path}. Loading the most recently modified."
            )
            nc_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        results = cls._load_nc_file(nc_files[0])
        if results is None:
            raise RuntimeError(f"Could not load valid simulation results from {nc_files[0]}")
        return results

    @classmethod
    def extract_config(cls, data_path: str | Path, out_path: str | Path | None = None):
        data_path = Path(data_path)
        out_path = Path(out_path) if out_path else Path.cwd()

        if out_path.is_dir():
            out_path = out_path / f"{data_path.stem}_config.toml"

        try:
            with xr.open_dataset(data_path, engine=cls.NETCDF_ENGINE) as ds:
                config_json_str = ds.attrs["_simulation_config"]

            config_reformatted = SimConfig.model_validate_json(config_json_str).format_to_save()

            with open(out_path, "wb") as f:
                tomli_w.dump(config_reformatted, f)
            logging.info(f"Config successfully extracted to: {out_path}")

        except KeyError:
            logging.error(f'File {data_path} does not have "_simulation_config" attribute.')
        except ValidationError as e:
            logging.error(f"Config validation failed: {e}.")
        except (OSError, ValueError) as e:
            logging.exception(f"Couldn't extract the configuration file: {e}")
