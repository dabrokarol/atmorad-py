import logging
import re
import shutil
from pathlib import Path

import tomli_w
import xarray as xr
from matplotlib.figure import Figure
from pydantic import ValidationError

from atmorad.config.schemas import SimConfig


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Normalizes all variables with 'photons' units to fractions."""
    if ds.attrs.get("is_normalized", 0):
        return ds

    num_photons = ds.attrs.get("num_photons", 1)
    if num_photons <= 1:
        return ds

    norm_ds = ds.copy(deep=False)

    for var_name, var_data in norm_ds.data_vars.items():
        if var_data.attrs.get("units") == "photons":
            norm_ds[var_name] = var_data / num_photons
            norm_ds[var_name].attrs["units"] = "fraction"

    norm_ds.attrs["is_normalized"] = 1
    return norm_ds


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
        return f"Result File:\n  {self.base_dir}/{self.results_filename}"

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
        logging.debug(f"Output directory initialized at: {self.base_dir}")

    def _save_nc_file(self, ds: xr.Dataset, target_path: Path, normalize: bool) -> None:
        tmp_path = target_path.with_name(target_path.name + ".tmp")
        if normalize:
            ds = normalize_dataset(ds)
        ds.to_netcdf(tmp_path, engine=self.NETCDF_ENGINE)
        shutil.move(tmp_path, target_path)

    def save_simulation_run(self, results_ds: xr.Dataset) -> None:
        assert self.base_dir is not None
        self._save_nc_file(results_ds, self.base_dir / self.results_filename, normalize=True)

    def save_checkpoint(self, results_ds: xr.Dataset) -> None:
        self._save_nc_file(results_ds, self.checkpoint_path, normalize=False)

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

    def load_checkpoint(self) -> xr.Dataset | None:
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
    def _load_nc_file(cls, path: Path) -> xr.Dataset | None:
        if not path.exists() and not path.is_file():
            return None
        try:
            with xr.open_dataset(path, engine=cls.NETCDF_ENGINE) as ds:
                ds.load()
                return ds
        except (OSError, ValueError, TypeError) as e:
            logging.exception(f"Failed to load data file: {path}. Error: {e}")

    @classmethod
    def load_simulation_results(cls, data_path: str | Path) -> xr.Dataset:
        data_path = Path(data_path)
        results = cls._load_nc_file(data_path)
        if results is None:
            raise RuntimeError(f"Could not load valid simulation results from {data_path}")
        return results

    @classmethod
    def extract_config(cls, data_path: str | Path, out_path: str | Path | None = None) -> None:
        data_path = Path(data_path)
        out_path = Path(out_path) if out_path else Path.cwd()

        if out_path.is_dir():
            out_path = out_path / f"{data_path.stem}_config.toml"

        try:
            with xr.open_dataset(data_path, engine=cls.NETCDF_ENGINE) as ds:
                config_json_str = ds.attrs["_simulation_config"]

            config = SimConfig.model_validate_json(config_json_str)

            with open(out_path, "wb") as f:
                clean_dict = config.model_dump(mode="json", exclude_unset=True)
                tomli_w.dump(clean_dict, f)

            logging.info(f"config successfully extracted to: {out_path}")

        except KeyError:
            logging.error(f'file {data_path.name} does not have "_simulation_config" attribute.')
        except ValidationError as e:
            logging.error(f"config validation failed: {e}")
        except OSError as e:
            logging.exception(f"file I/O error during config extraction: {e}")
