import dataclasses
import datetime
import json
import logging
import pickle
import shutil
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from atmorad.config import SimConfig
from atmorad.output import ResultAnalyzer


class DataIO:
    def __init__(self, config: SimConfig) -> None:
        self.config = config

        output_dir = Path(config.output.path)
        exp_name = config.metadata.experiment_name.replace(" ", "-")
        resume = config.engine.resume_from_checkpoint
        overwrite = config.output.overwrite

        if resume:
            valid_dirs = [
                d
                for d in output_dir.glob(f"{exp_name}*")
                if d.is_dir() and (d / "checkpoint.pkl").exists()
            ]

            if valid_dirs:
                self.base_dir = max(valid_dirs, key=lambda p: p.stat().st_mtime)
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

    def save_config_file(self, config_file_path: Path):
        if not config_file_path.exists():
            logging.error(f"Cannot find original config at {config_file_path.resolve()}")
            return

        destination_path = self.base_dir / "runtime_config.toml"
        shutil.copy2(config_file_path, destination_path)

        logging.info(f"Config saved to {destination_path}.")

    def save_metadata(self, config: SimConfig, results_dict: dict) -> None:
        def _safe_serialize(obj):
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)

            if hasattr(obj, "__dict__"):
                res = {"_class": obj.__class__.__name__}
                for k, v in obj.__dict__.items():
                    if not k.startswith("_"):
                        res[k] = v
                return res

            if hasattr(obj, "__class__"):
                return str(obj.__class__.__name__)
            return str(obj)

        metadata = json.loads(json.dumps(dataclasses.asdict(config), default=_safe_serialize))

        if "cpu_time_s" in results_dict:
            metadata["cpu_time_s"] = results_dict["cpu_time_s"]
        if "simulation_time_s" in results_dict:
            metadata["simulation_time_s"] = results_dict["simulation_time_s"]

        with open(self.base_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def save_plot(self, fig: Figure, plot_name: str, dpi: int = 300) -> None:
        fig.savefig(self.base_dir / plot_name, dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)

    def save_results(self, results_dict: dict) -> None:
        npz_ready_dict = {}
        for k, v in results_dict.items():
            if isinstance(v, dict):
                npz_ready_dict[k] = np.array(v, dtype=object)
            else:
                npz_ready_dict[k] = v
        np.savez_compressed(self.base_dir / "data_compressed.npz", **npz_ready_dict)

    def save_all_artifacts(self, analyzer: ResultAnalyzer, results_dict: dict):
        config = self.config

        if config.output.save_absorption_maps:
            fig_map = analyzer.plot_surface_absorption_map()
            if fig_map:
                self.save_plot(fig_map, "surface_absorption_map.png")
            else:
                logging.warning("2d surface absorption map not generated")
            fig_toa_map = analyzer.plot_toa_flux_map()
            if fig_toa_map:
                self.save_plot(fig_toa_map, "toa_flux_map.png")
            else:
                logging.warning("2d toa flux map not generated")

        if config.output.save_incident_flux_maps:
            subfolder_name = "incident_flux"
            subfolder_path = self.base_dir / subfolder_name
            subfolder_path.mkdir(exist_ok=True)

            down_maps = results_dict.get("incident_flux_down_maps_2d", {})
            for z_val, flux_map in down_maps.items():
                title = f"Incident Downward Flux Map\nHeight: {z_val} km"
                fig = analyzer.plot_2d_map(flux_map, title=title)
                if fig:
                    self.save_plot(fig, f"{subfolder_name}/downward_z_{z_val:g}km.png")

            up_maps = results_dict.get("incident_flux_up_maps_2d", {})
            for z_val, flux_map in up_maps.items():
                title = f"Incident Upward Flux Map\nHeight: {z_val} km"
                fig = analyzer.plot_2d_map(flux_map, title=title)
                if fig:
                    self.save_plot(fig, f"{subfolder_name}/upward_z_{z_val:g}km.png")

        if config.output.save_vertical_profiles:
            fig_flux = analyzer.plot_flux_profile()
            if fig_flux:
                self.save_plot(fig_flux, "vertical_flux_profile.png")

            fig_heat = analyzer.plot_absorption_profile()
            if fig_heat:
                self.save_plot(fig_heat, "absorption_profile.png")

        if config.output.save_photon_paths:
            fig_paths = analyzer.plot_paths()
            if fig_paths:
                self.save_plot(fig_paths, "3d_photon_paths.png")

        self.save_metadata(config, results_dict)
        self.save_results(results_dict)

    @property
    def checkpoint_path(self):
        return self.base_dir / "checkpoint.pkl"

    def save_checkpoint(self, simulated_photons: int, results: dict, config: SimConfig):
        state = {"simulated_photons": simulated_photons, "results": results, "config": config}
        tmp_path = self.checkpoint_path.with_suffix(".pkl.tmp")

        with open(tmp_path, "wb") as f:
            pickle.dump(state, f)

        shutil.move(tmp_path, self.checkpoint_path)

    def load_checkpoint(self):
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "rb") as f:
                state = pickle.load(f)
                return state["simulated_photons"], state["results"], state["config"]
        logging.info("No checkpoint found. Starting a fresh simulation.")
        return 0, {}, None

    def delete_checkpoint(self):
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
