import dataclasses
import datetime
import json
import logging
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        overwrite = config.output.overwrite
        if overwrite:
            self.base_dir = output_dir / f"{exp_name}"
            if self.base_dir.exists():
                import shutil

                shutil.rmtree(self.base_dir)
        else:
            self.base_dir = output_dir / f"{exp_name}-{timestamp}"

        self.base_dir.mkdir(parents=True, exist_ok=True)

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
                    self.save_plot(fig, f"{subfolder_name}/downward_z_{z_val}km.png")

            up_maps = results_dict.get("incident_flux_up_maps_2d", {})
            for z_val, flux_map in up_maps.items():
                title = f"Incident Upward Flux Map\nHeight: {z_val} km"
                fig = analyzer.plot_2d_map(flux_map, title=title)
                if fig:
                    self.save_plot(fig, f"{subfolder_name}/upward_z_{z_val}km.png")

        if config.output.save_vertical_profile:
            fig_flux = analyzer.plot_flux_profile()
            if fig_flux:
                self.save_plot(fig_flux, "vertical_flux_profile.png")

            fig_heat = analyzer.plot_heating_rate()
            if fig_heat:
                self.save_plot(fig_heat, "heating_profile.png")

        if config.output.save_photon_paths:
            fig_paths = analyzer.plot_paths()
            if fig_paths:
                self.save_plot(fig_paths, "3d_photon_paths.png")

        self.save_metadata(config, results_dict)
        self.save_results(results_dict)
