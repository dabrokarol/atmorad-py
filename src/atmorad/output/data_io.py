import datetime
import json
import logging
import pickle
import shutil
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from atmorad.config import SimConfig


class DataIO:
    RESULTS_FILE = "data_compressed.npz"
    METADATA_FILE = "metadata.json"
    CONFIG_FILE = "runtime_config.toml"
    CHECKPOINT_FILE = "checkpoint.pkl"

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

    def save_results(self, results_dict: dict) -> None:
        npz_ready_dict = {}
        for k, v in results_dict.items():
            if isinstance(v, dict):
                npz_ready_dict[k] = np.array(v, dtype=object)
            else:
                npz_ready_dict[k] = v

        results_path = self.base_dir / self.RESULTS_FILE
        np.savez_compressed(results_path, **npz_ready_dict)

    @property
    def checkpoint_path(self):
        return self.base_dir / self.CHECKPOINT_FILE

    def save_checkpoint(self, simulated_photons: int, results: dict):
        state = {"simulated_photons": simulated_photons, "results": results, "config": self.config}
        tmp_path = self.checkpoint_path.with_suffix(".pkl.tmp")

        with open(tmp_path, "wb") as f:
            pickle.dump(state, f)

        shutil.move(tmp_path, self.checkpoint_path)

    def load_checkpoint(self):
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "rb") as f:
                state = pickle.load(f)
                return state["simulated_photons"], state["results"], state["config"]
        return 0, {}, None

    def delete_checkpoint(self):
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
