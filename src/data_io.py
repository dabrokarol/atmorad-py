import logging
import datetime
import json
import tomllib
import numpy as np
import shutil
import sys

from pathlib import Path
from typing import Any
from dataclasses import asdict
from matplotlib.figure import Figure

from src.results import Results
from src.config import SimConfig

class OutputHandler:
    def __init__(self, base_dir: str, overwrite = False) -> None:
        self.base_dir = Path.cwd() / base_dir
        if overwrite:
            self.base_dir.mkdir(exist_ok=True)
        else:
            try:
                self.base_dir.mkdir()
            except FileExistsError:
                logging.info('directory exists, adding timestamp...')
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.base_dir = Path.cwd() / f"{base_dir}_{timestamp}"
                self.base_dir.mkdir()
    
    def save_metadata(self, config: SimConfig, execution_time_s: float|None = None) -> None:
        config_dict = asdict(config)
        if execution_time_s:
            config_dict['execution_time_s'] = execution_time_s
        with open(self.base_dir / 'metadata.json', 'w') as f:
            json.dump(config_dict, f, indent=4)
        shutil.copy(sys.argv[0], self.base_dir / 'experiment_setup.py')

    def save_plot(self, fig: Figure, plot_name: str) -> None:
        fig.savefig(self.base_dir / plot_name, dpi=300, bbox_inches='tight')

    def save_results(self, results: Results) -> None:
        np.savez_compressed(
            self.base_dir / 'data_compressed',
            final_positions=results.final_positions,
            space_mask=results.space_mask,
            surface_mask=results.surface_mask,
            layer_idx=results.layer_idx,
            sample_paths=np.array(results.sample_paths, dtype=object),
            surface_hits=results.surface_hits,
            scatter_counts=results.scatter_counts,
            measure_z=results.measure_z,
            flux_up=results.flux_up,
            flux_down=results.flux_down
        )

    def print_results(self, results: Results) -> None:
        print(results.summary())

def read_results(path: Path|str) -> Results:
    try:
        data = np.load(path, allow_pickle=True)
        res = Results(
            final_positions=data['final_positions'],
            space_mask=data['space_mask'],
            surface_mask=data['surface_mask'],
            layer_idx=data['layer_idx'],
            sample_paths=data['sample_paths'].item(), # .item() because it was a dict dumped to np.array
            surface_hits=data['surface_hits'],
            scatter_counts=data['scatter_counts'],
            measure_z=data['measure_z'],
            flux_up=data['flux_up'],
            flux_down=data['flux_down']
        )

    except FileNotFoundError:
        raise FileNotFoundError(f"data file missing at {path}")
    except KeyError as e:
        raise KeyError(f"Outdated or broken data file. Missing key: {e}")
    return res