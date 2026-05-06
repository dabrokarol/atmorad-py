import logging
import time
import json
import tomllib
import numpy as np

from src.results import Results
from pathlib import Path

class OutputHandler:
    def __init__(self, base_dir: str, overwrite = False):
        self.base_dir = Path.cwd() / base_dir
        if overwrite:
            self.base_dir.mkdir(exist_ok=True)
        else:
            try:
                self.base_dir.mkdir()
            except Exception:
                logging.info('directory exists, adding timestamp...')
                self.base_dir = Path.cwd() / (base_dir + f"{time.ctime().replace(' ', '-')}")
                self.base_dir.mkdir()
    
    def save_metadata(self, config, execution_time_s: float):
        config['execution_time_s'] = execution_time_s
        with open(self.base_dir / 'metadata.json', 'w') as f:
            json.dump(config, f)

    def save_plot(self, fig, plot_name: str):
        fig.savefig(self.base_dir / plot_name, dpi=300, bbox_inches='tight')

    def save_results(self, results: Results):
        np.savez_compressed(
            self.base_dir / 'data_compressed',
            last_positions=results.last_positions,
            space_mask=results.space_mask,
            surface_mask=results.surface_mask,
            atmosphere_mask=results.atmosphere_mask,
            layer_idx=results.layer_idx,
            sample_paths=results.sample_paths
        )

    def print_results(self, results):
        print(results)
    
def read_config(path = 'config.toml'):
    try:
        with open(path, 'rb') as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"config file missing at {path}")
    return data

def read_results(path):
    try:
        data = np.fromfile(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"data file missing at {path}")
    return data