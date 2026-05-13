import logging
import datetime
import json
import numpy as np
import shutil
import sys

from pathlib import Path
from dataclasses import asdict
from matplotlib.figure import Figure

from atmorad.results import Results
from atmorad.config import SimConfig

class OutputHandler:
    def __init__(self, base_dir: str, overwrite = False) -> None:
        self.base_dir = Path.cwd() / base_dir
        if overwrite:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        else:
            try:
                self.base_dir.mkdir(parents=True)
            except FileExistsError:
                logging.info('directory exists, adding timestamp...')
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.base_dir = Path.cwd() / f"{base_dir}_{timestamp}"
                self.base_dir.mkdir()
    
    def save_metadata(self, config: SimConfig, results: Results) -> None:
        metadata = asdict(config)

        metadata['sim_duration_s'] = results.sim_duration_s
        metadata['summary'] = results.summary()
        metadata['total_surface_hits'] = results.surface_hits.shape[1]

        with open(self.base_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        script_path = Path(sys.argv[0]).resolve()
        if script_path.exists and script_path.suffix == '.py':
            shutil.copy(script_path, self.base_dir / 'experiment_setup.py')
        else:
            logging.warning('Failed to copy script to metadata')

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
            flux_down=data['flux_down'],
            sim_duration_s=data['sim_duration_s']
        )

    except FileNotFoundError:
        raise FileNotFoundError(f"data file missing at {path}")
    except KeyError as e:
        raise KeyError(f"Outdated or broken data file. Missing key: {e}")
    return res