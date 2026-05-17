import logging
import datetime
import json
import numpy as np
import shutil
import sys
import dataclasses
from pathlib import Path
from matplotlib.figure import Figure
from atmorad.config.config import SimConfig

class OutputHandler:
    def __init__(self, output_path: str = 'results', overwrite: bool = False) -> None:
        self.base_dir = Path(output_path)
        if overwrite:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        else:
            try:
                self.base_dir.mkdir(parents=True)
            except FileExistsError:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.base_dir = Path.cwd() / f"{output_path}_{timestamp}"
                self.base_dir.mkdir(parents=True)
                logging.info(f"Directory exists, saving to {self.base_dir}")
    
    def save_metadata(self, config: SimConfig, results_dict: dict) -> None:
        def _safe_serialize(obj):
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)

            if hasattr(obj, '__dict__'):
                res = {"_class": obj.__class__.__name__}
                for k, v in obj.__dict__.items():
                    if not k.startswith('_'):
                        res[k] = v
                return res

            if hasattr(obj, '__class__'):
                return str(obj.__class__.__name__)
            return str(obj)

        metadata = json.loads(json.dumps(dataclasses.asdict(config), default=_safe_serialize))

        if 'cpu_time_s' in results_dict:
            metadata['cpu_time_s'] = results_dict['cpu_time_s']
        if 'simulation_time_s' in results_dict:
            metadata['simulation_time_s'] = results_dict['simulation_time_s']

        with open(self.base_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        script_path = Path(sys.argv[0]).resolve()
        if script_path.exists() and script_path.suffix == '.py':
            shutil.copy(script_path, self.base_dir / 'experiment_setup.py')
        else:
            logging.warning('Failed to copy script to metadata')

    def save_plot(self, fig: Figure, plot_name: str, dpi: int = 300) -> None:
        fig.savefig(self.base_dir / plot_name, dpi=dpi, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)

    def save_results(self, results_dict: dict) -> None:
        npz_ready_dict = {}
        for k, v in results_dict.items():
            if isinstance(v, dict):
                npz_ready_dict[k] = np.array(v, dtype=object)
            else:
                npz_ready_dict[k] = v
        np.savez_compressed(self.base_dir / 'data_compressed.npz', **npz_ready_dict)

def read_results(path: Path | str) -> dict:
    try:
        data = np.load(path, allow_pickle=True)
        res_dict = {}
        for key in data.files:
            val = data[key]
            if val.dtype == 'O' and val.shape == ():
                res_dict[key] = val.item()
            else:
                res_dict[key] = val
        return res_dict
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file missing at {path}")