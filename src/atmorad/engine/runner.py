import time
import numpy as np
import concurrent.futures
from dataclasses import replace

from atmorad.environment.scene import Scene
from atmorad.environment.atmosphere import Atmosphere
from atmorad.engine.engine import Engine
from atmorad.config.config import SimConfig
from atmorad.detectors.heating import HeatingRateDetector
from atmorad.detectors.flux import VerticalFluxDetector
from atmorad.detectors.surface_hits import BoundaryMapDetector
from atmorad.detectors.photon_paths import PathTrackingDetector

class MCRadiationRunner:
    def __init__(self, config: SimConfig):
        self.config = config
        self.scene = Scene(config.surface, Atmosphere(config.layers))

    def run(self):
        start_time = time.perf_counter()
        chunk_results = parallel_simulation(self.config, self.scene)
        self.results = self._merge_results(chunk_results)
        end_time = time.perf_counter()
        self.results['simulation_time_s'] = end_time - start_time
        
    def _merge_results(self, chunk_results: list[dict]) -> dict:
        if not chunk_results:
            return {}

        merged = {}
        template = chunk_results[0]

        for key in template.keys():
            if key in ["measure_z", "layer_boundaries_z", "x_edges", "y_edges"]:
                merged[key] = template[key]
            elif key in ["flux_up", "flux_down", "surface_flux_map_2d", "toa_flux_map_2d", 
                        "heating_profile_1d", "scatter_counts", "cpu_time_s"]:
                merged[key] = sum(chunk[key] for chunk in chunk_results)
            elif key in ["final_positions", "final_directions", "surface_hits"]:
                arrays = [chunk[key] for chunk in chunk_results if chunk[key].size > 0]
                if arrays:
                    merged[key] = np.concatenate(arrays, axis=1)
                else:
                    merged[key] = template[key]
            elif key == "sample_paths":
                merged[key] = {}
                for chunk in chunk_results:
                    for path_id, path_list in chunk[key].items():
                        if path_id not in merged[key]:
                            merged[key][path_id] = []
                        merged[key][path_id].extend(path_list)
            else:
                raise ValueError(f"Błąd łączenia: Nieznany klucz '{key}' w wynikach. Dodaj go do _merge_results!")

        return merged

    def get_results(self):
        return self.results

def build_detectors_from_config(config: SimConfig):
    detectors = []
    if config.detectors.num_full_paths > 0:
        detectors.append(PathTrackingDetector())
        
    if config.output.save_vertical_profile:
        detectors.append(VerticalFluxDetector())
        
    if config.output.save_flux_maps:
        detectors.append(BoundaryMapDetector())

    if config.output.save_heating_rates:
        detectors.append(HeatingRateDetector())
    
    return detectors

def parallel_simulation(config: SimConfig, scene: Scene):
    if config.engine.cpu_cores > 1:
        chunk_size = config.engine.num_photons // config.engine.cpu_cores
        remainder = config.engine.num_photons % config.engine.cpu_cores
        seeds = np.random.SeedSequence(config.engine.random_seed).spawn(config.engine.cpu_cores)

        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.engine.cpu_cores) as executor:
            for i in range(config.engine.cpu_cores):
                future = executor.submit(run_chunk, chunk_size + (remainder if i == 0 else 0), seeds[i], config, scene, i)
                futures.append(future)

        all_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        return all_results
    else:
        return [run_chunk(config.engine.num_photons, config.engine.random_seed, config, scene, 0)]
        

def run_chunk(chunk_size: int, seed, config: SimConfig, scene: Scene, i):
    new_engine_config = replace(
        config.engine,
        num_photons=chunk_size,
        random_seed=seed,        
    )
    new_detector_config = replace(
        config.detectors,
        num_full_paths=config.detectors.num_full_paths if i == 0 else 0
    )
    new_config = replace(config, engine=new_engine_config, detectors=new_detector_config)
    
    detectors = build_detectors_from_config(new_config)
    
    sim = Engine(new_config, scene, detectors)
    sim.run()
    
    return sim.get_results()