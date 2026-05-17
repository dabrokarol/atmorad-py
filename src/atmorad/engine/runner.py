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
        self.results = self.parallel_simulation(self.config, self.scene)
        end_time = time.perf_counter()
        self.results['simulation_time_s'] = end_time - start_time
        
    def _merge_incremental(self, first: dict, second: dict) -> dict:
        if not first:
            return second

        for key in first.keys():
            if key in ["measure_z", "layer_boundaries_z", "x_edges", "y_edges"]:
                continue
            elif key in ["flux_up", "flux_down", "surface_flux_map_2d", "toa_flux_map_2d", 
                        "heating_profile_1d", "scatter_counts", "cpu_time_s"]:
                first[key] += second[key]
            elif key in ["final_positions", "final_directions", "surface_hits"]:
                if second[key].size > 0:
                    first[key] = np.concatenate([first[key], second[key]], axis=1)
            elif key == "sample_paths":
                for path_id, path_list in second[key].items():
                    if path_id not in first[key]:
                        first[key][path_id] = []
                    first[key][path_id].extend(path_list)
        return first

    def get_results(self):
        return self.results
    
    def parallel_simulation(self, config: SimConfig, scene: Scene):
        total_photons = config.engine.num_photons
        batch_size = config.engine.batch_size
        cores = config.engine.cpu_cores
        
        batches = []
        while total_photons > 0:
            current_batch = min(batch_size, total_photons)
            batches.append(current_batch)
            total_photons -= current_batch
        num_batches = len(batches)
            
        seeds = np.random.SeedSequence(config.engine.random_seed).spawn(num_batches)
        
        all_results = {}
        
        if cores > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
                futures = []
                for i, (size, seed) in enumerate(zip(batches, seeds)):
                    future = executor.submit(run_chunk, size, seed, config, scene, i)
                    futures.append(future)
                    
                for future in concurrent.futures.as_completed(futures):
                    chunk_res = future.result()
                    all_results = self._merge_incremental(all_results, chunk_res)

            return all_results
        else:
            for i, (size, seed) in enumerate(zip(batches, seeds)):
                res = run_chunk(size, seed, config, scene, i)
                all_results = self._merge_incremental(all_results, res)
            return all_results

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