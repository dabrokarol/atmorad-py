import time
import numpy as np
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from dataclasses import replace

from atmorad.environment import Scene, Atmosphere
from atmorad.engine.core import Engine
from atmorad.config import SimConfig
from atmorad.detectors import build_detectors_from_config, merge_incremental

class MCRadiationRunner:
    def __init__(self, config: SimConfig, scene: Scene):
        self.config = config
        self.scene = scene

    def run(self):
        start_time = time.perf_counter()
        self.results = self.parallel_simulation(self.config, self.scene)
        end_time = time.perf_counter()
        self.results['simulation_time_s'] = end_time - start_time
        
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
            ctx = multiprocessing.get_context("forkserver")
            with concurrent.futures.ProcessPoolExecutor(max_workers=cores, mp_context=ctx) as executor:
                futures = []
                for i, (size, seed) in enumerate(zip(batches, seeds)):
                    future = executor.submit(run_chunk, size, seed, config, scene, i)
                    futures.append(future)
                    
                for future in tqdm(concurrent.futures.as_completed(futures), total=num_batches, desc="Processed Photon Batches"):
                    chunk_res = future.result()
                    all_results = merge_incremental(all_results, chunk_res)

            return all_results
        else:
            for i, (size, seed) in tqdm(enumerate(zip(batches, seeds)), total=num_batches, desc="Processed Photon Batches"):
                res = run_chunk(size, seed, config, scene, i)
                all_results = merge_incremental(all_results, res)
            return all_results


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