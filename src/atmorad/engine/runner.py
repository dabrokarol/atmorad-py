import time
import numpy as np
import concurrent.futures

from atmorad.environment.scene import Scene
from atmorad.engine.batch import PhotonBatch
from atmorad.engine.engine import Simulation
from atmorad.config.config import SimConfig
from atmorad.detectors.results import Results

class MCRadiation:
    def __init__(self, config: SimConfig, scene: Scene):
        self.config = config
        self.scene = scene

    def run(self):
        start_time = time.perf_counter()
        results = parallel_simulation(self.config, self.scene)
        self.results = Results.merge_all(results)
        end_time = time.perf_counter()
        self.results.simulation_time_s = end_time - start_time

    def get_results(self):
        return self.results


def parallel_simulation(config: SimConfig, scene: Scene):
    if config.cpu_cores > 1:
        chunk_size = config.num_photons // config.cpu_cores
        remainder = config.num_photons % config.cpu_cores
        seeds = np.random.SeedSequence(config.random_seed).spawn(config.cpu_cores)

        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.cpu_cores) as executor:
            for i in range(config.cpu_cores):
                future = executor.submit(run_chunk, chunk_size + (remainder if i == 0 else 0), seeds[i], config, scene, i)
                futures.append(future)

        all_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        return all_results
    else:
        return [run_chunk(config.num_photons, config.random_seed, config, scene, 0)]
        

def run_chunk(chunk_size: int, seed, config: SimConfig, scene: Scene, i):
    chunk_config = SimConfig(
        num_photons=chunk_size,
        num_track=config.num_track if i == 0 else 0,
        random_seed=seed,
        theta_sun_deg=config.theta_sun_deg,
        phi_sun_deg=config.phi_sun_deg,
        flux_measure_spacing=config.flux_measure_spacing
    )
    sim = Simulation(chunk_config, scene)
    sim.run()
    return sim.get_results()