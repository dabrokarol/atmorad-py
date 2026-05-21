import concurrent.futures
import multiprocessing
import time
import logging
from dataclasses import replace

import numpy as np
from tqdm import tqdm

from atmorad.constants import CHECKPOINT_INTERVAL
from atmorad.detectors import build_detectors_from_config, merge_incremental
from atmorad.models import SimContext
from atmorad.output import DataIO

from .core import Engine


class MCRadiationRunner:
    def __init__(self, context: SimContext):
        self.context = context
        self.data_io = DataIO(context.config)

    def run(self):
        start_time = time.perf_counter()
        self.results = self.parallel_simulation()
        end_time = time.perf_counter()

        if "simulation_time_s" in self.results:
            self.results["simulation_time_s"] += end_time - start_time
        else:
            self.results["simulation_time_s"] = end_time - start_time

    def get_results(self):
        return self.results

    def parallel_simulation(self):
        if self.context.config.engine.resume_from_checkpoint:
            simulated_photons, all_results, saved_config = self.data_io.load_checkpoint()

            if saved_config is not None:
                if not self.context.config.is_compatible_for_resume(saved_config):
                    logging.error(
                        "Configuration mismatch! The current setup differs from the saved simulation state."
                        "Starting a fresh simulation."
                    )
                    simulated_photons = 0
                    all_results = {}
                    saved_config = None
        else:
            simulated_photons = 0
            all_results = {}
            saved_config = None

        total_photons = self.context.config.engine.num_photons
        batch_size = self.context.config.engine.batch_size
        cores = self.context.config.engine.cpu_cores
        base_seed = self.context.config.engine.random_seed

        remaining_photons = total_photons - simulated_photons

        if remaining_photons <= 0 and saved_config == self.context.config:
            return all_results

        full_batches, remainder = divmod(remaining_photons, batch_size)
        batches = [batch_size] * full_batches
        if remainder > 0:
            batches.append(remainder)

        current_simulated_photons = simulated_photons

        if cores > 1:
            start_methods = multiprocessing.get_all_start_methods()
            start_method = "forkserver" if "forkserver" in start_methods else "spawn"
            ctx = multiprocessing.get_context(start_method)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=cores, mp_context=ctx
            ) as executor:
                futures = []
                _temp_photons = current_simulated_photons
                for i, size in enumerate(batches):
                    chunk_seed = np.random.SeedSequence((base_seed, _temp_photons))
                    future = executor.submit(
                        run_chunk, size, chunk_seed, self.context, _temp_photons
                    )
                    futures.append(future)
                    _temp_photons += size
                with tqdm(
                    total=total_photons, initial=simulated_photons, desc="Simulating Photons"
                ) as pbar:
                    for i, future in enumerate(futures):
                        chunk_res = future.result()
                        all_results = merge_incremental(all_results, chunk_res)

                        chunk_size = batches[i]
                        current_simulated_photons += chunk_size
                        pbar.update(chunk_size)

                        if (i + 1) % CHECKPOINT_INTERVAL == 0:
                            self.data_io.save_checkpoint(
                                simulated_photons=current_simulated_photons,
                                results=all_results,
                                config=self.context.config,
                            )

        else:
            with tqdm(
                total=total_photons, initial=simulated_photons, desc="Simulating Photons"
            ) as pbar:
                for i, size in enumerate(batches):
                    chunk_seed = np.random.SeedSequence((base_seed, current_simulated_photons))
                    chunk_res = run_chunk(size, chunk_seed, self.context, current_simulated_photons)
                    all_results = merge_incremental(all_results, chunk_res)

                    current_simulated_photons += size
                    pbar.update(size)

                    if (i + 1) % CHECKPOINT_INTERVAL == 0:
                        self.data_io.save_checkpoint(
                            simulated_photons=current_simulated_photons,
                            results=all_results,
                            config=self.context.config,
                        )
        
        logging.info("Simulation complete. Saving final results to disk...")
        self.data_io.save_metadata(self.context.config, self.results)
        self.data_io.save_results(self.results)
        
        return all_results


def run_chunk(
    chunk_size: int, seed: np.random.SeedSequence, context: SimContext, starting_photon_count: int
) -> dict:

    new_engine_config = replace(
        context.config.engine,
        num_photons=chunk_size,
        random_seed=seed,
    )
    new_detector_config = replace(
        context.config.detectors,
        num_full_paths=context.config.detectors.num_full_paths if starting_photon_count == 0 else 0,
    )
    new_config = replace(context.config, engine=new_engine_config, detectors=new_detector_config)

    detectors = build_detectors_from_config(new_config)

    sim = Engine(new_config, context.scene, detectors)
    sim.run()

    return sim.get_results()
