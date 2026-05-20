import concurrent.futures
import multiprocessing
import time
from dataclasses import replace

import numpy as np
from tqdm import tqdm

from atmorad.detectors import build_detectors_from_config, merge_incremental
from atmorad.models import SimContext

from .core import Engine


class MCRadiationRunner:
    def __init__(self, context: SimContext):
        self.context = context

    def run(self):
        start_time = time.perf_counter()
        self.results = self.parallel_simulation()
        end_time = time.perf_counter()
        self.results["simulation_time_s"] = end_time - start_time

    def get_results(self):
        return self.results

    def parallel_simulation(self):
        total_photons = self.context.config.engine.num_photons
        batch_size = self.context.config.engine.batch_size
        cores = self.context.config.engine.cpu_cores

        full_batches, remainder = divmod(total_photons, batch_size)
        batches = [batch_size] * full_batches
        if remainder > 0:
            batches.append(remainder)
        num_batches = len(batches)

        seeds = np.random.SeedSequence(self.context.config.engine.random_seed).spawn(num_batches)

        all_results = {}

        if cores > 1:
            ctx = multiprocessing.get_context("forkserver")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=cores, mp_context=ctx
            ) as executor:
                futures = []
                for i, (size, seed) in enumerate(zip(batches, seeds)):
                    future = executor.submit(run_chunk, size, seed, self.context, i)
                    futures.append(future)

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=num_batches,
                    desc="Processed Photon Batches",
                ):
                    chunk_res = future.result()
                    all_results = merge_incremental(all_results, chunk_res)

            return all_results
        else:
            for i, (size, seed) in tqdm(
                enumerate(zip(batches, seeds)), total=num_batches, desc="Processed Photon Batches"
            ):
                res = run_chunk(size, seed, self.context, i)
                all_results = merge_incremental(all_results, res)
            return all_results


def run_chunk(chunk_size: int, seed: np.random.SeedSequence, context: SimContext, i: int) -> dict:

    new_engine_config = replace(
        context.config.engine,
        num_photons=chunk_size,
        random_seed=seed,
    )
    new_detector_config = replace(
        context.config.detectors,
        num_full_paths=context.config.detectors.num_full_paths if i == 0 else 0,
    )
    new_config = replace(context.config, engine=new_engine_config, detectors=new_detector_config)

    detectors = build_detectors_from_config(new_config)

    sim = Engine(new_config, context.scene, detectors)
    sim.run()

    return sim.get_results()
