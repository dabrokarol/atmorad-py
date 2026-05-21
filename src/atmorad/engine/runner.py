import concurrent.futures
import multiprocessing
import time
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
            completed_batches, all_results = self.data_io.load_checkpoint()
        else:
            completed_batches = 0
            all_results = {}

        total_photons = self.context.config.engine.num_photons
        batch_size = self.context.config.engine.batch_size
        cores = self.context.config.engine.cpu_cores

        full_batches, remainder = divmod(total_photons, batch_size)
        batches = [batch_size] * full_batches
        if remainder > 0:
            batches.append(remainder)
        num_batches = len(batches)

        if completed_batches >= num_batches:
            return all_results

        seeds = np.random.SeedSequence(self.context.config.engine.random_seed).spawn(num_batches)

        batches = batches[completed_batches:]
        seeds = seeds[completed_batches:]

        if cores > 1:
            start_methods = multiprocessing.get_all_start_methods()
            start_method = "forkserver" if "forkserver" in start_methods else "spawn"
            ctx = multiprocessing.get_context(start_method)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=cores, mp_context=ctx
            ) as executor:
                futures = []
                for i, (size, seed) in enumerate(zip(batches, seeds), start=completed_batches):
                    future = executor.submit(run_chunk, size, seed, self.context, i)
                    futures.append(future)

                for i, future in tqdm(
                    enumerate(futures, start=completed_batches),
                    total=num_batches,
                    desc="Processed Photon Batches",
                    initial=completed_batches,
                ):
                    chunk_res = future.result()
                    all_results = merge_incremental(all_results, chunk_res)
                    i += 1
                    if i % CHECKPOINT_INTERVAL == 0:
                        self.data_io.save_checkpoint(completed_batches=i, results=all_results)

            return all_results
        else:
            for i, (size, seed) in tqdm(
                enumerate(zip(batches, seeds), start=completed_batches),
                total=num_batches,
                desc="Processed Photon Batches",
                initial=completed_batches,
            ):
                res = run_chunk(size, seed, self.context, i)
                all_results = merge_incremental(all_results, res)
                i += 1
                if i % CHECKPOINT_INTERVAL == 0:
                    self.data_io.save_checkpoint(completed_batches=i, results=all_results)
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
