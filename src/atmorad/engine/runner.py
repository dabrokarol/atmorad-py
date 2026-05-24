import concurrent.futures
import logging
import multiprocessing
import time
from typing import Callable

import numpy as np
from tqdm import tqdm

from atmorad.constants import CHECKPOINT_INTERVAL
from atmorad.models import SimContext, SimulationResults

from .core import Engine


class MCRadiationRunner:
    def __init__(
        self,
        context: SimContext,
        quiet: bool = False,
        on_checkpoint: Callable[[int, SimulationResults], None] = None,
        on_finish: Callable[[SimulationResults], None] = None,
        load_checkpoint_fn: Callable[[], tuple] = None,
        on_cleanup: Callable[[], None] = None,
    ):
        self.context = context
        self.quiet = quiet

        self.on_checkpoint = on_checkpoint
        self.on_finish = on_finish
        self.load_checkpoint_fn = load_checkpoint_fn
        self.on_cleanup = on_cleanup

    def run(self):
        self.results = self._run_simulation()

        if self.on_finish:
            self.on_finish(self.results)
        if self.on_cleanup:
            self.on_cleanup()

    def get_results(self):
        return self.results

    def _run_simulation(self):
        simulated_photons, all_results = self._load_initial_state()
        total_photons = self.context.config.engine.num_photons
        remaining_photons = total_photons - simulated_photons

        accumulated_time = all_results.engine.simulation_time_s

        if remaining_photons <= 0:
            logging.info(
                "Target number of photons has already been reached. Skipping simulation loop."
            )
            return all_results

        batch_size = self.context.config.engine.batch_size
        batches = self._calculate_batches(remaining_photons, batch_size)

        cores = self.context.config.engine.cpu_cores
        base_seed = self.context.config.engine.random_seed

        if cores > 1:
            results_generator = self._yield_results_parallel(
                batches, simulated_photons, base_seed, cores
            )
        else:
            results_generator = self._yield_results_serial(batches, simulated_photons, base_seed)

        current_photons = simulated_photons

        run_start_time = time.perf_counter()

        with tqdm(
            total=total_photons,
            initial=simulated_photons,
            desc="Simulating Photons",
            unit=" photons",
            disable=self.quiet,
            smoothing=0.3,
        ) as pbar:
            for i, (chunk_res, chunk_size) in enumerate(results_generator):
                current_photons += chunk_size
                pbar.update(chunk_size)
                all_results = all_results.merge(chunk_res)

                if (i + 1) % CHECKPOINT_INTERVAL == 0:
                    current_elapsed = time.perf_counter() - run_start_time
                    all_results.engine.simulation_time_s = accumulated_time + current_elapsed

                    if self.on_checkpoint:
                        self.on_checkpoint(current_photons, all_results)

        final_elapsed = time.perf_counter() - run_start_time
        all_results.engine.simulation_time_s = accumulated_time + final_elapsed
        all_results.config = self.context.config

        return all_results

    def _load_initial_state(self):
        if not self.context.config.engine.resume_from_checkpoint:
            return 0, SimulationResults()

        if not self.load_checkpoint_fn:
            return 0, SimulationResults()

        simulated_photons, all_results, saved_config = self.load_checkpoint_fn()

        if saved_config is not None:
            if not self.context.config.is_compatible_for_resume(saved_config):
                logging.warning(
                    "Configuration mismatch! The current setup differs from the saved simulation checkpoint. "
                    "Starting a fresh simulation."
                )
                return 0, SimulationResults()
            return simulated_photons, all_results

        return 0, SimulationResults()

    def _calculate_batches(self, remaining_photons: int, batch_size: int):
        full_batches, remainder = divmod(remaining_photons, batch_size)
        batches = [batch_size] * full_batches
        if remainder > 0:
            batches.append(remainder)
        return batches

    def _yield_results_serial(self, batches: list[int], start_photons: int, base_seed: int):
        current_photons = start_photons
        for size in batches:
            chunk_seed = np.random.SeedSequence((base_seed, current_photons))
            chunk_result = run_chunk(size, chunk_seed, self.context, current_photons)

            yield chunk_result, size
            current_photons += size

    def _yield_results_parallel(
        self, batches: list[int], start_photons: int, base_seed: int, cores: int
    ):
        start_methods = multiprocessing.get_all_start_methods()
        start_method = "forkserver" if "forkserver" in start_methods else "spawn"
        ctx = multiprocessing.get_context(start_method)

        with concurrent.futures.ProcessPoolExecutor(max_workers=cores, mp_context=ctx) as executor:
            futures = []
            current_photons = start_photons

            for size in batches:
                chunk_seed = np.random.SeedSequence((base_seed, current_photons))
                future = executor.submit(run_chunk, size, chunk_seed, self.context, current_photons)
                futures.append((future, size))
                current_photons += size

            for future, size in futures:
                yield future.result(), size


def run_chunk(
    chunk_size: int, seed: np.random.SeedSequence, context: SimContext, starting_photon_count: int
) -> SimulationResults:
    new_engine_config = context.config.engine.model_copy(
        update={
            "num_photons": chunk_size,
            "random_seed": seed,
        }
    )

    new_detector_config = context.config.detectors.model_copy(
        update={
            "num_full_paths": context.config.detectors.num_full_paths
            if starting_photon_count == 0
            else 0,
        }
    )

    new_config = context.config.model_copy(
        update={"engine": new_engine_config, "detectors": new_detector_config}
    )

    sim = Engine(new_config, context.scene)
    sim.run()

    return sim.get_results()
