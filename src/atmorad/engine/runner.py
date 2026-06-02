import concurrent.futures
import logging
import multiprocessing
import signal
import threading
import time
from contextlib import contextmanager
from typing import Callable

import numpy as np
from tqdm import tqdm

from atmorad.constants import CHECKPOINT_INTERVAL
from atmorad.models import SimContext, SimResults

from .core import Engine

_global_context = None
_progress_value = None


class MCRadiationRunner:
    def __init__(
        self,
        context: SimContext,
        quiet: bool = False,
        on_checkpoint: Callable[[SimResults], None] | None = None,
        on_finish: Callable[[SimResults], None] | None = None,
        load_checkpoint_fn: Callable[[], SimResults | None] | None = None,
        on_cleanup: Callable[[], None] | None = None,
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
        all_results = self._load_initial_state()
        all_results.config = self.context.config

        cfg_engine = self.context.config.engine
        total_photons = cfg_engine.num_photons
        simulated_photons = all_results.total_photons
        remaining_photons = total_photons - simulated_photons

        if remaining_photons <= 0:
            logging.info(
                "Target number of photons has already been reached. Skipping simulation loop."
            )
            return all_results

        batches = self._calculate_batches(remaining_photons, cfg_engine.batch_size)

        if cfg_engine.cpu_cores > 1:
            results_generator = self._yield_results_parallel(
                batches,
                simulated_photons,
                cfg_engine.random_seed,
                cfg_engine.cpu_cores,
            )
        else:
            results_generator = self._yield_results_serial(
                batches,
                simulated_photons,
                cfg_engine.random_seed,
            )

        accumulated_time = all_results.engine_result.simulation_time_s
        run_start_time = time.perf_counter()

        try:
            for i, (chunk_res, chunk_size) in enumerate(results_generator, start=1):
                chunk_res.total_photons = chunk_size
                all_results = all_results.merge(chunk_res)

                if (i + 1) % CHECKPOINT_INTERVAL == 0:
                    current_elapsed = time.perf_counter() - run_start_time
                    all_results.engine_result.simulation_time_s = accumulated_time + current_elapsed

                    if self.on_checkpoint:
                        self.on_checkpoint(all_results)
        except KeyboardInterrupt:
            current_elapsed = time.perf_counter() - run_start_time
            all_results.engine_result.simulation_time_s = accumulated_time + current_elapsed

            if self.on_checkpoint:
                self.on_checkpoint(all_results)
            raise

        final_elapsed = time.perf_counter() - run_start_time
        all_results.engine_result.simulation_time_s = accumulated_time + final_elapsed

        return all_results

    def _calculate_batches(self, remaining_photons: int, batch_size: int):
        full_batches, remainder = divmod(remaining_photons, batch_size)
        batches = [batch_size] * full_batches
        if remainder > 0:
            batches.append(remainder)
        return batches

    def _yield_results_serial(self, batches: list[int], start_photons: int, base_seed: int):
        progress_value = multiprocessing.Value("Q", 0)
        done_event = threading.Event()
        global _global_context, _progress_value
        _global_context = self.context
        _progress_value = progress_value

        current_photons = start_photons
        with self._track_progress(
            sum(batches) + start_photons, start_photons, progress_value, done_event
        ):
            try:
                for size in batches:
                    chunk_seed = np.random.SeedSequence((base_seed, current_photons))
                    chunk_result = run_chunk(size, chunk_seed, current_photons)

                    yield chunk_result, size
                    current_photons += size
            finally:
                done_event.set()

    def _load_initial_state(self) -> SimResults:
        cfg = self.context.config

        if cfg.engine.resume_from_checkpoint and self.load_checkpoint_fn:
            if results := self.load_checkpoint_fn():
                return results

        return SimResults()

    def _yield_results_parallel(
        self, batches: list[int], start_photons: int, base_seed: int, cores: int
    ):
        start_methods = multiprocessing.get_all_start_methods()
        start_method = "forkserver" if "forkserver" in start_methods else "spawn"
        ctx = multiprocessing.get_context(start_method)

        progress_value = ctx.Value("Q", 0)
        done_event = threading.Event()

        seeds = []
        photon_counts = []
        current_photons = start_photons
        for size in batches:
            seeds.append(np.random.SeedSequence((base_seed, current_photons)))
            photon_counts.append(current_photons)
            current_photons += size

        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=cores,
            initializer=init_worker,
            mp_context=ctx,
            initargs=(self.context, progress_value),
        )
        with self._track_progress(
            sum(batches) + start_photons, start_photons, progress_value, done_event
        ):
            try:
                results = executor.map(run_chunk, batches, seeds, photon_counts)
                for size, chunk_result in zip(batches, results):
                    yield chunk_result, size
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                for process in multiprocessing.active_children():
                    try:
                        process.terminate()
                    except Exception:
                        pass
                raise
            finally:
                executor.shutdown(wait=True)
                done_event.set()

    @contextmanager
    def _track_progress(
        self, total: int, initial: int, progress_value, done_event: threading.Event
    ):
        meta = self.context.config.metadata
        exp_str = (
            f"{meta.experiment_name}/{meta.scenario_name}"
            if meta.scenario_name
            else meta.experiment_name
        )

        with tqdm(
            total=total,
            initial=initial,
            desc=exp_str,
            unit=" photons",
            disable=self.quiet,
            smoothing=0,
        ) as pbar:

            def update_pbar():
                last_val = 0
                while not done_event.wait(timeout=0.1):
                    current_val = progress_value.value
                    if current_val > last_val:
                        pbar.update(current_val - last_val)
                        last_val = current_val

                current_val = progress_value.value
                if current_val > last_val:
                    pbar.update(current_val - last_val)

            monitor_thread = threading.Thread(target=update_pbar)
            monitor_thread.start()
            try:
                yield
            finally:
                monitor_thread.join()
                if not self.quiet:
                    pbar.refresh()


def init_worker(context: SimContext, progress_value):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _progress_value, _global_context
    _global_context = context
    _progress_value = progress_value


def run_chunk(
    chunk_size: int,
    seed: np.random.SeedSequence,
    starting_photon_count: int,
) -> SimResults:

    global _global_context, _progress_value
    if _global_context is None:
        raise RuntimeError("Worker context not initialized")

    new_config = _global_context.config.model_copy(deep=False)
    new_config.engine = _global_context.config.engine.model_copy()
    new_config.engine.num_photons = chunk_size
    new_config.engine.random_seed = seed

    if starting_photon_count > 0:
        new_config.detectors = _global_context.config.detectors.model_copy()
        new_config.detectors.num_full_paths = 0

    def update_progress_value(died_count):
        if _progress_value is not None and died_count > 0:
            with _progress_value.get_lock():
                _progress_value.value += died_count

    sim = Engine(new_config, _global_context.scene, update_progress_value)
    sim.run()

    return sim.get_results()
