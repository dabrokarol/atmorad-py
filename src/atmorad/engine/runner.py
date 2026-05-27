import concurrent.futures
import logging
import multiprocessing
import queue
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
        progress_queue = multiprocessing.Manager().Queue()

        if cfg_engine.cpu_cores > 1:
            results_generator = self._yield_results_parallel(
                batches,
                simulated_photons,
                cfg_engine.random_seed,
                cfg_engine.cpu_cores,
                progress_queue,
            )
        else:
            results_generator = self._yield_results_serial(
                batches, simulated_photons, cfg_engine.random_seed, progress_queue
            )

        accumulated_time = all_results.engine_result.simulation_time_s
        run_start_time = time.perf_counter()

        with self._track_progress(total_photons, simulated_photons, progress_queue):
            try:
                for i, (chunk_res, chunk_size) in enumerate(results_generator, start=1):
                    chunk_res.total_photons = chunk_size
                    all_results = all_results.merge(chunk_res)

                    if i % CHECKPOINT_INTERVAL == 0:
                        current_elapsed = time.perf_counter() - run_start_time
                        all_results.engine_result.simulation_time_s = (
                            accumulated_time + current_elapsed
                        )

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
        all_results.config = self.context.config

        return all_results

    def _calculate_batches(self, remaining_photons: int, batch_size: int):
        full_batches, remainder = divmod(remaining_photons, batch_size)
        batches = [batch_size] * full_batches
        if remainder > 0:
            batches.append(remainder)
        return batches

    def _yield_results_serial(
        self, batches: list[int], start_photons: int, base_seed: int, progress_queue
    ):
        _set_global_context(self.context)
        current_photons = start_photons
        for size in batches:
            chunk_seed = np.random.SeedSequence((base_seed, current_photons))
            chunk_result = run_chunk(size, chunk_seed, current_photons, progress_queue)

            yield chunk_result, size
            current_photons += size

    def _load_initial_state(self) -> SimResults:
        cfg = self.context.config

        if cfg.engine.resume_from_checkpoint and self.load_checkpoint_fn:
            if (results := self.load_checkpoint_fn()) and results.config:
                if cfg.is_compatible_for_resume(results.config):
                    return results

                logging.warning(
                    "Configuration mismatch! The current setup differs from the saved simulation checkpoint. "
                    "Starting a fresh simulation."
                )

        return SimResults()

    def _yield_results_parallel(
        self, batches: list[int], start_photons: int, base_seed: int, cores: int, progress_queue
    ):
        start_methods = multiprocessing.get_all_start_methods()
        start_method = "forkserver" if "forkserver" in start_methods else "spawn"
        ctx = multiprocessing.get_context(start_method)

        seeds = []
        photon_counts = []
        current_photons = start_photons
        for size in batches:
            seeds.append(np.random.SeedSequence((base_seed, current_photons)))
            photon_counts.append(current_photons)
            current_photons += size

        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=cores, initializer=init_worker, mp_context=ctx, initargs=(self.context,)
        )
        queues = [progress_queue] * len(batches)
        try:
            results = executor.map(run_chunk, batches, seeds, photon_counts, queues)

            for size, chunk_result in zip(batches, results):
                yield chunk_result, size
        except KeyboardInterrupt:
            workers = list(executor._processes.values())
            executor.shutdown(wait=False, cancel_futures=True)
            for process in workers:
                try:
                    process.terminate()
                except Exception:
                    pass
            raise
        finally:
            executor.shutdown(wait=True)

    @contextmanager
    def _track_progress(self, total: int, initial: int, progress_queue):
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
            smoothing=0.01,
        ) as pbar:

            def update_pbar():
                while True:
                    try:
                        died = progress_queue.get(timeout=0.1)
                        if died == "DONE":
                            break
                        if not self.quiet:
                            pbar.update(died)
                    except queue.Empty:
                        pass
                    except Exception:
                        break

            monitor_thread = threading.Thread(target=update_pbar)
            monitor_thread.start()

            try:
                yield
            finally:
                try:
                    progress_queue.put("DONE")
                except Exception:
                    pass
                monitor_thread.join()


def _set_global_context(context: SimContext):
    global _global_context
    _global_context = context


def init_worker(context: SimContext):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _set_global_context(context)


def run_chunk(
    chunk_size: int,
    seed: np.random.SeedSequence,
    starting_photon_count: int,
    progress_queue,
) -> SimResults:

    global _global_context
    if _global_context is None:
        raise RuntimeError("Worker context not initialized")

    new_config = _global_context.config.model_copy(deep=False)
    new_config.engine = _global_context.config.engine.model_copy()
    new_config.engine.num_photons = chunk_size
    new_config.engine.random_seed = seed

    if starting_photon_count > 0:
        new_config.detectors = _global_context.config.detectors.model_copy()
        new_config.detectors.num_full_paths = 0

    def put_in_queue(died_count):
        if progress_queue is not None and died_count > 0:
            progress_queue.put(died_count)

    sim = Engine(new_config, _global_context.scene, put_in_queue)
    sim.run()

    return sim.get_results()
