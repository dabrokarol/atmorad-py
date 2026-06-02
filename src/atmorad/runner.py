import concurrent.futures
import logging
import multiprocessing
import signal
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Tuple

import numpy as np
import xarray as xr
from tqdm import tqdm

from atmorad.config import SimConfig
from atmorad.constants import CHECKPOINT_INTERVAL
from atmorad.detectors import DETECTORS
from atmorad.environment import Scene
from atmorad.simulator import run_photon_batch

_global_config = None
_global_scene = None
_progress_value = None


def execute_simulation(
    config: SimConfig,
    scene: Scene,
    initial_state: xr.Dataset | None = None,
    quiet: bool = False,
    on_checkpoint: Callable[[xr.Dataset], None] | None = None,
) -> xr.Dataset:
    """
    Główna funkcja orkiestrująca symulację Monte Carlo.
    Zarządza podziałem na paczki (batches), zrównoleglaniem i zapisem punktów kontrolnych.
    """
    cfg_engine = config.engine
    chunk_results = {det_name: [] for det_name in config.detectors.active}

    simulated_photons = 0
    accumulated_time = 0.0

    # 1. Odtwarzanie stanu początkowego (Checkpoint)
    if initial_state is not None:
        simulated_photons = int(initial_state.attrs.get("num_photons", 0))
        accumulated_time = float(initial_state.attrs.get("engine_simulation_time_s", 0.0))

        # Rozdzielamy zmienne z powrotem do detektorów (usuwając prefixy)
        for det_name in config.detectors.active:
            prefix = f"{det_name}_"
            det_vars = [v for v in initial_state.data_vars if str(v).startswith(prefix)]
            if det_vars:
                det_ds = initial_state[det_vars].rename_vars(
                    {v: str(v)[len(prefix) :] for v in det_vars}
                )
                chunk_results[det_name].append(det_ds)

    remaining_photons = cfg_engine.num_photons - simulated_photons

    if remaining_photons <= 0:
        logging.info("Target number of photons has already been reached. Skipping simulation loop.")
        if initial_state is not None:
            return initial_state
        return _build_final_dataset(config, chunk_results, simulated_photons, accumulated_time)

    batches = _calculate_batches(remaining_photons, cfg_engine.batch_size)

    if cfg_engine.cpu_cores > 1:
        results_generator = _yield_results_parallel(
            config,
            scene,
            quiet,
            batches,
            simulated_photons,
            cfg_engine.random_seed,
            cfg_engine.cpu_cores,
        )
    else:
        results_generator = _yield_results_serial(
            config, scene, quiet, batches, simulated_photons, cfg_engine.random_seed
        )

    run_start_time = time.perf_counter()

    try:
        for i, (chunk_dict, chunk_size) in enumerate(results_generator, start=1):
            simulated_photons += chunk_size

            for det_name, det_ds in chunk_dict.items():
                chunk_results[det_name].append(det_ds)

            if on_checkpoint and i % CHECKPOINT_INTERVAL == 0:
                current_elapsed = time.perf_counter() - run_start_time
                checkpoint_ds = _build_final_dataset(
                    config, chunk_results, simulated_photons, accumulated_time + current_elapsed
                )
                on_checkpoint(checkpoint_ds)

    except KeyboardInterrupt:
        current_elapsed = time.perf_counter() - run_start_time
        checkpoint_ds = _build_final_dataset(
            config, chunk_results, simulated_photons, accumulated_time + current_elapsed
        )
        if on_checkpoint:
            on_checkpoint(checkpoint_ds)
        raise

    final_elapsed = time.perf_counter() - run_start_time
    final_ds = _build_final_dataset(
        config, chunk_results, simulated_photons, accumulated_time + final_elapsed
    )

    return final_ds


def _build_final_dataset(
    config: SimConfig,
    chunk_results: dict[str, list[xr.Dataset]],
    total_photons: int,
    sim_time: float,
) -> xr.Dataset:
    final_components = []

    for det_name in config.detectors.active:
        det_class = DETECTORS[det_name]

        merged_det_ds = det_class.merge_chunks(chunk_results[det_name])

        rename_dict = {}
        for name in merged_det_ds.variables:
            rename_dict[str(name)] = f"{det_name}_{name}"

        for dim in merged_det_ds.dims:
            if str(dim) not in rename_dict:
                rename_dict[str(dim)] = f"{det_name}_{dim}"

        merged_det_ds = merged_det_ds.rename(rename_dict)
        final_components.append(merged_det_ds)

    if not final_components:
        master_ds = xr.Dataset()
    else:
        master_ds = xr.merge(final_components, combine_attrs="no_conflicts")

    master_ds.attrs["num_photons"] = total_photons
    master_ds.attrs["engine_simulation_time_s"] = sim_time
    master_ds.attrs["is_normalized"] = 0
    master_ds.attrs["_simulation_config"] = config.model_dump_json()
    master_ds.attrs.update(config.get_experiment_attributes())

    return master_ds


def _calculate_batches(remaining_photons: int, batch_size: int) -> list[int]:
    full_batches, remainder = divmod(remaining_photons, batch_size)
    batches = [batch_size] * full_batches
    if remainder > 0:
        batches.append(remainder)
    return batches


def _yield_results_serial(
    config: SimConfig,
    scene: Scene,
    quiet: bool,
    batches: list[int],
    start_photons: int,
    base_seed: int,
) -> Iterator[Tuple[dict[str, xr.Dataset], int]]:
    progress_value = multiprocessing.Value("Q", 0)
    done_event = threading.Event()

    global _global_config, _global_scene, _progress_value
    _global_config = config
    _global_scene = scene
    _progress_value = progress_value

    current_photons = start_photons
    total_target = sum(batches) + start_photons

    with _track_progress(config, quiet, total_target, start_photons, progress_value, done_event):
        try:
            for size in batches:
                chunk_seed = np.random.SeedSequence((base_seed, current_photons))
                chunk_result = _run_chunk(size, chunk_seed, current_photons)

                yield chunk_result, size
                current_photons += size
        finally:
            done_event.set()


def _yield_results_parallel(
    config: SimConfig,
    scene: Scene,
    quiet: bool,
    batches: list[int],
    start_photons: int,
    base_seed: int,
    cores: int,
) -> Iterator[Tuple[dict[str, xr.Dataset], int]]:
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

    total_target = sum(batches) + start_photons

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=cores,
        initializer=_init_worker,
        mp_context=ctx,
        initargs=(config, scene, progress_value),
    )

    with _track_progress(config, quiet, total_target, start_photons, progress_value, done_event):
        try:
            results = executor.map(_run_chunk, batches, seeds, photon_counts)
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
    config: SimConfig,
    quiet: bool,
    total: int,
    initial: int,
    progress_value,
    done_event: threading.Event,
):
    meta = config.metadata
    exp_str = (
        f"{meta.experiment_name}/{meta.scenario_name}"
        if meta.scenario_name
        else meta.experiment_name
    )

    with tqdm(
        total=total, initial=initial, desc=exp_str, unit=" photons", disable=quiet, smoothing=0
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
            if not quiet:
                pbar.refresh()


def _init_worker(config: SimConfig, scene: Scene, progress_value):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _progress_value, _global_config, _global_scene
    _global_config = config
    _global_scene = scene
    _progress_value = progress_value


def _run_chunk(
    chunk_size: int, seed: np.random.SeedSequence, starting_photon_count: int
) -> dict[str, xr.Dataset]:
    global _global_config, _global_scene, _progress_value
    if _global_config is None or _global_scene is None:
        raise RuntimeError("Worker context not initialized")

    new_config = _global_config.model_copy(deep=False)
    new_config.engine = _global_config.engine.model_copy()
    new_config.engine.num_photons = chunk_size
    new_config.engine.random_seed = seed

    if starting_photon_count > 0:
        new_config.detectors = _global_config.detectors.model_copy()
        new_config.detectors.num_full_paths = 0

    def update_progress_value(died_count):
        if _progress_value is not None and died_count > 0:
            with _progress_value.get_lock():
                _progress_value.value += died_count

    return run_photon_batch(new_config, _global_scene, update_progress_value)
