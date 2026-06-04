import concurrent.futures
import logging
import multiprocessing
import signal
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Tuple

import numpy as np
import xarray as xr
from tqdm import tqdm

from atmorad.config.schemas import SimConfig
from atmorad.constants import CHECKPOINT_INTERVAL
from atmorad.detectors import DETECTORS, BaseDetector, PathTrackingDetector
from atmorad.environment import Scene
from atmorad.simulation import run_photon_batch

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
    cfg_engine = config.engine

    accumulated_results: dict[str, xr.Dataset] = {}

    simulated_photons = 0
    accumulated_time = 0.0

    if initial_state is not None:
        simulated_photons = int(initial_state.attrs.get("num_photons", 0))
        accumulated_time = float(initial_state.attrs.get("simulation_time_s", 0.0))

        for det_name in config.detectors.active:
            det_vars = [
                v
                for v in initial_state.data_vars
                if initial_state[v].attrs.get("source_detector") == det_name
            ]
            if det_vars:
                accumulated_results[det_name] = initial_state[det_vars]

    remaining_photons = cfg_engine.num_photons - simulated_photons

    if remaining_photons <= 0:
        logging.info("Target number of photons has already been reached. Skipping simulation loop.")
        if initial_state is not None:
            return initial_state
        return _build_final_dataset(
            config, accumulated_results, simulated_photons, accumulated_time
        )

    batches = _calculate_batches(remaining_photons, cfg_engine.batch_size)

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)

        if cfg_engine.num_threads > 1:
            results_generator = _yield_results_parallel(
                config,
                scene,
                quiet,
                batches,
                simulated_photons,
                cfg_engine.random_seed,
                cfg_engine.num_threads,
                temp_dir,
            )
        else:
            results_generator = _yield_results_serial(
                config, scene, quiet, batches, simulated_photons, cfg_engine.random_seed, temp_dir
            )

        run_start_time = time.perf_counter()

        try:
            for i, (chunk_dict, chunk_size) in enumerate(results_generator, start=1):
                simulated_photons += chunk_size

                for det_name, filepath in chunk_dict.items():
                    with xr.open_dataset(filepath) as chunk_ds:
                        chunk_ds.load()

                    if det_name not in accumulated_results:
                        accumulated_results[det_name] = chunk_ds
                    else:
                        det_class = DETECTORS[det_name]
                        merged_ds = det_class.merge_chunks(
                            [accumulated_results[det_name], chunk_ds]
                        )
                        accumulated_results[det_name] = merged_ds

                    filepath.unlink(missing_ok=True)

                if on_checkpoint and i % CHECKPOINT_INTERVAL == 0:
                    current_elapsed = time.perf_counter() - run_start_time
                    checkpoint_ds = _build_final_dataset(
                        config,
                        accumulated_results,
                        simulated_photons,
                        accumulated_time + current_elapsed,
                    )
                    on_checkpoint(checkpoint_ds)

        except KeyboardInterrupt:
            current_elapsed = time.perf_counter() - run_start_time
            checkpoint_ds = _build_final_dataset(
                config, accumulated_results, simulated_photons, accumulated_time + current_elapsed
            )
            if on_checkpoint:
                on_checkpoint(checkpoint_ds)
            raise

    final_elapsed = time.perf_counter() - run_start_time
    final_ds = _build_final_dataset(
        config, accumulated_results, simulated_photons, accumulated_time + final_elapsed
    )

    return final_ds


def _build_final_dataset(
    config: SimConfig,
    accumulated_results: dict[str, xr.Dataset],
    total_photons: int,
    sim_time: float,
) -> xr.Dataset:

    if not accumulated_results:
        return xr.Dataset()

    seen_vars = {}

    # unique name validation
    for det_name, ds in accumulated_results.items():
        for var_name in ds.data_vars:
            if var_name in seen_vars:
                raise ValueError(
                    f"Name collision: Variable '{var_name}' is defined by both "
                    f"'{seen_vars[var_name]}' and '{det_name}'. "
                    "Data variables must be globally unique."
                )
            seen_vars[var_name] = det_name
            # source in attributes
            ds[var_name].attrs["source_detector"] = det_name

    master_ds = xr.merge(list(accumulated_results.values()), combine_attrs="drop_conflicts")
    master_ds.attrs["simulation_time_s"] = sim_time
    master_ds.attrs["num_photons"] = total_photons
    master_ds.attrs["_simulation_config"] = config.model_dump_json()
    master_ds.attrs["active_detectors"] = list(accumulated_results.keys())
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
    temp_dir: Path,
) -> Iterator[Tuple[dict[str, Path], int]]:
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
                chunk_result = _run_chunk(size, chunk_seed, current_photons, temp_dir)

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
    temp_dir: Path,
) -> Iterator[Tuple[dict[str, Path], int]]:
    start_methods = multiprocessing.get_all_start_methods()
    start_method = "forkserver" if "forkserver" in start_methods else "spawn"
    ctx = multiprocessing.get_context(start_method)

    progress_value = ctx.Value("Q", 0)
    done_event = threading.Event()

    seeds = []
    photon_counts = []
    temp_dirs = []
    current_photons = start_photons

    for size in batches:
        seeds.append(np.random.SeedSequence((base_seed, current_photons)))
        photon_counts.append(current_photons)
        temp_dirs.append(temp_dir)
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
            results = executor.map(_run_chunk, batches, seeds, photon_counts, temp_dirs)
            for size, chunk_result in zip(batches, results):
                yield chunk_result, size
        except KeyboardInterrupt:
            done_event.set()
            executor.shutdown(wait=False, cancel_futures=True)
            for process in multiprocessing.active_children():
                try:
                    process.terminate()
                except Exception:
                    pass
            raise
        finally:
            done_event.set()
            executor.shutdown(wait=False, cancel_futures=True)


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
        total=total,
        initial=initial,
        desc=exp_str,
        unit=" photons",
        disable=quiet,
        smoothing=0,
        mininterval=0.3,
    ) as pbar:

        def update_pbar():
            last_val = 0
            while not done_event.wait(timeout=0.2):
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
    chunk_size: int, seed: np.random.SeedSequence, starting_photon_count: int, temp_dir: Path
) -> dict[str, Path]:
    global _global_config, _global_scene, _progress_value
    if _global_config is None or _global_scene is None:
        raise RuntimeError("Worker context not initialized")

    def update_progress_value(died_count):
        if _progress_value is not None and died_count > 0:
            with _progress_value.get_lock():
                _progress_value.value += died_count

    detectors: dict[str, BaseDetector] = {}
    for det_name in _global_config.detectors.active:
        detector_class = DETECTORS[det_name]
        if detector_class is PathTrackingDetector and starting_photon_count > 0:
            continue
        detectors[det_name] = detector_class(_global_scene, _global_config)

    chunk_ds_dict = run_photon_batch(
        _global_config, chunk_size, seed, _global_scene, detectors, update_progress_value
    )

    # save results to a disk to avoid python serialization
    saved_paths = {}
    chunk_id = uuid.uuid4().hex

    for det_name, ds in chunk_ds_dict.items():
        out_path = temp_dir / f"{det_name}_{chunk_id}.nc"
        ds.to_netcdf(out_path)
        ds.close()
        saved_paths[det_name] = out_path

    return saved_paths
