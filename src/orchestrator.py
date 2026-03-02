import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


async def run_worker_queue(
    data: list[dict],
    process_fns: list[Callable[[dict], Coroutine[Any, Any, dict | None]]],
    output_path: str,
    batch_size: int = 1,
    gpu_stagger_seconds: float = 30.0,
):
    """Distribute work items across GPU workers and write results to a JSONL file.

    Args:
        data: List of example dicts to process.
        process_fns: One async callable per GPU. Each takes an example dict
                     and returns a result dict (or None).
        output_path: Path to output JSONL file (results are appended).
        batch_size: Number of concurrent examples per GPU.
        gpu_stagger_seconds: Delay between starting each GPU worker.
    """
    num_gpus = len(process_fns)
    total = len(data)
    total_concurrent = batch_size * num_gpus

    work_queue = asyncio.Queue()
    for example in data:
        work_queue.put_nowait(example)

    completed = 0
    in_flight = 0
    all_done = asyncio.Event()
    state_lock = asyncio.Lock()

    async def write_result(result, example_id, gpu_idx):
        nonlocal completed, in_flight
        async with state_lock:
            completed += 1
            in_flight -= 1
            if result is not None:
                with open(output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
                logger.info(
                    f"[Progress: {completed}/{total}] [GPU {gpu_idx}] Wrote result for example {result['id']}"
                )
            else:
                logger.warning(
                    f"[Progress: {completed}/{total}] [GPU {gpu_idx}] Example {example_id} returned None"
                )

            if completed >= total:
                all_done.set()

    async def gpu_worker(gpu_idx: int):
        nonlocal in_flight, completed
        process_fn = process_fns[gpu_idx]
        active_tasks = set()

        while not all_done.is_set() or active_tasks:
            # Fill up to batch_size concurrent tasks for this GPU
            while len(active_tasks) < batch_size and not all_done.is_set():
                # Atomic: get from queue and increment in_flight together
                async with state_lock:
                    try:
                        example = work_queue.get_nowait()
                        in_flight += 1
                    except asyncio.QueueEmpty:
                        example = None
                if example is None:
                    break
                task = asyncio.create_task(process_fn(example))
                task.example_id = example["id"]
                active_tasks.add(task)

            if not active_tasks:
                # Queue is empty and no local tasks - check if globally done
                async with state_lock:
                    if completed >= total or (work_queue.empty() and in_flight == 0):
                        break
                # Other workers still have tasks - wait a bit and retry
                await asyncio.sleep(0.1)
                continue

            # Wait for at least one task to complete (with timeout to recheck queue)
            done, active_tasks = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=1.0
            )

            for task in done:
                try:
                    result = task.result()
                    await write_result(result, task.example_id, gpu_idx)
                except Exception as e:
                    async with state_lock:
                        completed += 1
                        in_flight -= 1
                        logger.error(
                            f"[Progress: {completed}/{total}] [GPU {gpu_idx}] Example {task.example_id} failed: {e}"
                        )
                        if completed >= total:
                            all_done.set()

    logger.info(
        f"Processing {total} examples with {num_gpus} GPU(s), {batch_size} concurrent per GPU ({total_concurrent} total)"
    )

    workers = []
    for i in range(num_gpus):
        workers.append(asyncio.create_task(gpu_worker(i)))
        if i < num_gpus - 1:
            await asyncio.sleep(gpu_stagger_seconds)

    await asyncio.gather(*workers)
