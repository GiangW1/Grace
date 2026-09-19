"""One actor GPU and independent, single-GPU vLLM rollout processes."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import time


def validate_layout(cfg):
    n = int((cfg.get("hardware") or {}).get("n_gpu", 1))
    workers = int((cfg.get("rollout") or {}).get("workers", 0))
    if int((cfg.get("vllm") or {}).get("tensor_parallel", 1)) != 1:
        raise ValueError("the single-actor rollout layout requires tensor_parallel=1")
    if (n == 1 and workers == 0):
        return workers
    if n < 2 or workers != n - 1:
        raise ValueError("n_gpu>1 requires rollout.workers=n_gpu-1; one GPU is reserved for the actor")
    return workers


def _worker_main(connection, device, model, config, rank):
    # Spawned process: bind before importing the CUDA runtime or vLLM.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    llm = None
    try:
        from grace_gc.backends.verl_trainer import build_vllm_engine, _shutdown_vllm_engine
        llm = build_vllm_engine(model, config, rank)
        connection.send({"ok": True, "value": build_vllm_engine.last})
        while True:
            method, args, kwargs = connection.recv()
            if method == "__close__":
                break
            execution = None
            try:
                if method == 'generate':
                    from grace_gc.backends.vllm_two_phase import _execution_metadata, generate_with_params
                    execution = _execution_metadata(None, len(args[0]))
                    value = generate_with_params(llm, args[0], args[1], kwargs, execution)
                else:
                    # Match the single-engine adapter/cache API compatibility.
                    engine = getattr(llm, 'llm_engine', None) or getattr(llm, 'engine', None)
                    operation = getattr(llm, method, None) or getattr(engine, method, None)
                    if operation is None:
                        raise TypeError(f'vLLM has no {method}')
                    value = operation(*args, **kwargs)
                    from grace_gc.backends.weight_sync import _require_finished
                    _require_finished(value, method)
                connection.send({"ok": True, "value": value, "execution": execution})
            except Exception as exc:
                connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}", "execution": execution})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if llm is not None:
            _shutdown_vllm_engine(llm)
        connection.close()


class ProcessWorker:
    def __init__(self, device, model, config, rank):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker_main, args=(child, device, model, config, rank))
        self.process.start()
        child.close()

    def receive(self):
        while not self.connection.poll(.1):
            if not self.process.is_alive():
                raise RuntimeError(f"rollout worker exited with code {self.process.exitcode}")
        try:
            result = self.connection.recv()
        except EOFError as exc:
            raise RuntimeError("rollout worker disconnected") from exc
        self.last_execution = result.get('execution')
        if not result["ok"]:
            raise RuntimeError(result["error"])
        return result["value"]

    def call(self, method, args, kwargs):
        self.connection.send((method, args, kwargs))
        return self.receive()

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(("__close__", (), {}))
            except (BrokenPipeError, OSError):
                pass
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        self.connection.close()


class RolloutPool:
    def __init__(self, workers, devices):
        if not workers or len(workers) != len(devices):
            raise ValueError("rollout workers and devices must be nonempty and aligned")
        self.workers, self.devices = workers, devices
        self.executor = ThreadPoolExecutor(max_workers=len(workers))
        self.failed = False
        self.last_execution = {}

    @classmethod
    def launch(cls, model, config, rank, n_gpu):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            import torch
            devices = [str(i) for i in range(torch.cuda.device_count())]
        else:
            devices = [value.strip() for value in visible.split(',') if value.strip() and value.strip() != '-1']
        if len(devices) < n_gpu or len(set(devices)) != len(devices):
            raise ValueError("CUDA_VISIBLE_DEVICES does not provide the requested distinct GPUs")
        workers = []
        try:
            for device in devices[1:n_gpu]:
                workers.append(ProcessWorker(device, model, {**config, 'tensor_parallel': 1}, rank))
            metadata = [worker.receive() for worker in workers]
            pool = cls(workers, devices[1:n_gpu])
            pool.metadata = {"actor_device": devices[0], "rollout_devices": pool.devices,
                             "workers": metadata, "n_gpu_reserved": n_gpu}
            return pool
        except Exception:
            for worker in workers:
                worker.close()
            raise

    def _call(self, method, jobs, request_count=0):
        if self.failed:
            raise RuntimeError("rollout pool failed; refusing a mixed or incomplete snapshot")
        started = time.perf_counter()
        detail = {"method": method, "request_count": request_count, "workers": [], "status": "running",
                  "scope": "worker RPC times overlap; already included in parent phase wall and reserved GPU time"}
        self.last_execution = detail

        def invoke(index, indices, args, kwargs):
            row = {"worker": index, "device": self.devices[index], "request_indices": indices, "status": "running"}
            tick = time.perf_counter()
            try:
                result = self.workers[index].call(method, args, kwargs)
                if method != 'generate' and result is False:
                    raise RuntimeError(f"rollout worker {index} {method} returned false")
                row['status'] = 'complete'
                return result, row
            except Exception as exc:
                row.update(status='failed', error=str(exc))
                return exc, row
            finally:
                row['wall_seconds'] = time.perf_counter() - tick
                row['generate_execution'] = getattr(self.workers[index], 'last_execution', None)

        futures = [self.executor.submit(invoke, *job) for job in jobs]
        values, failures = [], []
        for future in futures:
            result, row = future.result()
            detail['workers'].append(row)
            values.append(result)
            if isinstance(result, Exception): failures.append(result)
        detail['wall_seconds'] = time.perf_counter() - started
        detail['status'] = 'failed' if failures else 'complete'
        if failures:
            self.failed = True
            raise RuntimeError(f"rollout {method} failed: {failures[0]}") from failures[0]
        return values

    def generate(self, prompts, sampling_params, **kwargs):
        params = sampling_params if isinstance(sampling_params, list) else [sampling_params]*len(prompts)
        if len(params) != len(prompts):
            raise ValueError("sampling_params and prompts must have equal length")
        jobs = []
        for i in range(len(self.workers)):
            indices = list(range(i, len(prompts), len(self.workers)))
            if indices:
                jobs.append((i, indices, ([prompts[j] for j in indices], [params[j] for j in indices]), kwargs))
        values = self._call('generate', jobs, len(prompts))
        outputs = [None]*len(prompts)
        for job, result in zip(jobs, values):
            if len(result) != len(job[1]):
                self.failed = True
                self.last_execution.update(status='failed', error='worker output count mismatch')
                raise ValueError("rollout worker output count does not match its shard")
            for i, output in zip(job[1], result): outputs[i] = output
        return outputs

    def _broadcast(self, method, *args):
        self._call(method, [(i, [], args, {}) for i in range(len(self.workers))])
        return True

    def add_lora(self, request): return self._broadcast('add_lora', request)
    def remove_lora(self, ident): return self._broadcast('remove_lora', ident)
    def reset_prefix_cache(self): return self._broadcast('reset_prefix_cache')

    def close(self):
        self.executor.shutdown(wait=True)
        for worker in self.workers:
            close = getattr(worker, 'close', None)
            if close is not None: close()
