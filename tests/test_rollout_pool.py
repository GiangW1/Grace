from types import SimpleNamespace
import threading

import pytest
from grace_gc.backends.rollout_pool import _worker_main as real_worker_main
from tests.test_rollout_observability import simulated_gpu_train


class CpuLLM:
    def __init__(self):
        self.llm_engine = SimpleNamespace(add_lora=lambda request: request == 'snapshot')

    def generate(self, prompts, sampling_params, **kwargs):
        if isinstance(sampling_params, list):
            raise TypeError('sampling_params must be SamplingParams, not list')
        return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'], seed=sampling_params.seed)
                for p in prompts]

def cpu_worker_main(connection, device, model, config, rank):
    from grace_gc.backends import verl_trainer
    def build(*args): return CpuLLM()
    build.last = {'synthetic_cpu_worker':True}
    verl_trainer.build_vllm_engine = build
    verl_trainer._shutdown_vllm_engine = lambda llm: None
    real_worker_main(connection, device, model, config, rank)


def test_spawn_rpc_seed_preserving_fallback_and_cleanup(monkeypatch):
    from grace_gc.backends import rollout_pool as module
    monkeypatch.setattr(module, '_worker_main', cpu_worker_main)
    worker = module.ProcessWorker('3', 'synthetic', {}, 2)
    assert worker.receive() == {'synthetic_cpu_worker':True}
    pool = module.RolloutPool([worker], ['3'])
    try:
        pool.add_lora('snapshot')
        result = pool.generate([{'prompt_token_ids':[i]} for i in range(3)],
                               [SimpleNamespace(seed=i+9) for i in range(3)])
        assert [x.seed for x in result] == [9,10,11]
        detail = pool.last_execution['workers'][0]['generate_execution']
        assert detail['serial_fallback'] and detail['num_generate_calls'] == 4
        assert detail['failed_generate_batches'] == 1
        with pytest.raises(RuntimeError, match='false'):
            pool.add_lora('bad snapshot')
    finally:
        pool.close()
    assert not worker.process.is_alive()


def test_pool_shards_concurrently_and_restores_request_order():
    from grace_gc.backends.rollout_pool import RolloutPool
    barrier = threading.Barrier(3)
    seen = []

    class Worker:
        def call(self, method, args, kwargs):
            prompts, params = args
            seen.append([x.seed for x in params])
            barrier.wait(timeout=3)
            return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'], seed=s.seed)
                    for p, s in zip(prompts, params)]
    pool = RolloutPool([Worker(), Worker(), Worker()], devices=['1', '2', '3'])
    try:
        result = pool.generate([{'prompt_token_ids': [i]} for i in range(8)],
                               [SimpleNamespace(seed=i+20) for i in range(8)])
        assert [x.seed for x in result] == list(range(20, 28))
        assert sorted(sum(seen, [])) == list(range(20, 28))
        assert pool.last_execution['request_count'] == 8
        assert sorted(sum([r['request_indices'] for r in pool.last_execution['workers']], [])) == list(range(8))
    finally:
        pool.close()


def test_pool_broadcast_failure_is_visible_and_disables_next_generation():
    from grace_gc.backends.rollout_pool import RolloutPool
    class Worker:
        def __init__(self, fail=False): self.fail = fail
        def call(self, method, args, kwargs):
            if self.fail: raise RuntimeError('worker sync failed')
            return True
    pool = RolloutPool([Worker(), Worker(True)], devices=['1','2'])
    try:
        with pytest.raises(RuntimeError, match='worker sync failed'):
            pool.add_lora('snapshot')
        assert pool.last_execution['status'] == 'failed'
        with pytest.raises(RuntimeError, match='failed'):
            pool.generate([{'prompt_token_ids':[1]}], [SimpleNamespace(seed=1)])
    finally:
        pool.close()


def test_missing_worker_outputs_abort_without_partial_batch():
    from grace_gc.backends.rollout_pool import RolloutPool
    worker = SimpleNamespace(call=lambda *args:[])
    pool = RolloutPool([worker], ['1'])
    try:
        with pytest.raises(ValueError, match='output count'):
            pool.generate([{'prompt_token_ids':[1]}], [SimpleNamespace(seed=1)])
        assert pool.last_execution['status'] == 'failed'
        with pytest.raises(RuntimeError, match='failed'):
            pool.add_lora('next')
    finally:
        pool.close()


@pytest.mark.parametrize('n,workers,valid', [(1,0,True),(4,3,True),(2,1,True),(4,0,False),(4,4,False),(1,1,False)])
def test_rollout_layout_reserves_one_actor_card(n,workers,valid):
    from grace_gc.backends.rollout_pool import validate_layout
    cfg={'hardware':{'n_gpu':n},'rollout':{'workers':workers}}
    if valid: assert validate_layout(cfg) == workers
    else:
        with pytest.raises(ValueError, match='rollout|n_gpu'):
            validate_layout(cfg)


def test_gpu_entry_uses_pool_and_closes_on_failed_initial_sync(simulated_gpu_train, monkeypatch):
    import torch
    from grace_gc.backends.rollout_pool import RolloutPool
    trainer, run, cfg = simulated_gpu_train
    events = []
    pool = SimpleNamespace(metadata={'synthetic':True}, close=lambda:events.append('closed'))
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device:events.append(('actor',device)))
    monkeypatch.setattr(RolloutPool, 'launch', lambda *args:pool)
    def engines(actor, llm, *args):
        assert llm is pool
        def sync(): raise RuntimeError('synthetic sync failure')
        return SimpleNamespace(last_rollout={}), {'sync':sync}
    monkeypatch.setattr(trainer, 'make_gpu_engines', engines)
    with pytest.raises(RuntimeError, match='synthetic sync'):
        trainer.train({**cfg,'hardware':{'n_gpu':4}, 'rollout':{'workers':3}}, run)
    assert events == [('actor',0), 'closed']


@pytest.mark.parametrize('n,workers', [(1,0),(4,3)])
def test_tensor_parallel_cannot_use_unaccounted_devices(n, workers):
    from grace_gc.backends.rollout_pool import validate_layout
    with pytest.raises(ValueError, match='tensor_parallel=1'):
        validate_layout({'hardware':{'n_gpu':n}, 'rollout':{'workers':workers},
                         'vllm':{'tensor_parallel':2}})
