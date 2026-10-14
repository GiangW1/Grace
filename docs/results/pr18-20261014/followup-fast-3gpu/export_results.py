"""Export the completed PR18 run without weights, checkpoints or full trajectories."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/wja/grace-pr18-fast')
RUN = Path('/mnt/mydata/wja/grace/runs/pr18-20261014/followup-fast-3gpu')
PREFIX = Path('docs/results/pr18-20261014/followup-fast-3gpu')
PAYLOAD = HERE / 'payload'
DEST = PAYLOAD / PREFIX
HEAD = '332b816c8385d4cbd0105aaa0ea2e7c9cec078af'


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def write(rel, value):
    p = DEST / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


copied = []


def copy(rel):
    src, dst = RUN / rel, DEST / rel
    assert src.is_file() and src.stat().st_size < 1024 * 1024, rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    copied.append({'path': str(rel), 'sha256': hashlib.sha256(src.read_bytes()).hexdigest()})


def load(rel):
    return json.loads((RUN / rel).read_text())


suite = load('suite.json')
assert suite['status'] == 'complete' and len(suite['jobs']) == 9
assert all(j['status'] == 'complete' and j['returncode'] == 0 for j in suite['jobs'])
remote = {x['path']: x for x in json.loads((HERE / 'head-tree.json').read_text())['tree'] if x['type'] == 'blob'}
comparison = json.loads((HERE / 'source-comparison.json').read_text())
assert comparison['head'] == HEAD and not comparison['local_changes']
assert all(p.startswith('docs/results/') for p in comparison['missing_local_paths'])

for rel in ['suite.json', 'comparison_summary.json', *[p.name for p in RUN.glob('*.yaml')]]:
    copy(Path(rel))

common = ['config.yaml', 'run_meta.json', 'sampling.json', 'tokenizer.json', 'vllm_engine.json', 'run.log', 'compute_ledger.json']
arms = ['core8_full_pg-seed17', 'core8_uniform_ht-seed17', 'core8_grace-seed17']
source_checks = {}
for arm in arms:
    summary = load(arm + '/summary.json')
    assert summary['run_status'] == 'complete' and summary['step'] == 50
    steps = [json.loads(line) for line in (RUN / arm / 'steps.jsonl').read_text().splitlines() if line.strip()]
    assert len(steps) == 50 and [s['step'] for s in steps] == list(range(1, 51))
    for name in [*common, 'effective_config.yaml', 'health.json', 'cost_control.json', 'checkpoints.json', 'steps.jsonl', 'logprob_probe.jsonl', 'data_conflicts.json', 'data_exclusions.json']:
        copy(Path(arm) / name)
    compact = {k: v for k, v in summary.items() if k not in ['versions', 'summary']}
    compact['summary'] = {k: v for k, v in summary['summary'].items() if k != 'versions'}
    compact['export_note'] = 'Original numerical summary; embedded source texts and environment dump omitted. See versions_compact.json and source_identity.json.'
    write(arm + '/summary_compact.json', compact)
    versions = summary['versions']
    safe_keys = ['python', 'platform', 'packages', 'package_version_sources', 'file_hashes', 'missing', 'cuda_visible_devices', 'cuda', 'gpu_name', 'gpu_count', 'gpu_memory_bytes', 'model_path', 'data_path', 'model', 'data', 'run_knobs', 'started']
    write(arm + '/versions_compact.json', {k: versions[k] for k in safe_keys if k in versions})
    hashes = {}
    for rel, content in versions['source_files'].items():
        data = content.encode()
        assert (SOURCE / rel).read_bytes() == data, (arm, rel, 'run source changed')
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert remote[rel]['sha'] == blob, (arm, rel, 'PR source differs')
        hashes[rel] = {'sha256': hashlib.sha256(data).hexdigest(), 'git_blob_sha': blob}
    source_checks[arm] = {'recorded_source_files': len(hashes), 'all_match_run_local_and_pr18': True, 'files': hashes}

rows, eval_manifests = [], []
evals = [('Initial zero adapter', 'initial-eval'), ('Full-PG', arms[0] + '-eval'), ('Uniform-HT', arms[1] + '-eval'), ('GRACE', arms[2] + '-eval')]
for label, folder in evals:
    e = load(folder + '/eval_summary.json')
    assert e['n_problems'] == 128 and e['n_samples'] == 512 and e['requested_k'] == 4
    manifest = load(folder + '/evaluation_manifest.json')
    eval_manifests.append(manifest['ordered_records_sha256'])
    raw = [json.loads(line) for line in (RUN / folder / 'eval_per_problem.jsonl').read_text().splitlines() if line.strip()]
    assert len(raw) == 128 and all(len(r['rewards']) == 4 for r in raw)
    assert sum(sum(r['rewards']) for r in raw) / 512 == e['avg']
    assert sum(any(r['rewards']) for r in raw) / 128 == e['pass_at_k']
    for name in [*common, 'actor_source.json', 'data_splits.json', 'eval_summary.json', 'evaluation_manifest.json']:
        copy(Path(folder) / name)
    dst = DEST / folder / 'eval_per_problem_compact.jsonl'
    keys = ['problem_id', 'gold', 'truncated', 'rewards', 'extracted', 'response_tokens', 'finish_reasons', 'sample_seeds', 'prompt_truncated', 'vllm_finish_reasons']
    dst.write_text(''.join(json.dumps(clean({k: r[k] for k in keys if k in r}), ensure_ascii=False, allow_nan=False) + '\n' for r in raw))
    rows.append({'arm': label, 'n_problems': 128, 'n_samples': 512, 'correct_samples': round(e['avg'] * 512), 'avg': e['avg'], 'pass_at_4': e['pass_at_k'], 'parse_rate': e['parse_rate'], 'truncate_rate': e['truncate_rate']})
assert len(set(eval_manifests)) == 1

audit = 'core8_grace-seed17-audit'
for name in [*common, 'actor_source.json', 'data_conflicts.json', 'audit_manifest.json', 'audit_summary.json', 'direct_prediction_summary.json']:
    copy(Path(audit) / name)
prediction = load(audit + '/direct_prediction_summary.json')
assert prediction['all_prefixes']['n_prefixes'] == 8 and prediction['all_prefixes']['n_problems'] == 4
assert all(p['n_continuations'] == 8 for p in prediction['prefixes'])
assert all(p['prediction_norm_sq'] == 0 for p in prediction['prefixes'])
write('source_identity.json', {
    'repository': 'https://github.com/GiangW1/Grace', 'pr': 18, 'executed_source_matches_pr_head': HEAD,
    'local_checkout_metadata_note': 'The execution checkout retained older Git metadata; exact executed source is verified by recorded source text and Git blob hashes against PR18, not by its local git HEAD.',
    'source_comparison': {'local_source_differences': [], 'absent_historical_result_paths_preserved_on_remote': len(comparison['missing_local_paths'])},
    'training_source_checks': source_checks,
})

start = datetime.fromisoformat(suite['jobs'][0]['started_at'])
end = datetime.fromisoformat(suite['jobs'][-1]['finished_at'])
write('result-summary.json', {
    'status': 'complete', 'source_commit': HEAD, 'started_at': start.isoformat(), 'finished_at': end.isoformat(),
    'total_wall_seconds': (end - start).total_seconds(), 'completed_jobs': 9, 'training_steps_each': 50,
    'training_seeds': [17], 'physical_gpus': [7, 2, 3], 'evaluations': rows,
    'frozen_prediction': prediction['all_prefixes'],
    'limitations': ['Single training seed and 128-problem diagnostic; not evidence of a quality improvement.', 'Frozen effective prediction has zero norm for all 8 prefixes; MSE ratio versus zero is 1.0 and no predictive benefit is demonstrated.', 'Frozen audit has 4 independent problems; the 64 continuations are not 64 independent problems.', 'Training command runtimes are sequential observations and do not establish a replicated speedup.'],
})

ops = DEST / 'operations'
ops.mkdir(exist_ok=True)
shutil.copyfile('/mnt/mydata/wja/grace/operations/pr18-20261014/start-fast-single-seed.sh', ops / 'start-fast-single-seed.sh')
log = HERE / 'pytest.log'
if log.exists():
    shutil.copyfile(log, DEST / 'validation-pytest.log')
    assert '68 passed' in log.read_text(), log.read_text()[-1000:]

table = '\n'.join(f"| {r['arm']} | {r['correct_samples']}/512 | {r['avg']:.4%} | {r['pass_at_4']:.4%} |" for r in rows)
(DEST / 'README.md').write_text(f'''# PR18 三卡快速诊断实测结果（2026-10-14）

9/9 项任务正常完成，退出码均为 0。北京时间 2026-10-14 15:24:57 开始，22:46:59 结束，合计 7 小时 22 分钟。

执行代码与 PR18 提交 `{HEAD}` 的源文件逐项一致；见 `source_identity.json`。训练时本地 Git 元数据较旧，不能直接用其 HEAD 作为实测版本。此次上传保留远端历史结果，不删除本地缺失的旧归档。

## 配置与范围

- Qwen3-4B，关闭 thinking，输出上限 8192 token；rank-8 small-core，梯度维度 4608。
- 物理卡 7 用作 HF actor，2、3 是独立 TP=1 vLLM rollout worker；共三张 RTX5090。vLLM memory utilization 0.75，max sequences 2，chunked prefill，max batched tokens 2048。
- core8 Full-PG / Uniform-HT / GRACE 各 50 步，训练 seed=17，固定每批 8 题、32 条轨迹，固定 baseline=0.5，predictor warmup=10 步。
- 同一个固定 128 题 MATH-500 子集（题目选择 seed=20261013，评测 seed=17），每题 4 条答案；共享一次初始零适配器评测，各方法仅做最终评测，共 2048 条答案。
- 仅冻结 GRACE 最终检查点做审计：4 题 × 2 前缀 × 8 条续写，决策点 512；另有一次 CPU 预测报告。
- 启动命令见 `operations/start-fast-single-seed.sh`；有效配置、依赖版本、数据哈希和检查点哈希随结果保存。实际训练/评测/审计耗时见 `suite.json`。

## 结果

| 检查点 | 正确答案条数 | 平均正确率 | pass@4 |
|---|---:|---:|---:|
{table}

平均正确率以 512 条答案为分母；pass@4 是 128 题中至少答对一次的比例。四个评测的题目与顺序哈希一致。`eval_summary.json` 保留原始汇总及按题统计，`eval_per_problem_compact.jsonl` 保留分数、提取答案、seed、长度及截断信息，省去整段推理和 token ID。

这轮结果没有证明 GRACE 的质量收益。其正确率低于初始零适配器及 Full-PG；仅一个训练 seed，不能据此作多 seed 或显著性结论。

冻结审计的 8 个前缀上，有效预测向量范数全部为 0。预测 MSE 与零预测 MSE 同为 762.3473580259277，比值为 1.0；本轮没有显示预测收益。末步健康记录显示 `m_shrink=0.0`。这里记录观测，尚未完成原因定位，不能把“流程完成”解读为“方法有效”。审计只有 4 道独立题，64 条续写不能当成 64 道独立题。

训练命令耗时：Full-PG 112.23 分钟、Uniform-HT 82.80 分钟、GRACE 85.85 分钟。三组顺序执行、只有一个 seed，这些是实际运行记录，不能作为重复实验证实的加速结论。审计中的 token 成本代理不等于 GPU 墙钟速度。

## 文件与校验

- `result-summary.json`、`comparison_summary.json`、`suite.json`：总览、对比与 9 个作业的完整时间。
- 三个训练目录：50 步指标、健康记录、计算账本、配置、精简版本信息和检查点索引。
- 四个评测目录：原始评测汇总与题目清单、精简逐题答案分数。
- `core8_grace-seed17-audit/`：原始审计汇总、题目清单及冻结预测报告。原始 `audit_summary.json` 沿用执行程序的 NaN 表示；派生汇总将非有限数写为 null。
- `validation-pytest.log`：本次上传前运行已有相关 CPU 测试，68 passed；未重新执行 GPU 训练。
- `export-manifest.json`、`SHA256SUMS`：文件清单和校验和；`export_results.py` 是此次筛选导出脚本。

权重、NPZ/NPY、LoRA 检查点、完整训练轨迹、完整审计 bundles、环境/源代码的重复大段转储均未上传。单个结果文件小于 1 MiB。原始文件仍在服务器 `/mnt/mydata/wja/grace/runs/pr18-20261014/followup-fast-3gpu`；早期已中断的 `followup-full` 不属于这次完成结果。
''')
shutil.copyfile(Path(__file__), DEST / 'export_results.py')

files = [p for p in DEST.rglob('*') if p.is_file() and p.name not in ['export-manifest.json', 'SHA256SUMS']]
assert all(p.stat().st_size < 1024 * 1024 for p in files)
assert all(p.suffix not in ['.npz', '.npy', '.safetensors', '.pt', '.bin'] for p in files)
write('export-manifest.json', {
    'created_at': datetime.now(timezone.utc).isoformat(), 'source_run_directory': str(RUN),
    'source_commit': HEAD, 'original_copied_files': copied,
    'files': [{'path': str(p.relative_to(DEST)), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)],
    'derived_files_note': 'Compact summary removes duplicated embedded source/environment; compact per-problem rows remove full answers/token IDs. Original copied files preserve exact bytes.',
})
files = sorted(p for p in DEST.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
(DEST / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(DEST)) + '\n' for p in files))
print(json.dumps({'export_directory': str(DEST), 'files': len(files) + 1, 'bytes': sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()), 'all_jobs_complete': True, 'source_snapshots_verified': source_checks.keys().__len__()}, indent=2))
