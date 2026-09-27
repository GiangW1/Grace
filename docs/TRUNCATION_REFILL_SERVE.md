# Sparse truncation and refill on one A100

`scripts/truncation_refill_serve.py` is an independent, frozen-actor experiment
using the same `vllm serve` completion interface as the earlier scheduler probe.
It does not update the actor. The primary workload is 64 problems × 8 starts =
512 predetermined starts, with checkpoints at 512 and 1024 generated tokens and
a 2048-token maximum. A stopped start remains in the denominator N; the next
not-yet-started answer comes from the predetermined queue. All arms share one
request-level seed/uniform plan within each repeat, and repeat order alternates.

Use a server with the exact same base model and LoRA actor snapshot throughout
one run. `--server-model` must be a name returned by `/v1/models`. Record the
server launch flags, adapter path, `max_num_seqs`, token budget, and GPU memory
setting with the run. When running learned arms on one A100, leave enough GPU
memory for the separate frozen HF actor; the repository's colocated actor
configuration uses `gpu_memory_utilization: 0.5`. The CLI client checks token
IDs from the response and uses temperature 1 by default, matching GRACE
training. All arms use the same chosen temperature.

Before the main comparison, repeat `--arms full_pg` at client capacities 64,
128, 256, and 512 under the same server setup to locate the throughput plateau.
Use one selected capacity for the paired barrier/stream arms, and report the
fastest measured Full-PG setting as an additional strong baseline. A client
capacity of 8 is only retained in the older diagnostic probe, not this primary
comparison.

First run a separate Full-PG calibration workload from problems excluded from
the timing measurement. This estimates the expected token cost of the second
check. A compact calibration run is sufficient to start; use more problems if
the estimated long tail is noisy.

```bash
python scripts/truncation_refill_serve.py \
  --config configs/experiments/truncation_refill_serve.yaml \
  --model-path "$MODEL_PATH" --server-model grace-actor \
  --data-path "$CALIB_DATA" --n-problems 16 --starts-per-problem 8 \
  --arms full_pg --repeats 1 --capacity 128 \
  --run-dir runs/phase12-trunc-calibration
```

Then run the full scheduler comparison. `--calibration-records` calculates the
second random continuation probability from the observed full lengths. It
matches the **expected generated tokens** of `single_stream` and `two_stream`
under the calibration distribution. The realized token totals and timing
comparison still appear in `summary.json`; the match is not forced on the
measurement problems. If no calibration is supplied, `--p-two-second` is used
as an exploratory setting and `cost_match_calibration` is null.

```bash
python scripts/truncation_refill_serve.py \
  --config configs/experiments/truncation_refill_serve.yaml \
  --model-path "$MODEL_PATH" --server-model grace-actor \
  --data-path "$MEASURE_DATA" --n-problems 64 --starts-per-problem 8 \
  --capacity 128 --first 512 --second 1024 --max-new-tokens 2048 \
  --p-single 0.5 --p-two-first 0.65 \
  --calibration-records runs/phase12-trunc-calibration/records.jsonl \
  --repeats 5 --run-dir runs/phase12-trunc-refill
```

The default arms are `full_pg`, `single_keep`, `two_keep`, `single_barrier`,
`single_stream`, and `two_stream`. The two `*_keep` arms measure extra HTTP and
prefill cost with no truncation. `single_barrier` and `single_stream` have the
same first-stage selection draws and client capacity, isolating the global
prefix barrier. Stream arms decide when each prefix finishes and submit a new
start when that logical answer stops or completes. A selected continuation
retains its logical slot. vLLM still controls the actual number of active GPU
sequences; client capacity is not GPU occupancy.

For learned arms, fit two frozen predictor artifacts from the same shared
actor checkpoint using the existing `train_offline_predictor.py` command:
once with `decision_tokens: 512`, once with
`configs/experiments/predictor_1024.yaml` overlaid. Each artifact uses its
own calibration run directory. The script verifies each artifact's decision
point, max length, actor hash, and synchronized basis before timing. Supply
the same snapshot to the server. Run the random and learned arms together so
the frozen HF actor is resident for every arm in that comparison:

```bash
python scripts/truncation_refill_serve.py \
  --config configs/experiments/truncation_refill_serve.yaml \
  --model-path "$MODEL_PATH" --server-model grace-actor \
  --data-path "$MEASURE_DATA" --calibration-records "$CALIB_RECORDS" \
  --actor-checkpoint "$ACTOR_CHECKPOINT" \
  --predictor-first "$PREDICTOR_512" \
  --predictor-second "$PREDICTOR_1024" \
  --arms full_pg,single_keep,two_keep,single_stream,two_stream,learned_single,learned_two \
  --lambda-first 1 --lambda-second 1 --p-min 0.2 --uniform-shrink 0.5 \
  --compare-feature-batches 1,4 --feature-batch-wait-ms 5 \
  --capacity 128 --repeats 5 --run-dir runs/phase12-trunc-learned
```

The learned local rule is `clip(sqrt(r_hat / (lambda * c_hat)), p_min, 1)`,
mixed with a uniform probability by `uniform_shrink`. `lambda` is fixed during
each trial; sweep it on separate calibration problems to study the cost and
variance tradeoff, then compare on the held-out timing workload. The two
stages require independently fitted predictors, because a 512-token head is
not validated at 1024 tokens. `--compare-feature-batches 1,4` runs each learned
arm twice per repeat with the same starts and random plan. The two trial orders
alternate across repeats. The 5 ms setting is the maximum coalescing wait once
the worker reaches a batch; requests can wait longer behind earlier batches.
One worker serializes HF actor forwards on the colocated A100, and both queue
waiting and model work are included in trial wall time. For a single setting,
omit `--compare-feature-batches` and use `--feature-batch-size` instead.

`records.jsonl` stores every start, its decisions, cumulative inclusion
probability, generated token IDs, and terminal status. `requests.jsonl` stores
the HTTP stages and timings. `trials.jsonl` and `summary.json` store wall time,
tokens, decision latency, queue wait, observed batch size, feature/prediction
work, and the time from 90% settled starts to the end. The paired comparison
also reports whether selections, token lengths, and actual token trajectories
match for every start.
Interpret the wall-time ratio as a batching effect only when the generated
workload is comparable; otherwise report it alongside the token ratio and
decision latency. Summed `feature_seconds` counts each batch once through
per-request apportionment, while `decision_latency_p95_seconds` includes queue
wait and the complete batch forward. `server_metrics.jsonl` samples vLLM
running/waiting requests, KV usage,
and preemptions from `/metrics` once per second by default. An unavailable
endpoint records a metrics error without stopping the trial. The predictor
load time is written separately to `preparation.json`.
No server startup, offline predictor fitting, or actor training time is folded
into the rollout timing. Same-seed schedules are paired by intended starts;
vLLM batching can still change exact sampled trajectories.

For an offline signal check at both stages, generate an audit with
`configs/experiments/predictor_suite_audit_2point.yaml`, then run the existing
suite once for each decision point:

```bash
python scripts/diagnose_predictor_suite.py --bundles "$AUDIT_RUN" \
  --decision-tokens 512 --models zero,simple,ridge,mlp \
  --include-shuffled --device cpu --run-dir runs/phase12-predictor-512
python scripts/diagnose_predictor_suite.py --bundles "$AUDIT_RUN" \
  --decision-tokens 1024 --models zero,simple,ridge,mlp \
  --include-shuffled --device cpu --run-dir runs/phase12-predictor-1024
```

Run this CPU analysis after serve timing on the same host, so its JSONL reads
and fits do not perturb wall time. `two_checkpoint_ht_estimate` in
`grace_gc/core/estimator.py` is the nested control-variate correction for a
later training integration. The serve experiment records the probabilities
needed by that estimator; it does not claim gradient or training improvement
from rollout timing alone.
