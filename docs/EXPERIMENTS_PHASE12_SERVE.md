# Phase12 serving-path scheduler probe

This is an independent experiment next to PR4's colocated offline probe. It
does not construct a vLLM engine or load an actor in the client process. Start
one frozen actor snapshot with `vllm serve`, then run
`scripts/scheduler_probe_serve.py` against its OpenAI-compatible
`/v1/completions` endpoint.

The client uses the same frozen workload and per-start request plan as the
PR4 scheduler probe. It sends tokenized prompts (`prompt` is a list of token
IDs) and requests `return_token_ids=true`, so a response can be checked against
the prompt and recorded without detokenizing in the client.

The scheduler arms are deliberately different:

- `all`: submit every logical start to the vLLM server. This is the serving
  baseline in which vLLM owns the waiting queue.
- `no_refill`: submit at most `capacity` logical starts. A selected start sends
  its suffix immediately, but a stopped/finished start leaves the client slot
  empty until the current batch settles.
- `refill`: submit the next pending logical start as soon as a stopped/finished
  start settles. A selected suffix still stays with the same logical start.

This experiment measures an application admission policy. It does not claim
that `capacity` is vLLM's internal `max_num_seqs`; keep the server flags fixed
across arms and enforce the comparison capacity in the client. The `all` arm
is a separate production baseline, not the no-refill counterfactual.

## Start one A100 server

Use the same base model/tokenizer and frozen actor adapter that the offline
probe uses. The adapter must be a PEFT `save_pretrained` directory (for
example, one produced by `grace_gc.backends.weight_sync.save_lora_adapter`),
not the NumPy training checkpoint. Register it as a static LoRA model so the
probe cannot accidentally measure the base model:

```bash
SERVED_MODEL=grace-actor
LORA_ADAPTER=/path/to/frozen-actor-adapter
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --enable-lora \
  --lora-modules "$SERVED_MODEL=$LORA_ADAPTER" \
  --gpu-memory-utilization 0.80 \
  --generation-config vllm \
  --enable-prefix-caching \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --host 127.0.0.1 --port 8000
```

The server must be ready before the probe starts. The script checks `/health`
and `/v1/models`, and fails if the requested `server_model` is not served.
The model id is the value returned by `/v1/models`, not necessarily the local
filesystem path.

## Run the probe

```bash
python scripts/scheduler_probe_serve.py \
  --config configs/hardware/a100_1.yaml \
  --config configs/experiments/scheduler_probe_serve.yaml \
  --model-path "$MODEL_PATH" \
  --server-model "$SERVED_MODEL" \
  --data-path "$DATA_PATH" \
  --n-problems 8 --starts-per-problem 16 \
  --capacity 8 \
  --decision-tokens 512 --max-new-tokens 2048 \
  --temperature 0.2 \
  --p 0 --p 0.5 --p 0.75 --p 1 \
  --repeats 3 \
  --mode both --workload both \
  --run-dir runs/phase12-scheduler-serve
```

For a service with an API key, set the environment variable named by
`api_key_env` (default `VLLM_API_KEY`). The key is never written into the run
metadata.

The run directory contains:

- `scheduler_serve_trials.jsonl`: one row per arm/trial;
- `scheduler_serve_records.jsonl`: one row per logical start;
- `scheduler_serve_requests.jsonl`: one row per HTTP generation request;
- `scheduler_serve_comparisons.jsonl`: paired arm comparisons;
- `scheduler_serve_summary.json`: the complete summary and server metadata.

The request ledger includes client wall time, prompt/output token counts,
finish reasons, and relative submit/return times. If Prometheus is enabled,
collect vLLM's server-side request queue, TTFT, prefill/decode, KV-cache and
prefix-cache metrics separately; they are not inferred from client timings.

## Interpretation

The primary causal contrast is

```text
time(no_refill, same server, same client capacity)
  - time(refill, same server, same client capacity)
```

The `all` arm answers a different deployment question: whether client-side
capacity limiting is useful compared with handing the whole pending pool to
vLLM. A positive refill result against `no_refill` does not by itself prove a
gain over `all`.

Because different arrival orders can change sampling and batching, the script
records exact trajectory matches and generated-token differences but does not
require exact outputs for the wall-clock comparison. For a strict paired
mechanism test, use greedy decoding or a frozen token-length trace in a
follow-up run.
