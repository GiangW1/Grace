# Expected-gain audit results (2026-09-25)

The archive contains the completed run's JSON summaries, run configurations,
data split, replay provenance, logs, and the optimizer probe's 192 output rows.
The source runs are `expected-gain-fixed-36ae00c` and
`expected-gain-data-fixed-36ae00c`.

The predictor audit covered 256 problems and produced 382 live prefixes from
229 problems. The replay used 6,112 saved continuations. The global suite split
the live prefixes into 241 train, 74 validation, and 67 diagnostic rows (160,
48, and 48 problems). Its validation-selected gradient predictor was the zero
model. On diagnostic problems, the selected residual-risk head had MSE
1,888,759,523 versus 1,782,549,613 for the constant-risk baseline. The two
independent reference-gradient estimates had cosine 0.159. These results do
not establish a predictor benefit or a training-reward gain.

Key files:

- [Complete text-results archive](expected-gain-20260925-results.tar.gz)
- [Global suite](summary/global_suite.json)
- [Predictor replay](summary/predictor_replay.json)
- [Reference replay](summary/reference_replay.json)
- [Reference-check replay](summary/reference_check_replay.json)
- [Optimizer probe](summary/update_probe.json)
- [Problem split](summary/split.json)

Full-gradient `.npy` arrays, basis arrays, and adapter weights are omitted from
the archive. The large raw audit/replay JSONL files are included in the archive
but are not tracked as separate files. Full binary run directories remain on
the experiment server; this snapshot supports result review but not
recomputation of the full-space suite from scratch.
