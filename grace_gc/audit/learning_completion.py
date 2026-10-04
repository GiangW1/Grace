"""Experiment A: does the expected update finish before the answer does?

At a decision position t every live prefix h of problem x carries three
vectors written by the trajectory replay: the prefix score g_h and the two
independent half means A_h, B_h of the full label G=(R-b)(g_h+g_s).  With
binary rewards and on-policy suffixes

    mu(h) = E[G | h] = P(h) + c(h),        P(h) = (q(h) - b) g_h,

so P(h) is the part of the expected update that is fixed once h is written,
up to the scalar q(h), and c(h) is what the suffix still contributes.

Per-prefix energies are dominated by ||P(h)||, and those terms cancel across
problems because g_h has zero mean under the policy.  Learning is driven by
the population update grad J = E_x S_x, where S_x sums mu(h) over the live
prefixes of problem x.  Every population inner product is therefore a
U-statistic over distinct problems.  Same-problem and same-prefix terms only
enter the finite-batch curve F_batch(m); same-prefix terms always pair the
two independent halves.

The answer side uses the same prefixes: rho_A = E Var(R|h) / E Var(R|x) over
live prefixes, with Var(R|h) from the binary reward counts and Var(R|x) from
distinct prefixes of the same problem.  Only problems with at least two rows
in a group enter that group, so both curves are measured on one population.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from grace_gc.audit.dynamic_score import load_diagonal_metric
from grace_gc.versions import sha256_array, sha256_file

VECTOR_FILES = ("prefix_score_gradients.npy", "half_mean_full_grads_a.npy",
                "half_mean_full_grads_b.npy")
GRAM_CACHE_VERSION = 2
GROUP_VIEWS = {
    "all": "pooled",
    "text_pre_answer": "pooled",
    "functional_pre_answer": "pooled",
    "undecided_half_a": "half_b",
}
DEFINITIONS = {
    "P(h)": "(q(h)-b) g_h, the prefix-determined expected update",
    "mu(h)": "E[G|h] for G=(R-b) grad log pi(full response|prompt)",
    "S_x": "sum of the vector over the group's live prefixes of problem x at position t",
    "U_ab": "mean over ordered pairs of distinct problems of <S^a_x, S^b_y>; unbiased for <E S^a, E S^b>",
    "D_ab": "mean over problems of <S^a_x, S^b_x>; same-prefix terms pair independent halves",
    "F_pop": "U_PM / U_MM: share of the population update grad J carried by the prefix-determined part",
    "residual_pop": "||E S^mu - E S^P||^2 / ||E S^mu||^2: squared relative error of the prefix part",
    "cos_pop": "U_PM / sqrt(U_PP U_MM)",
    "F_batch(m)": "(D_PM + (m-1) U_PM) / (D_MM + (m-1) U_MM): the same share for a batch of m problems",
    "F_problem": "F_batch(1)",
    "F_prefix": "sum_h <P(h),mu(h)> / sum_h ||mu(h)||^2, the per-prefix energy share used in PR11-PR13",
    "aggregation_survival": "U_MM / D_MM: fraction of per-problem update energy surviving aggregation",
    "rho_A": "sum_h Var(R|h) / sum_x k_x Var(R|x), live prefixes only",
    "gap": "rho_A - residual_pop; positive means the update is more settled than the answer",
    "views": "pooled uses both halves; half_b selects on half A and estimates on half B only",
}


def row_key(row):
    return (str(row["problem_id"]), str(row.get("path_id")), int(row["t"]))


def load_replay_rows(replay_dirs):
    """Read prefixes.jsonl from one or more replay shards.

    The dense arrays are opened only when a Gram matrix has to be computed,
    so cached Grams remain usable after the large files are deleted.
    """
    rows, sources, seen = [], [], set()
    for source, path in enumerate(replay_dirs):
        root = Path(path)
        prefixes = root / "prefixes.jsonl"
        local = [json.loads(line) for line in prefixes.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if not local:
            raise ValueError(f"{root} has no replayed prefixes")
        digest = sha256_file(prefixes)
        provenance = root / "score_gradient_provenance.json"
        if provenance.is_file():
            recorded = json.loads(provenance.read_text(encoding="utf-8")).get("replay_prefixes_sha256")
            if recorded is not None and recorded != digest:
                raise ValueError(f"{root}: prefixes.jsonl differs from the stored score gradients")
        for index, row in enumerate(local):
            key = row_key(row)
            if key in seen:
                raise ValueError(f"prefix {key} appears in more than one replay row")
            seen.add(key)
            rows.append({**row, "_source": source, "_index": index})
        provenance_hashes = {name: sha256_file(root / name)
                             for name in ("score_gradient_provenance.json", "replay_provenance.json")
                             if (root / name).is_file()}
        sources.append({"root": root, "prefixes_sha256": digest, "n_rows": len(local), "arrays": None,
                        "provenance_sha256": provenance_hashes})
    return rows, sources


def prefix_scalars(row):
    """Half counts, binary reward sums and the fixed baseline of one prefix."""
    counts = row.get("half_counts")
    if counts is None:
        raise ValueError("replay rows need half_counts; replay with --store-half-means "
                         "--store-trajectory-decomposition")
    n_a, n_b = (int(value) for value in counts)
    if n_a < 1 or n_b < 1:
        raise ValueError("every prefix needs at least one continuation in each half")
    sums = row.get("half_reward_sum")
    if sums is None:
        means = row.get("half_mean_reward") or [row.get("half_mean_reward_a"),
                                                row.get("half_mean_reward_b")]
        sums = [float(means[0]) * n_a, float(means[1]) * n_b]
    rounded = []
    for value, n in zip(sums, (n_a, n_b)):
        value = float(value)
        if not np.isfinite(value) or abs(value - round(value)) > 1e-6 or not 0 <= round(value) <= n:
            raise ValueError("learning completion needs binary rewards in every half")
        rounded.append(float(round(value)))
    baseline = row.get("mean_baseline", row.get("baseline"))
    if baseline is None:
        raise ValueError("replay row lacks its baseline")
    baseline = float(baseline)
    if not np.isfinite(baseline) or not 0.0 <= baseline <= 1.0:
        raise ValueError("baseline must be finite and in [0, 1]")
    halves = row.get("half_mean_baseline")
    if halves is not None and max(abs(float(value) - baseline) for value in halves) > 1e-9:
        raise ValueError("baseline must be constant within a prefix")
    return {"n_a": n_a, "n_b": n_b, "s_a": rounded[0], "s_b": rounded[1], "baseline": baseline}


def position_members(rows):
    members = {}
    for index, row in enumerate(rows):
        members.setdefault(int(row["t"]), []).append(index)
    return dict(sorted(members.items()))


def stacked_gram(vectors, dimension, metric=None, block_columns=65536):
    """Exact Gram matrix of row vectors, streamed over coordinate blocks."""
    k = len(vectors)
    gram = np.zeros((k, k), dtype=np.float64)
    scale = None if metric is None else np.sqrt(np.asarray(metric, dtype=np.float64))
    for start in range(0, int(dimension), int(block_columns)):
        stop = min(start + int(block_columns), int(dimension))
        block = np.empty((k, stop - start), dtype=np.float64)
        for slot, vector in enumerate(vectors):
            block[slot] = vector[start:stop]
        if scale is not None:
            block *= scale[start:stop]
        if not np.all(np.isfinite(block)):
            raise ValueError("stored gradients contain non-finite values")
        gram += block @ block.T
    return 0.5 * (gram + gram.T)


def _open_vectors(source):
    if source["arrays"] is None:
        root = source["root"]
        missing = [name for name in VECTOR_FILES if not (root / name).is_file()]
        if missing:
            raise ValueError(f"{root} lacks {', '.join(missing)}; replay with --store-half-means "
                             "--store-trajectory-decomposition or pass a matching --gram-cache")
        arrays = [np.load(root / name, mmap_mode="r") for name in VECTOR_FILES]
        if arrays[0].ndim == 3 and arrays[0].shape[2] == 1:
            arrays[0] = arrays[0][:, :, 0]
        shape = arrays[1].shape
        if len(shape) != 2 or shape[0] != source["n_rows"] or any(a.shape != shape for a in arrays):
            raise ValueError(f"{root}: trajectory arrays do not match prefixes.jsonl")
        source["arrays"] = arrays
    return source["arrays"]


def _gram_fingerprint(rows, sources, members, metric_sha256, vector_hashes):
    contributing = [{"root": str(sources[index]["root"].resolve()),
                     "prefixes_sha256": sources[index]["prefixes_sha256"],
                     "provenance_sha256": sources[index]["provenance_sha256"]}
                    for index in sorted({rows[i]["_source"] for i in members})]
    payload = {"cache_version": GRAM_CACHE_VERSION, "keys": [list(row_key(rows[i])) for i in members],
               "sources": contributing, "metric_sha256": metric_sha256, "vectors": vector_hashes}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def position_grams(rows, sources, metric_file=None, cache_dir=None, block_mb=1024.0, log=None):
    """One exact 3n x 3n Gram of [g; A; B] per decision position.

    Positions never interact, so each Gram reads only that position's rows.
    With cache_dir, a Gram is reused when the rows, replay files and metric
    match, including vector content hashes. Verified cached vector hashes can
    be used after dense arrays are deleted position by position.
    """
    metric_sha = "euclidean" if metric_file is None else sha256_file(metric_file)
    grams = {}
    available_hashes = {}
    for source in sources:
        source["arrays"] = None
    for t, members in position_members(rows).items():
        cache = None if cache_dir is None else Path(cache_dir) / f"gram_t{t}.npz"
        cached = None
        if cache is not None and cache.is_file():
            with np.load(cache, allow_pickle=False) as stored:
                required = {"cache_version", "fingerprint", "vector_hashes", "gram", "gram_sha256"}
                if required.issubset(stored.files) and int(stored["cache_version"]) == GRAM_CACHE_VERSION:
                    cached = {"fingerprint": str(stored["fingerprint"]),
                              "vector_hashes": json.loads(str(stored["vector_hashes"])),
                              "gram": np.array(stored["gram"], dtype=np.float64),
                              "gram_sha256": str(stored["gram_sha256"])}
        vector_hashes = {}
        for source_index in sorted({rows[i]["_source"] for i in members}):
            source = sources[source_index]
            root = str(source["root"].resolve())
            vector_hashes[root] = {}
            for name in VECTOR_FILES:
                path = source["root"] / name
                key = (root, name)
                if key not in available_hashes:
                    available_hashes[key] = sha256_file(path) if path.is_file() else None
                value = available_hashes[key]
                if value is None and cached is not None:
                    value = cached["vector_hashes"].get(root, {}).get(name)
                vector_hashes[root][name] = value
        fingerprint = _gram_fingerprint(rows, sources, members, metric_sha, vector_hashes)
        if cached is not None and cached["fingerprint"] == fingerprint:
            gram = cached["gram"]
            shape = (3 * len(members), 3 * len(members))
            if (gram.shape == shape and np.all(np.isfinite(gram))
                    and sha256_array(gram) == cached["gram_sha256"]):
                grams[t] = gram
                if log is not None:
                    log(f"gram t={t}: cached")
                continue
        vectors, dimension = [], None
        for part in range(3):
            for index in members:
                arrays = _open_vectors(sources[rows[index]["_source"]])
                if dimension is None:
                    dimension = arrays[0].shape[1]
                elif arrays[0].shape[1] != dimension:
                    raise ValueError("replay shards use different gradient dimensions")
                vectors.append(arrays[part][rows[index]["_index"]])
        metric = None if metric_file is None else load_diagonal_metric(metric_file, dimension)
        columns = max(1, int(float(block_mb) * 2 ** 20 // (8 * len(vectors))))
        gram = stacked_gram(vectors, dimension, metric, columns)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            partial = cache.with_name(cache.name + ".partial")
            with partial.open("wb") as handle:
                np.savez(handle, gram=gram, fingerprint=np.array(fingerprint),
                         cache_version=np.array(GRAM_CACHE_VERSION),
                         vector_hashes=np.array(json.dumps(vector_hashes, sort_keys=True)),
                         gram_sha256=np.array(sha256_array(gram)))
            os.replace(partial, cache)
        grams[t] = gram
        if log is not None:
            log(f"gram t={t}: computed from {len(members)} prefixes")
    return grams


def answer_variance_terms(scalars, problem_ids, problems, view):
    """Per-problem sum of Var(R|h) and k_x Var(R|x), both unbiased for binary R."""
    n_a = np.array([s["n_a"] for s in scalars], dtype=np.float64)
    n_b = np.array([s["n_b"] for s in scalars], dtype=np.float64)
    s_a = np.array([s["s_a"] for s in scalars], dtype=np.float64)
    s_b = np.array([s["s_b"] for s in scalars], dtype=np.float64)
    if view == "pooled":
        n, s = n_a + n_b, s_a + s_b
    elif view == "half_b":
        n, s = n_b, s_b
    else:
        raise ValueError(f"unknown view {view}")
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where(n >= 2, s * (n - s) / (n * (n - 1)), np.nan)
    q = s / n
    membership = _membership(problem_ids, problems)
    k = membership.sum(1)
    q_sum = membership @ q
    cross = q_sum ** 2 - membership @ (q * q)
    within = membership @ variance
    total = q_sum - cross / (k - 1)
    return within, total


def _membership(problem_ids, problems):
    ids = np.asarray(problem_ids, dtype=object)
    return np.stack([(ids == problem).astype(np.float64) for problem in problems])


def _aggregate(pair, self_terms, membership):
    off = np.array(pair, dtype=np.float64)
    np.fill_diagonal(off, 0.0)
    problem = membership @ off @ membership.T
    same_problem = np.diag(problem).copy()
    between = problem - np.diag(same_problem)
    self_sum = membership @ self_terms
    return {"between": between, "within": same_problem + self_sum, "self": self_sum}


def position_statistics(gram, scalars, problem_ids, view):
    """Problem-level sufficient statistics for one group at one position.

    gram is the Gram of [g; A; B] restricted to the group's rows.  Rows of
    problems with a single row in the group are dropped first.
    """
    ids = [str(value) for value in problem_ids]
    counts = {problem: ids.count(problem) for problem in set(ids)}
    keep = [i for i, problem in enumerate(ids) if counts[problem] >= 2]
    problems = sorted({ids[i] for i in keep})
    dropped_rows = len(ids) - len(keep)
    report = {"n_rows": len(keep), "n_problems": len(problems), "view": view,
              "excluded_single_row_problems": sum(1 for value in counts.values() if value < 2),
              "excluded_rows": dropped_rows}
    if len(problems) < 2:
        report["problems"] = problems
        return report
    n_all = len(ids)
    index = np.concatenate([np.asarray(keep) + offset for offset in (0, n_all, 2 * n_all)])
    sub = np.asarray(gram, dtype=np.float64)[np.ix_(index, index)]
    n = len(keep)
    gg, ga, gb = sub[:n, :n], sub[:n, n:2 * n], sub[:n, 2 * n:]
    aa, ab, bb = sub[n:2 * n, n:2 * n], sub[n:2 * n, 2 * n:], sub[2 * n:, 2 * n:]
    kept = [scalars[i] for i in keep]
    kept_ids = [ids[i] for i in keep]
    n_a = np.array([s["n_a"] for s in kept], dtype=np.float64)
    n_b = np.array([s["n_b"] for s in kept], dtype=np.float64)
    s_a = np.array([s["s_a"] for s in kept], dtype=np.float64)
    s_b = np.array([s["s_b"] for s in kept], dtype=np.float64)
    baseline = np.array([s["baseline"] for s in kept], dtype=np.float64)
    alpha_a, alpha_b = s_a / n_a - baseline, s_b / n_b - baseline
    if view == "pooled":
        w_a, w_b = n_a / (n_a + n_b), n_b / (n_a + n_b)
        alpha = (s_a + s_b) / (n_a + n_b) - baseline
        g_mu = w_a[None, :] * ga + w_b[None, :] * gb
        mm = (np.outer(w_a, w_a) * aa + np.outer(w_a, w_b) * ab +
              np.outer(w_b, w_a) * ab.T + np.outer(w_b, w_b) * bb)
        pm = alpha[:, None] * g_mu
        pp = np.outer(alpha, alpha) * gg
        self_pm = 0.5 * (alpha_a * np.diag(gb) + alpha_b * np.diag(ga))
        self_mm = np.diag(ab).copy()
        self_pp = alpha_a * alpha_b * np.diag(gg)
    elif view == "half_b":
        pm = alpha_b[:, None] * gb
        mm = bb
        pp = np.outer(alpha_b, alpha_b) * gg
        self_pm = self_mm = self_pp = np.full(n, np.nan)
    else:
        raise ValueError(f"unknown view {view}")
    membership = _membership(kept_ids, problems)
    within, total = answer_variance_terms(kept, kept_ids, problems, view)
    report.update(problems=problems,
                  PM=_aggregate(pm, self_pm, membership),
                  MM=_aggregate(mm, self_mm, membership),
                  PP=_aggregate(pp, self_pp, membership),
                  answer_within=within, answer_total=total)
    return report


def estimate(stats, counts, batch_problems=(1, 8, 32, 128)):
    """Ratios for one or many problem-count vectors (rows of counts)."""
    weights = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    total = weights.sum(1)
    pairs = total ** 2 - (weights * weights).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = {name: np.einsum("rx,xy,ry->r", weights, stats[name]["between"], weights) / pairs
             for name in ("PM", "MM", "PP")}
        d = {name: (weights @ stats[name]["within"]) / total for name in ("PM", "MM", "PP")}
        s = {name: (weights @ stats[name]["self"]) / total for name in ("PM", "MM")}
        positive_mm = u["MM"] > 0
        out = {
            "F_pop": np.where(positive_mm, u["PM"] / u["MM"], np.nan),
            "residual_pop": np.where(positive_mm, (u["MM"] - 2 * u["PM"] + u["PP"]) / u["MM"], np.nan),
            "cos_pop": np.where(positive_mm & (u["PP"] > 0),
                                u["PM"] / np.sqrt(np.abs(u["MM"] * u["PP"])), np.nan),
            "F_problem": np.where(d["MM"] > 0, d["PM"] / d["MM"], np.nan),
            "F_prefix": np.where(s["MM"] > 0, s["PM"] / s["MM"], np.nan),
            "aggregation_survival": np.where(d["MM"] > 0, u["MM"] / d["MM"], np.nan),
            "U_PM": u["PM"], "U_MM": u["MM"], "U_PP": u["PP"],
            "D_PM": d["PM"], "D_MM": d["MM"], "D_PP": d["PP"],
        }
        for m in batch_problems:
            denominator = d["MM"] + (m - 1) * u["MM"]
            out[f"F_batch_{int(m)}"] = np.where(denominator > 0,
                                                (d["PM"] + (m - 1) * u["PM"]) / denominator, np.nan)
        answer_total = weights @ stats["answer_total"]
        answer_within = weights @ stats["answer_within"]
        out["rho_A"] = np.where(answer_total > 0, answer_within / answer_total, np.nan)
        out["gap"] = out["rho_A"] - out["residual_pop"]
    return out


def _clean(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _crossing(values, positions, reached):
    """First crossing, len(grid) if censored, or -1 if missing values prevent identification."""
    hit = reached(values) & np.isfinite(values)
    first = np.where(hit.any(1), hit.argmax(1), len(positions))
    missing_before = (~np.isfinite(values) & (np.arange(len(positions))[None, :] <= first[:, None])).any(1)
    return np.where(missing_before, -1, first)


def _order_report(learning, answer, n_positions):
    unavailable = (learning < 0) | (answer < 0)
    both_censored = (learning == n_positions) & (answer == n_positions)
    defined = ~(unavailable | both_censored)
    count = int(defined.sum())
    return {"P(t_L<t_A)": float(np.mean(learning[defined] < answer[defined])) if count else None,
            "P(t_L==t_A)": float(np.mean(learning[defined] == answer[defined])) if count else None,
            "P(t_L>t_A)": float(np.mean(learning[defined] > answer[defined])) if count else None,
            "n_total": int(learning.size), "n_defined": count,
            "n_unavailable": int(unavailable.sum()), "n_both_censored": int(both_censored.sum())}


def _functional_status(row, probe):
    if probe is not None:
        status = probe.get(row_key(row))
        if status is not None:
            return status
    value = row.get("functional_recoverable")
    return None if value is None else bool(value)


def group_indices(rows, members, probe=None):
    """Row indices (into members) of every group at one position."""
    scalars = [prefix_scalars(rows[i]) for i in members]
    text = [i for i, index in enumerate(members)
            if rows[index].get("answer_emitted") is False and not rows[index].get("finished", False)]
    groups = {"all": list(range(len(members))), "text_pre_answer": text,
              "functional_pre_answer": [i for i in text
                                        if _functional_status(rows[members[i]], probe) is False],
              "undecided_half_a": [i for i, s in enumerate(scalars) if 0 < s["s_a"] < s["n_a"]]}
    return groups, scalars


def load_probe_rescores(path):
    """Map (problem_id, path_id, t) to the independent functional-recovery label."""
    probe = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        value = record.get("functional_recoverable")
        if value is not None:
            probe[row_key(record)] = bool(value)
    return probe


def analyze(rows, grams, probe=None, batch_problems=(1, 8, 32, 128), bootstrap=1000, seed=17,
            epsilon=None, delta=None):
    positions = sorted(grams)
    members = position_members(rows)
    if positions != list(members):
        raise ValueError("Gram positions and replay rows disagree")
    universe = sorted({str(row["problem_id"]) for row in rows})
    column = {problem: i for i, problem in enumerate(universe)}
    rng = np.random.default_rng(seed)
    draws = (np.stack([np.bincount(rng.integers(0, len(universe), len(universe)),
                                   minlength=len(universe)) for _ in range(bootstrap)])
             if bootstrap else np.zeros((0, len(universe)), dtype=np.int64))
    functional_known = any(_functional_status(row, probe) is not None for row in rows)
    layout = {t: group_indices(rows, members[t], probe) for t in positions}
    report = {}
    for group, view in GROUP_VIEWS.items():
        if group == "functional_pre_answer" and not functional_known:
            report[group] = {"available": False,
                             "reason": "no functional_recoverable labels; pass --probe-rescores"}
            continue
        by_t, point_curves, draw_curves = [], [], []
        for t in positions:
            groups, scalars = layout[t]
            local = groups[group]
            n_all = len(members[t])
            index = np.concatenate([np.asarray(local, dtype=np.int64) + offset
                                    for offset in (0, n_all, 2 * n_all)])
            stats = position_statistics(grams[t][np.ix_(index, index)],
                                        [scalars[i] for i in local],
                                        [rows[members[t][i]]["problem_id"] for i in local], view)
            entry = {"t": t, **{key: stats[key] for key in
                                ("n_rows", "n_problems", "excluded_single_row_problems", "excluded_rows")}}
            if stats["n_problems"] < 2:
                entry["estimates"] = None
                by_t.append(entry)
                point_curves.append(None)
                draw_curves.append(None)
                continue
            point = estimate(stats, np.ones(len(stats["problems"])), batch_problems)
            entry["estimates"] = {key: _clean(value[0]) for key, value in point.items()}
            point_curves.append(point)
            if bootstrap:
                selected = draws[:, [column[problem] for problem in stats["problems"]]]
                resampled = estimate(stats, selected, batch_problems)
                entry["ci95"] = {}
                entry["bootstrap_defined"] = {}
                for key, values in resampled.items():
                    finite = values[np.isfinite(values)]
                    entry["bootstrap_defined"][key] = int(finite.size)
                    entry["ci95"][key] = ([float(np.percentile(finite, 2.5)),
                                           float(np.percentile(finite, 97.5))] if finite.size else None)
                draw_curves.append(resampled)
            by_t.append(entry)
        group_report = {"available": True, "view": view, "by_t": by_t}
        if epsilon is not None and delta is not None:
            group_report["crossings"] = _crossings(positions, point_curves, draw_curves,
                                                   epsilon, delta, bootstrap)
        report[group] = group_report
    return {"definitions": DEFINITIONS, "positions": positions, "n_problems": len(universe),
            "n_rows": len(rows), "batch_problems": [int(m) for m in batch_problems],
            "bootstrap": int(bootstrap), "bootstrap_unit": "problem, shared across positions and groups",
            "seed": int(seed), "groups": report}


def _stack(curves, key, n_draws):
    columns = [np.full(n_draws, np.nan) if curve is None else np.asarray(curve[key], dtype=np.float64)
               for curve in curves]
    return np.stack(columns, axis=1)


def _crossings(positions, point_curves, draw_curves, epsilon, delta, bootstrap):
    rules = {"t_L_projection": ("F_pop", lambda v: v >= 1.0 - epsilon),
             "t_L_residual": ("residual_pop", lambda v: v <= epsilon),
             "t_A": ("rho_A", lambda v: v <= delta)}
    out = {"epsilon": float(epsilon), "delta": float(delta),
           "rule": "first decision position meeting the threshold; status distinguishes observed, not_reached and unavailable"}
    for name, (key, reached) in rules.items():
        index = _crossing(_stack(point_curves, key, 1), positions, reached)[0]
        out[name] = positions[index] if 0 <= index < len(positions) else None
        out[name + "_status"] = ("unavailable" if index < 0 else
                                  "not_reached" if index == len(positions) else "observed")
    if bootstrap:
        draws = {name: _crossing(_stack(draw_curves, key, bootstrap), positions, reached)
                 for name, (key, reached) in rules.items()}
        out["bootstrap_order"] = {
            "rule": "conditional on identifiable order; unavailable draws and two censored times are excluded",
            "t_L_projection_vs_t_A": _order_report(draws["t_L_projection"], draws["t_A"], len(positions)),
            "t_L_residual_vs_t_A": _order_report(draws["t_L_residual"], draws["t_A"], len(positions)),
        }
    return out
