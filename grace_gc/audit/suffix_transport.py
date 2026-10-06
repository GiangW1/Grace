"""Small sufficient statistics for paired full-space suffix-transport audits."""

from __future__ import annotations

import numpy as np


class PairedMoments:
    """No sketches: retain full-vector means and only scalar second moments."""

    def __init__(self, dim):
        self.n = 0
        self.sum = np.zeros((2, dim), dtype=np.float64)  # donor G, mean-zero correction C
        self.gram = np.zeros((2, 2), dtype=np.float64)
        self.other_sum = {}
        self.other_second = {}

    def add(self, donor, correction, **others):
        pair = np.stack([donor, correction])
        if pair.shape != self.sum.shape or not np.isfinite(pair).all():
            raise ValueError("paired gradient layout or values are invalid")
        if self.n and set(others) != set(self.other_sum):
            raise ValueError("audit methods changed inside a group")
        self.n += 1
        self.sum += pair
        self.gram += pair @ pair.T
        for name, gradient in others.items():
            g = np.asarray(gradient, dtype=np.float64)
            if g.shape != pair[0].shape or not np.isfinite(g).all():
                raise ValueError("control gradient layout or values are invalid")
            self.other_sum[name] = self.other_sum.get(name, np.zeros_like(g)) + g
            self.other_second[name] = self.other_second.get(name, 0.0) + float(g @ g)

    def report(self, strengths=(0.0, 1.0)):
        if not self.n:
            raise ValueError("no audit draws")
        mean = self.sum / self.n
        covariance = None if self.n < 2 else (self.gram - self.n * (mean @ mean.T)) / (self.n - 1)
        methods = {}
        for strength in strengths:
            direction = np.array([1.0, strength])
            methods[f"st_lambda_{strength:g}"] = {
                "mean_norm_squared": float(np.linalg.norm(direction @ mean) ** 2),
                "second_moment": float(direction @ self.gram @ direction / self.n),
                "conditional_trace_variance": None if covariance is None else float(direction @ covariance @ direction),
            }
        for name, total in self.other_sum.items():
            norm2 = float(np.linalg.norm(total / self.n) ** 2)
            methods[name] = {"mean_norm_squared": norm2, "second_moment": self.other_second[name] / self.n,
                             "conditional_trace_variance": None if self.n < 2 else
                             float((self.other_second[name] - self.n * norm2) / (self.n - 1))}
        # Diagnostic only. Fitting lambda on these same draws and reporting its
        # improvement would reuse outcomes; train takes an explicit frozen value.
        optimum = (None if covariance is None or covariance[1, 1] <= 0 else
                   float(-covariance[0, 1] / covariance[1, 1]))
        return {"n_draws": self.n, "methods": methods, "correction_mean_norm_squared": float(mean[1] @ mean[1]),
                "centered_donor_correction_gram": None if covariance is None else covariance.tolist(),
                "lambda_optimum_in_sample_diagnostic_only": optimum}


def summarize_groups(rows, bootstrap=1000, seed=17):
    """Bootstrap problems, never continuations as independent problem replicates."""
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    live = [r for r in rows if r.get("n_live", 0)]
    names = sorted({name for row in rows for name in row["moments"]["methods"]})
    rng = np.random.default_rng(seed)
    result = {}
    ids = sorted({row["problem_id"] for row in rows})
    draws = [rng.choice(ids, len(ids), replace=True).tolist() for _ in range(bootstrap)] if ids else []
    for name in names:
        by_problem = {}
        for row in rows:
            value = row["moments"]["methods"][name]["conditional_trace_variance"]
            if value is not None:
                # Collector already includes natural finishes and the fixed N.
                by_problem.setdefault(row["problem_id"], []).append(value)
        averages = {pid: float(np.mean(values)) for pid, values in by_problem.items()}
        samples = [float(np.mean([averages[pid] for pid in draw if pid in averages]))
                   for draw in draws if any(pid in averages for pid in draw)]
        result[name] = {"mean_conditional_trace_variance": float(np.mean(list(averages.values()))) if averages else None,
                        "problem_bootstrap_95ci": np.quantile(samples, [.025, .975]).tolist() if samples else None}
    return {"n_groups": len(rows), "n_live_groups": len(live), "n_problems": len({r['problem_id'] for r in rows}),
            "methods": result, "scope": "Conditional on frozen prefixes; includes live/N weighting, excludes between-prefix variance. No efficiency claim from replay time."}


def summarize_population(rows, root):
    """Trace variance over prefixes/suffixes and the uniform selected problem set.

    Correct the squared global mean for finite repeated-prefix noise. Each
    problem needs two independent prefix groups for this correction; with one
    group the second moment remains available and variance is reported as null.
    No individual dense draw or quadratic Gram matrix across problems is saved.
    """
    from pathlib import Path

    by_problem = {}
    for row in rows:
        by_problem.setdefault(row["problem_id"], []).append(row)
    if not rows:
        return {}
    names = sorted(rows[0]["moments"]["methods"])
    output = {}
    for name in names:
        global_sum = None
        noise_total = 0.0
        noise_available = True
        second_total = 0.0
        for problem_rows in by_problem.values():
            problem_sum = None
            sum_mean_norms = 0.0
            for row in problem_rows:
                with np.load(Path(root) / row["moments_file"], allow_pickle=False) as stored:
                    n = int(stored["n"])
                    if name.startswith("st_lambda_"):
                        strength = float(name.removeprefix("st_lambda_"))
                        pair = stored["donor_correction_sum"]
                        mean = (pair[0] + strength * pair[1]) / n
                    else:
                        mean = stored[f"{name}_sum"] / n
                problem_sum = mean.copy() if problem_sum is None else problem_sum + mean
                sum_mean_norms += float(mean @ mean)
            count = len(problem_rows)
            problem_mean = problem_sum / count
            norm = float(problem_mean @ problem_mean)
            if count < 2:
                noise_available = False
            else:
                # Var(mean of independent prefix-group sample means).
                noise_total += (sum_mean_norms - count * norm) / (count - 1) / count
            global_sum = problem_mean.copy() if global_sum is None else global_sum + problem_mean
            second_total += float(np.mean([r["moments"]["methods"][name]["second_moment"] for r in problem_rows]))
        p = len(by_problem)
        second = second_total / p
        norm_plugin = float(np.linalg.norm(global_sum / p) ** 2)
        norm_corrected = norm_plugin - noise_total / p ** 2 if noise_available else None
        variance = None if norm_corrected is None else second - norm_corrected
        output[name] = {"second_moment": second, "mean_norm_squared_plugin": norm_plugin,
                        "mean_norm_squared_noise_corrected": norm_corrected,
                        "population_trace_variance": variance,
                        "note": "Finite-sample corrected estimates can be negative; they are not clipped or used as launch gates."}
    return {"methods": output, "scope": "Uniform selected problems; fresh independent prefix groups and suffix draws, original group denominator. Includes between-prefix variability. Does not turn audit replay wall time into training efficiency."}
