from dataclasses import replace

import numpy as np

from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles


def test_audit_can_switch_existing_lag_calculations_to_full_trajectory_gradient():
    suffix = np.asarray([[1., 0.], [1., 0.], [2., 0.], [2., 0.]])
    full = suffix + np.asarray([0., 1.])
    bundle = PrefixBundle(
        "p", 1, np.asarray([0., 1., 0., 1.]), suffix,
        trajectory_grads=full,
        trajectory_grad_norm_sq=np.sum(full * full, axis=1).tolist(),
        answer_emitted=False,
    )
    result = audit_bundles([bundle], np.zeros((2, 0)),
                           {"gradient_target": "trajectory", "require_pre_emit": False},
                           np.random.default_rng(2))
    assert result["gradient_target"] == "trajectory"
    assert result["measurement_version"] == 3

