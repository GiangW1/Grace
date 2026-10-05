import numpy as np

from scripts.build_adam_metric import adam_diagonal_from_checkpoint


def test_adam_metric_follows_checkpoint_layout_order():
    payload = {
        "layout_names": ["q.lora_A", "q.lora_B"],
        "actor": {"other": np.zeros(1), "q.lora_A": np.zeros(2), "q.lora_B": np.zeros(1)},
        "optimizer": {
            "param_groups": [{"params": [3, 4, 5], "betas": [.9, 0.]}],
            "state": {
                3: {"exp_avg_sq": np.asarray([9.]), "step": 1},
                4: {"exp_avg_sq": np.asarray([1., 4.]), "step": 1},
                5: {"exp_avg_sq": np.asarray([16.]), "step": 1},
            },
        },
    }
    np.testing.assert_allclose(
        adam_diagonal_from_checkpoint(payload, eps=1.),
        [1. / 4., 1. / 9., 1. / 25.],
    )


def test_adam_metric_applies_step_bias_correction():
    payload = {
        "layout_names": ["q.lora_A"],
        "actor": {"q.lora_A": np.zeros(1)},
        "optimizer": {
            "param_groups": [{"params": [3], "betas": [.9, .9]}],
            "state": {3: {"exp_avg_sq": np.asarray([.5]), "step": 2}},
        },
    }
    corrected = .5 / (1. - .9 ** 2)
    np.testing.assert_allclose(
        adam_diagonal_from_checkpoint(payload, eps=1.),
        [1. / (np.sqrt(corrected) + 1.) ** 2],
    )
