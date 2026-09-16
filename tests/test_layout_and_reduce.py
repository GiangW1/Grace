import numpy as np
import pytest

from grace_gc.backends.distributed import apply_update_order
from grace_gc.core.layout import ParamLayout, LayoutEntry, layout_hash, pack_grads, unpack_to_dict


def test_pack_roundtrip():
    layout = ParamLayout(
        entries=[
            LayoutEntry("layers.0.q_proj.lora_A", (2, 3), 6, 0),
            LayoutEntry("layers.0.v_proj.lora_B", (3, 2), 6, 6),
        ],
        dim=12,
    )
    grads = {
        "layers.0.q_proj.lora_A": np.arange(6).reshape(2, 3),
        "layers.0.v_proj.lora_B": np.ones((3, 2)),
    }
    vec = pack_grads(grads.items(), layout)
    back = unpack_to_dict(vec, layout)
    np.testing.assert_allclose(back["layers.0.q_proj.lora_A"], grads["layers.0.q_proj.lora_A"])
    assert layout_hash(layout)


def test_reduce_uses_global_n_once():
    u = np.eye(2)
    f = np.zeros((4, 2))
    z = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    real = np.array([1.0, -1.0])
    shard = np.array([1.0, -1.0])
    out = apply_update_order(real, u, f, z, p, global_n=4, shard_grads=[shard], clip=10.0, amp_scale=2.0)
    assert out.global_n == 4
    np.testing.assert_allclose(out.grad, (real / 2.0) + shard)


def test_missing_lora_raises():
    from grace_gc.core.layout import collect_lora_layout

    with pytest.raises(ValueError):
        collect_lora_layout([("embed.weight", np.zeros(2))])


def test_pack_missing_name_raises():
    layout = ParamLayout(
        entries=[
            LayoutEntry("layers.0.q_proj.lora_A", (2, 3), 6, 0),
            LayoutEntry("layers.0.q_proj.lora_B", (3, 2), 6, 6),
        ],
        dim=12,
    )
    with pytest.raises(ValueError, match="missing grad"):
        pack_grads({"layers.0.q_proj.lora_A": np.zeros((2, 3))}.items(), layout)


def test_correction_writeback_follows_layout_names():
    pytest.importorskip("torch")
    import torch

    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.trainer.actor_update import apply_correction_clip_step

    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(2))
    a.grad = torch.tensor([3.0, 0.0])
    b.grad = torch.tensor([0.0, 4.0])
    named_layout = [("q_proj.lora_A", a), ("v_proj.lora_B", b)]
    layout = collect_lora_layout(named_layout)
    shuffled = [("v_proj.lora_B", b), ("q_proj.lora_A", a)]

    class _Opt:
        def step(self):
            return None

    apply_correction_clip_step(
        shuffled,
        layout,
        np.zeros((4, 1)),
        np.zeros((1, 1)),
        np.ones(1),
        np.ones(1),
        1,
        _Opt(),
        clip=0.0,
        use_correction=False,
    )
    np.testing.assert_allclose(a.grad.detach().cpu().numpy(), [3.0, 0.0])
    np.testing.assert_allclose(b.grad.detach().cpu().numpy(), [0.0, 4.0])


def test_reduce_nonzero_f_uses_global_n():
    u = np.eye(2)
    f = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    z = np.array([1.0, 0.0, 0.0, 0.0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    out = apply_update_order(np.zeros(2), u, f, z, p, global_n=4, clip=10.0, amp_scale=1.0)
    np.testing.assert_allclose(out.grad, np.array([0.25, 0.0]))
