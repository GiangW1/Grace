"""Large fixed-basis hashes keep their byte protocol without a bytes-sized copy."""

import hashlib

import numpy as np
import pytest

from grace_gc import versions


@pytest.mark.parametrize('array', [np.arange(12.).reshape(3, 4),
                                 np.arange(12.).reshape(3, 4).T,
                                 np.arange(12, dtype='>i4')[::2],
                                 np.zeros((0, 3)), np.asarray(7.)])
def test_array_hash_preserves_bytes_and_passes_contiguous_buffer(array, monkeypatch):
    expected = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    original = hashlib.sha256

    def observe(buffer):
        assert isinstance(buffer, np.ndarray)
        assert buffer.flags.c_contiguous
        return original(buffer)

    monkeypatch.setattr(versions.hashlib, 'sha256', observe)
    assert versions.sha256_array(array) == expected
