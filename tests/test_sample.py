import numpy as np
import torch
from dataflux.sample import Sample

def test_sample_from_any() -> None:
    # 1. From dict
    d = {"input": np.array([1, 2]), "target": 1, "metadata": {"id": "test"}}
    s = Sample.from_any(d)
    assert np.array_equal(s.input, d["input"])
    assert s.target == 1
    assert s.metadata["id"] == "test"

    # 2. From tuple (input, target)
    t = (np.array([3, 4]), 0)
    s2 = Sample.from_any(t)
    assert np.array_equal(s2.input, t[0])
    assert s2.target == 0

    # 3. From single item (input only)
    val = np.array([5, 6])
    s3 = Sample.from_any(val)
    assert np.array_equal(s3.input, val)
    assert s3.target is None

def test_sample_to_tuple() -> None:
    s = Sample(input=1, target=2, metadata={"a": 3})
    t = s.to_tuple()
    assert t == (1, 2, {"a": 3})
