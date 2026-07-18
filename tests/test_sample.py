from typing import Any, cast

import numpy as np
import pytest

from sampleflux.sample import Sample


def test_sample_from_any() -> None:
    # 1. From dict
    d = {"input": np.array([1, 2]), "target": 1, "metadata": {"id": "test"}}
    s = Sample.from_any(d)
    assert np.array_equal(s.input, cast(Any, d["input"]))
    assert s.target == 1
    assert s.meta["id"] == "test"

    # 2. From tuple (input, target)
    t = (np.array([3, 4]), 0)
    s2 = Sample.from_any(t)
    assert np.array_equal(s2.input, t[0])
    assert s2.target == 0

    # 3. From tuple (input,) - hits line 22
    t3 = (np.array([7, 8]),)
    s3 = Sample.from_any(t3)
    assert np.array_equal(s3.input, t3[0])
    assert s3.target is None

    # 4. From single item (input only)
    val = np.array([5, 6])
    s4 = Sample.from_any(val)
    assert np.array_equal(s4.input, val)
    assert s4.target is None


def test_sample_to_tuple() -> None:
    s = Sample(input=1, target=2, metadata={"a": 3})
    t = s.to_tuple()
    assert t == (1, 2, {"a": 3})


# --- Batch vs single metadata (Union schema) -------------------------------


def test_single_sample_metadata_is_a_dict() -> None:
    s = Sample(input=1, target=0, metadata={"id": "a"})
    assert s.is_batched is False
    assert s.meta == {"id": "a"}
    assert s.meta["id"] == "a"


def test_batched_sample_metadata_is_a_list_of_dicts() -> None:
    # The collate form: one Sample carrying N stacked items + per-item metadata dicts.
    batch = Sample(input=[1, 2], target=[0, 1], metadata=[{"id": "a"}, {"id": "b"}])
    assert batch.is_batched is True
    assert batch.batch_meta == [{"id": "a"}, {"id": "b"}]
    assert [m["id"] for m in batch.batch_meta] == ["a", "b"]


def test_meta_accessor_raises_on_a_batch() -> None:
    batch = Sample(input=[1], target=[0], metadata=[{"id": "a"}])
    with pytest.raises(TypeError, match="batched"):
        _ = batch.meta


def test_batch_meta_accessor_raises_on_a_single_sample() -> None:
    single = Sample(input=1, target=0, metadata={"id": "a"})
    with pytest.raises(TypeError, match="single"):
        _ = single.batch_meta


def test_describe_falls_back_to_inference_on_a_batch() -> None:
    # A batched sample carries no per-reserved-key stored type -> describe infers (no crash on the list).
    batch = Sample(input=np.zeros((2, 4)), target=np.array([0, 1]), metadata=[{}, {}])
    assert batch.describe() is not None


def test_with_type_rejects_a_batch() -> None:
    from sampleflux.typespec import infer_sample_type

    single = Sample(input=np.zeros((4,)), target=0, metadata={})
    typed = single.with_type(infer_sample_type(single))  # single sample: OK
    assert typed.describe() is not None

    batch = Sample(input=np.zeros((2, 4)), target=np.array([0, 1]), metadata=[{}, {}])
    with pytest.raises(TypeError, match="batched"):
        batch.with_type(infer_sample_type(single))
