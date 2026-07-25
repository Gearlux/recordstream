"""``Pipeline`` — the compose-group unit: identity, None-propagation, close(), Fluid entries,
bare library transforms, import safety."""

import subprocess
import sys
from pathlib import Path

import confluid
import numpy as np
import pytest

from sampleflux import FilterOp, Image, Label, Mask, Pipeline, Record
from sampleflux.ops.structure import RenameField
from tests._fixtures import FixtureFlip


def _rec() -> Record:
    return {"image": Image(np.ones((4, 6, 3), dtype=np.float32)), "class": Label("x")}


class TestImportSafety:
    def test_package_imports_without_torchvision(self) -> None:
        # The top-level package must import without torchvision/albumentations (the engine's
        # op-family dispatch detects them by MRO module NAME — no import), so discovery stays
        # safe on hosts missing the libraries.
        code = (
            "import sys; import sampleflux; "
            "assert 'torchvision' not in sys.modules; "
            "assert 'albumentations' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True, cwd=str(Path(__file__).resolve().parents[1]))


class TestPipelineSemantics:
    def test_zero_arg_is_identity(self) -> None:
        rec = _rec()
        out = Pipeline()(rec)
        assert out is rec  # no ops -> the record passes through untouched

    def test_none_propagation_mid_chain(self) -> None:
        # A filter-drop mid-chain stops the pipeline and propagates None; later ops never run.
        ran = []

        def probe(record: Record) -> Record:
            ran.append(True)
            return record

        p = Pipeline([FilterOp(lambda r: False), probe])
        assert p(_rec()) is None
        assert ran == []  # the op after the drop never fired

    def test_close_propagates_to_inner_ops(self) -> None:
        closed = []

        class _Closeable:
            def __call__(self, record: Record) -> Record:
                return record

            def close(self) -> None:
                closed.append(True)

        p = Pipeline([_Closeable(), FilterOp(lambda r: True)])  # FilterOp has no close -> skipped
        p.close()
        assert closed == [True]

    def test_fluid_entry_flowed_and_cached(self) -> None:
        # A config-deferred entry (confluid Class marker) is flowed on first call and the
        # live op is cached back into the transforms list.
        p = Pipeline(transforms=[confluid.Class(RenameField, src="class", dst="klass")])
        out = p(_rec())
        assert out is not None and "klass" in out and "class" not in out
        assert isinstance(p.transforms[0], RenameField)  # cached in place
        out2 = p(_rec())  # second call uses the cached live op
        assert out2 is not None and "klass" in out2

    def test_native_ops_chain(self) -> None:
        s = {"image": Image(np.arange(4 * 6 * 3).reshape(4, 6, 3).astype(np.float32))}
        out = Pipeline([FixtureFlip(p=1.0), FixtureFlip(p=1.0)])(s)
        assert out is not None
        # Two flips cancel out.
        assert np.array_equal(np.asarray(out["image"]), np.asarray(s["image"]))


class TestBareLibraryEntries:
    def test_bare_albumentations_entry(self) -> None:
        import albumentations as A

        rec = {
            "image": Image(np.arange(6 * 8 * 3).reshape(6, 8, 3).astype(np.float32)),
            "mask": Mask(np.arange(6 * 8).reshape(6, 8).astype(np.uint8)),
            "class": Label("x"),
        }
        out = Pipeline([A.HorizontalFlip(p=1.0)])(rec)
        assert out is not None
        # ONE joint draw moved image and mask together; item types + metadata survive.
        assert isinstance(out["image"], Image) and out["image"].layout == "HWC"
        assert isinstance(out["mask"], Mask)
        assert np.array_equal(np.asarray(out["image"]), np.asarray(rec["image"])[:, ::-1])
        assert np.array_equal(np.asarray(out["mask"]), np.asarray(rec["mask"])[:, ::-1])
        assert out["class"].value == "x"  # not an albumentations key — never reached the library

    def test_bare_torchvision_v2_entries(self) -> None:
        import torch

        v2 = pytest.importorskip("torchvision.transforms.v2")

        # The EXPLICIT v2.ToImage() conversion first (numpy HWC -> CHW tv_tensor), then any v2
        # transform — the engine passes the dict straight through (never converts silently).
        rec = {"image": np.arange(4 * 6 * 3).reshape(4, 6, 3).astype(np.float32)}
        converted = Pipeline([v2.ToImage()])(rec)
        assert converted is not None
        assert isinstance(converted["image"], torch.Tensor) and tuple(converted["image"].shape) == (3, 4, 6)
        cropped = Pipeline([v2.RandomCrop(2)])(converted)
        assert cropped is not None and tuple(cropped["image"].shape) == (3, 2, 2)

    def test_mixed_native_and_bare_library(self) -> None:
        import albumentations as A

        rec = {"image": Image(np.arange(6 * 8 * 3).reshape(6, 8, 3).astype(np.float32))}
        # bare albumentations flip + native fixture flip = identity (both moved the image once).
        out = Pipeline([A.HorizontalFlip(p=1.0), FixtureFlip(p=1.0)])(rec)
        assert out is not None
        assert np.array_equal(np.asarray(out["image"]), np.asarray(rec["image"]))
        assert isinstance(out["image"], Image)
