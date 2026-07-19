"""Tests for the augmentation adapters + the generated per-transform op families.

``AlbumentationsOp`` / ``TorchvisionTransformOp`` are classic sample-scoped ops
(``__call__(sample)`` — the form every engine, composing op, AND visual-canvas node
classifier handles) applying ONE library draw jointly to input and target per the
``target`` mode. Transforms are configured Confluid-natively (nested ``!class:`` nodes or
the generated ``Alb*`` / ``Tv*`` per-transform ops from
:mod:`sampleflux.ops.albumentations_transforms` /
:mod:`sampleflux.ops.torchvision_transforms`).
"""

import subprocess
import sys
from pathlib import Path

import albumentations as A
import confluid  # type: ignore[import-not-found]
import numpy as np
import pytest
import torch
from PIL import Image

import sampleflux.ops.albumentations_transforms as albt
from sampleflux.core import Flux
from sampleflux.kinds import op_contract
from sampleflux.ops.albumentations import AlbumentationsOp
from sampleflux.ops.target import MasksToDetectionBoxesOp
from sampleflux.ops.torchvision import TorchvisionTransformOp
from sampleflux.sample import Sample


def _image() -> np.ndarray:
    return np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)


def _mask() -> np.ndarray:
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 0:2] = 1
    return mask


def _sample(**meta: object) -> Sample:
    return Sample(_image(), _mask(), dict(meta))


def _detection_sample() -> Sample:
    # MasksToDetectionBoxesOp derives the tight xyxy box from the mask — the exact
    # detection-dict contract both adapters consume in target="boxes" mode.
    return MasksToDetectionBoxesOp()(Sample(_image(), _mask(), {}))


class TestAlbumentationsOp:
    def test_zero_arg_construction_and_lazy_validation(self) -> None:
        op = AlbumentationsOp()
        with pytest.raises(ValueError, match="transform"):
            op(_sample())

    def test_transform_and_transforms_mutually_exclusive(self) -> None:
        op = AlbumentationsOp(transform=A.HorizontalFlip(p=1.0), transforms=[A.HorizontalFlip(p=1.0)])
        with pytest.raises(ValueError, match="not both"):
            op(_sample())

    def test_contract_is_sample_scoped(self) -> None:
        # Sample scope is what the engine fast-path, every composing op, AND the visual
        # canvas classifier handle — guard the signature.
        contract = op_contract(AlbumentationsOp())
        assert contract.accepts == "sample"

    def test_single_transform_input_only(self) -> None:
        out = list(Flux(source=[_sample(idx=7)], ops=[AlbumentationsOp(A.HorizontalFlip(p=1.0))]))[0]
        assert np.array_equal(out.input, _image()[:, ::-1])
        assert np.array_equal(out.target, _mask())  # target NOT flipped in "none" mode
        assert out.meta == {"idx": 7}

    def test_transforms_list_mask_mode(self) -> None:
        op = AlbumentationsOp(transforms=[A.HorizontalFlip(p=1.0)], target="mask", seed=0)
        out = list(Flux(source=[_sample(idx=7)], ops=[op]))[0]
        assert np.array_equal(out.input, _image()[:, ::-1])
        assert np.array_equal(out.target, _mask()[:, ::-1])
        assert out.meta == {"idx": 7}  # metadata untouched

    def test_prebuilt_compose_accepted_seed_rejected(self) -> None:
        compose = A.Compose([A.HorizontalFlip(p=1.0)])
        out = AlbumentationsOp(compose, target="mask")(_sample())
        assert np.array_equal(out.target, _mask()[:, ::-1])
        with pytest.raises(ValueError, match="seed"):
            AlbumentationsOp(compose, target="mask", seed=3)(_sample())

    def test_pipeline_rebuilds_when_transform_changes(self) -> None:
        op = AlbumentationsOp(A.HorizontalFlip(p=1.0))
        first = op.pipeline
        op.transform = A.HorizontalFlip(p=0.0)
        assert op.pipeline is not first  # the lazy cache keys on the configured objects

    def test_pil_input_accepted(self) -> None:
        out = AlbumentationsOp(A.HorizontalFlip(p=1.0))(Sample(Image.fromarray(_image()), None, {}))
        assert isinstance(out.input, np.ndarray)
        assert np.array_equal(out.input, _image()[:, ::-1])

    def test_boxes_mode_auto_bbox_params(self) -> None:
        # When the op builds the Compose itself, bbox_params are added automatically.
        op = AlbumentationsOp(transforms=[A.HorizontalFlip(p=1.0)], target="boxes")
        sample = _detection_sample()
        out = list(Flux(source=[sample], ops=[op]))[0]
        width = _image().shape[1]
        x0, _, x1, _ = sample.target["boxes"][0].tolist()
        assert isinstance(out.target["boxes"], torch.Tensor)
        assert out.target["boxes"].dtype == torch.float32
        assert out.target["labels"].dtype == torch.int64
        assert np.allclose(out.target["boxes"][0].tolist(), [width - x1, 1.0, width - x0, 3.0], atol=1e-4)

    def test_boxes_mode_prebuilt_compose_requires_bbox_params(self) -> None:
        op = AlbumentationsOp(A.Compose([A.HorizontalFlip(p=1.0)]), target="boxes")
        with pytest.raises(ValueError, match="bbox_params"):
            op(_detection_sample())

    def test_boxes_mode_requires_detection_dict(self) -> None:
        op = AlbumentationsOp(transforms=[A.HorizontalFlip(p=1.0)], target="boxes")
        with pytest.raises(TypeError, match="detection"):
            op(Sample(_image(), "not-a-dict", {}))

    def test_confluid_native_yaml_roundtrip(self) -> None:
        # The YAML surface is nested !class: nodes — no library-specific dict formats.
        yaml_text = (
            "!class:sampleflux.ops.albumentations.AlbumentationsOp\n"
            "target: mask\n"
            "transforms:\n"
            "  - !class:albumentations.HorizontalFlip\n"
            "    p: 1.0\n"
        )
        op = confluid.load(yaml_text)
        out = op(_sample())
        assert np.array_equal(out.target, _mask()[:, ::-1])
        # Pipeline Parity: dump → reload → identical output.
        reloaded = confluid.load(confluid.dump(op))
        out2 = reloaded(_sample())
        assert np.array_equal(out.input, out2.input)
        assert np.array_equal(out.target, out2.target)


class TestGeneratedAlbumentationsOps:
    def test_family_generated(self) -> None:
        assert len(albt.__all__) > 50
        for name in ("AlbHorizontalFlip", "AlbAffine", "AlbRandomBrightnessContrast"):
            assert name in albt.__all__

    def test_flip_parity_with_raw_library(self) -> None:
        out = list(Flux(source=[_sample(idx=1)], ops=[albt.AlbHorizontalFlip(p=1.0, target="mask")]))[0]
        assert np.array_equal(out.input, _image()[:, ::-1])
        assert np.array_equal(out.target, _mask()[:, ::-1])
        assert out.meta == {"idx": 1}

    def test_marks_and_canvas_classification(self) -> None:
        cls = albt.AlbHorizontalFlip
        assert cls.__confluid_category__ == "op"
        assert cls.__confluid_group__ == "augment/albumentations"
        assert cls.__confluid_random__ is True
        # The canvas op classifier reads vars(cls) — the base __call__ must be re-stated.
        assert "__call__" in vars(cls)
        assert cls.LIBRARY_CLS is A.HorizontalFlip

    def test_signature_mirrors_transform_plus_adapter_knobs(self) -> None:
        import inspect

        params = list(inspect.signature(albt.AlbHorizontalFlip).parameters)
        assert "p" in params
        assert params[-2:] == ["target", "seed"]
        docs = confluid.parse_param_docs(albt.AlbHorizontalFlip)
        assert docs.get("target")  # adapter knobs documented in the spliced Args block

    def test_required_param_lazy_error(self) -> None:
        crop = albt.AlbRandomCrop()  # zero-arg construction always works
        with pytest.raises(Exception, match="height"):
            crop(_sample())  # the library's own missing-argument error, raised lazily

    def test_post_construction_reconfigure_rebuilds(self) -> None:
        op = albt.AlbHorizontalFlip(p=0.0)
        assert np.array_equal(op(_sample()).input, _image())  # p=0 → identity
        op.p = 1.0  # confluid post-construction paradigm
        assert np.array_equal(op(_sample()).input, _image()[:, ::-1])

    def test_generated_op_unwraps_in_transforms_list(self) -> None:
        # A generated op wired into an adapter's transforms slot (the canvas pattern)
        # unwraps to its inner library transform via raw_transform.
        op = AlbumentationsOp(transforms=[albt.AlbHorizontalFlip(p=1.0)], target="mask")
        out = op(_sample())
        assert np.array_equal(out.target, _mask()[:, ::-1])

    def test_short_name_yaml_roundtrip(self) -> None:
        yaml_text = "!class:AlbHorizontalFlip\np: 1.0\ntarget: mask\n"
        op = confluid.load(yaml_text)
        out = op(_sample())
        assert np.array_equal(out.target, _mask()[:, ::-1])
        dumped = confluid.dump(op)
        assert "AlbHorizontalFlip" in dumped and "p: 1.0" in dumped
        out2 = confluid.load(dumped)(_sample())
        assert np.array_equal(out.input, out2.input)

    def test_composes_inside_random_apply_and_chain(self) -> None:
        # Composing ops route inner ops through core._apply_op, so the generated ops nest
        # inside the gate/chain — the canonical "gate an augmentation" pattern.
        from sampleflux.ops.random_apply import RandomApply
        from sampleflux.ops.transform_chain import TransformChain

        chain = TransformChain(
            ops=[RandomApply(op=albt.AlbHorizontalFlip(p=1.0, target="mask"), probability=1.0, random_state=0)]
        )
        out = list(Flux(source=[_sample(idx=3)], ops=[chain]))[0]
        assert np.array_equal(out.input, _image()[:, ::-1])
        assert np.array_equal(out.target, _mask()[:, ::-1])
        assert out.meta == {"idx": 3}

    def test_composes_inside_enable(self) -> None:
        from sampleflux.ops.enable import Enable

        enable = Enable(ops=[albt.AlbHorizontalFlip(p=1.0, target="mask")])
        setattr(enable, "augment", True)  # the toggle flag arrives post-construction (Confluid paradigm)
        out = list(Flux(source=[_sample()], ops=[enable]))[0]
        assert np.array_equal(out.input, _image()[:, ::-1])
        assert np.array_equal(out.target, _mask()[:, ::-1])


class TestTorchvisionTransformOp:
    v2 = pytest.importorskip("torchvision.transforms.v2")

    def test_zero_arg_construction_and_lazy_validation(self) -> None:
        op = TorchvisionTransformOp()
        with pytest.raises(ValueError, match="transform"):
            op(_sample())

    def test_contract_is_sample_scoped(self) -> None:
        assert op_contract(TorchvisionTransformOp()).accepts == "sample"

    def test_input_only_flip_emits_chw_tensor(self) -> None:
        op = TorchvisionTransformOp(self.v2.RandomHorizontalFlip(p=1.0))
        out = list(Flux(source=[_sample(idx=7)], ops=[op]))[0]
        assert type(out.input) is torch.Tensor  # tv_tensors subclass stripped
        assert out.input.shape == (3, 4, 6)
        assert np.array_equal(out.input.permute(1, 2, 0).numpy(), _image()[:, ::-1])
        assert np.array_equal(out.target, _mask())  # untouched in "none" mode
        assert out.meta == {"idx": 7}

    def test_transforms_list_mask_mode_matches_albumentations(self) -> None:
        # Cross-library parity: the same deterministic flip through either adapter
        # yields the same pixels (layouts differ — CHW tensor vs HWC array).
        tv = TorchvisionTransformOp(transforms=[self.v2.RandomHorizontalFlip(p=1.0)], target="mask")(_sample())
        alb = AlbumentationsOp(A.HorizontalFlip(p=1.0), target="mask")(_sample())
        assert np.array_equal(tv.input.permute(1, 2, 0).numpy(), alb.input)
        assert np.array_equal(tv.target.numpy(), alb.target)

    def test_two_dim_input_gains_channel_axis(self) -> None:
        out = TorchvisionTransformOp(self.v2.RandomHorizontalFlip(p=1.0))(Sample(_mask(), None, {}))
        assert out.input.shape == (1, 4, 6)

    def test_pil_input_stays_pil(self) -> None:
        out = TorchvisionTransformOp(self.v2.RandomHorizontalFlip(p=1.0))(Sample(Image.fromarray(_image()), None, {}))
        assert isinstance(out.input, Image.Image)
        assert np.array_equal(np.asarray(out.input), _image()[:, ::-1])

    def test_boxes_mode_mirrors_coordinates(self) -> None:
        sample = _detection_sample()
        op = TorchvisionTransformOp(self.v2.RandomHorizontalFlip(p=1.0), target="boxes")
        out = list(Flux(source=[sample], ops=[op]))[0]
        width = _image().shape[1]
        x0, _, x1, _ = sample.target["boxes"][0].tolist()
        assert type(out.target["boxes"]) is torch.Tensor  # BoundingBoxes subclass stripped
        assert out.target["boxes"].dtype == torch.float32
        assert out.target["labels"].dtype == torch.int64
        assert out.target["boxes"][0].tolist() == [width - x1, 1.0, width - x0, 3.0]

    def test_boxes_mode_requires_detection_dict(self) -> None:
        op = TorchvisionTransformOp(self.v2.RandomHorizontalFlip(p=1.0), target="boxes")
        with pytest.raises(TypeError, match="detection"):
            op(Sample(_image(), "not-a-dict", {}))

    def test_module_imports_without_torchvision(self) -> None:
        # The ADAPTER module is entry-pointed for discovery, so importing it must NOT
        # pull in torchvision (all library imports are lazy, inside __call__). The
        # torchvision_transforms module deliberately DOES import it (generation), with a
        # guarded fallback to zero ops when it is absent.
        code = "import sys; import sampleflux.ops.torchvision; assert 'torchvision' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True, cwd=str(Path(__file__).resolve().parents[1]))


class TestGeneratedTorchvisionOps:
    v2 = pytest.importorskip("torchvision.transforms.v2")

    def test_family_generated(self) -> None:
        import sampleflux.ops.torchvision_transforms as tvt

        assert len(tvt.__all__) > 30
        assert "TvRandomHorizontalFlip" in tvt.__all__
        assert "TvCompose" not in tvt.__all__  # containers excluded — chaining is native

    def test_flip_parity_and_marks(self) -> None:
        import sampleflux.ops.torchvision_transforms as tvt

        cls = tvt.TvRandomHorizontalFlip
        assert cls.__confluid_category__ == "op"
        assert cls.__confluid_group__ == "augment/torchvision"
        assert cls.__confluid_random__ is True
        assert "__call__" in vars(cls)
        out = list(Flux(source=[_sample()], ops=[cls(p=1.0, target="mask")]))[0]
        assert np.array_equal(out.input.permute(1, 2, 0).numpy(), _image()[:, ::-1])
        assert np.array_equal(out.target.numpy(), _mask()[:, ::-1])

    def test_short_name_yaml_loads(self) -> None:
        import sampleflux.ops.torchvision_transforms  # noqa: F401  (registers the Tv* names)

        op = confluid.load("!class:TvRandomHorizontalFlip\np: 1.0\ntarget: mask\n")
        out = op(_sample())
        assert np.array_equal(out.target.numpy(), _mask()[:, ::-1])
