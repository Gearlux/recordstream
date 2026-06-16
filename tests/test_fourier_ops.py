"""Tests for the 1-D Fourier-transform ops: ``dataflux.ops.numpy.FourierOp`` and
``dataflux.ops.torch.FourierOp``.

Both compute the 1-D DFT (``numpy.fft.fft`` / ``torch.fft.fft``) of ``sample.input`` and
ALWAYS yield a complex result — for real and complex inputs alike. The tests pin: the
real/complex/integer dtype-promotion rules, the round-trip against the inverse transform,
the ``n`` / ``axis``-``dim`` / ``norm`` parameters, the framework type guards, the
``ACCEPTS``/``PRODUCES`` contract conformance, and the closed-``Literal`` ``norm`` validation.
"""

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from dataflux.ops import FftShiftOp as FlatFftShiftOp
from dataflux.ops import FourierOp as FlatFourierOp
from dataflux.ops import IfftShiftOp as FlatIfftShiftOp
from dataflux.ops import InverseFourierOp as FlatInverseFourierOp
from dataflux.ops.numpy import FftShiftOp as NpFftShiftOp
from dataflux.ops.numpy import FourierNorm
from dataflux.ops.numpy import FourierOp as NpFourierOp
from dataflux.ops.numpy import IfftShiftOp as NpIfftShiftOp
from dataflux.ops.numpy import InverseFourierOp as NpInverseFourierOp
from dataflux.ops.numpy import SpectrumScalingOp as NpSpectrumScalingOp
from dataflux.ops.numpy import WindowOp as NpWindowOp
from dataflux.ops.torch import FftShiftOp as TorchFftShiftOp
from dataflux.ops.torch import FourierOp as TorchFourierOp
from dataflux.ops.torch import IfftShiftOp as TorchIfftShiftOp
from dataflux.ops.torch import InverseFourierOp as TorchInverseFourierOp
from dataflux.ops.torch import SpectrumScalingOp as TorchSpectrumScalingOp
from dataflux.ops.torch import WindowOp as TorchWindowOp
from dataflux.sample import Sample
from dataflux.typespec import infer_sample_type
from dataflux.windows import WINDOW_SUM_KEY


def test_flat_imports_are_torch_variants() -> None:
    """``from dataflux.ops import …`` resolves the FFT ops to their torch variants — the package's
    documented convention that flat data-op imports default to torch (mirrors RescaleOp etc.)."""
    assert FlatFourierOp is TorchFourierOp
    assert FlatInverseFourierOp is TorchInverseFourierOp
    assert FlatFftShiftOp is TorchFftShiftOp
    assert FlatIfftShiftOp is TorchIfftShiftOp


# ---------------------------------------------------------------------------
# numpy FourierOp
# ---------------------------------------------------------------------------


class TestNumpyFourierOp:
    def test_real_float32_matches_numpy_and_is_complex64(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = NpFourierOp()(Sample(input=x))
        assert isinstance(out.input, np.ndarray)
        assert out.input.dtype == np.complex64
        assert np.allclose(out.input, np.fft.fft(x))

    def test_real_float64_promotes_to_complex128(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        out = NpFourierOp()(Sample(input=x))
        assert out.input.dtype == np.complex128
        assert np.allclose(out.input, np.fft.fft(x))

    def test_integer_input_promotes_to_complex128(self) -> None:
        x = np.arange(8, dtype=np.int64)
        out = NpFourierOp()(Sample(input=x))
        assert out.input.dtype == np.complex128
        assert np.allclose(out.input, np.fft.fft(x))

    def test_complex64_input_stays_complex64(self) -> None:
        x = np.array([1 + 2j, 3 - 1j, 0 + 0j, -2 + 1j], dtype=np.complex64)
        out = NpFourierOp()(Sample(input=x))
        assert out.input.dtype == np.complex64
        assert np.allclose(out.input, np.fft.fft(x))

    def test_complex128_input_stays_complex128(self) -> None:
        x = np.array([1 + 2j, 3 - 1j, 0 + 0j, -2 + 1j], dtype=np.complex128)
        out = NpFourierOp()(Sample(input=x))
        assert out.input.dtype == np.complex128

    def test_constant_signal_has_only_dc_component(self) -> None:
        # FFT of a length-4 constant [1,1,1,1] is [4, 0, 0, 0] (all energy in the DC bin).
        out = NpFourierOp()(Sample(input=np.ones(4, dtype=np.float64)))
        assert np.allclose(out.input, np.array([4, 0, 0, 0]))

    def test_roundtrip_via_ifft_recovers_input(self) -> None:
        x = np.array([1.0, -2.0, 3.5, 0.0, 7.0], dtype=np.float64)
        out = NpFourierOp()(Sample(input=x))
        recovered = np.fft.ifft(out.input)
        assert np.allclose(recovered.real, x, atol=1e-9)

    def test_n_zero_pads(self) -> None:
        x = np.arange(4, dtype=np.float64)
        out = NpFourierOp(n=8)(Sample(input=x))
        assert out.input.shape == (8,)
        assert np.allclose(out.input, np.fft.fft(x, n=8))

    def test_n_truncates(self) -> None:
        x = np.arange(8, dtype=np.float64)
        out = NpFourierOp(n=4)(Sample(input=x))
        assert out.input.shape == (4,)
        assert np.allclose(out.input, np.fft.fft(x, n=4))

    def test_axis_transforms_per_row_of_batch(self) -> None:
        x = np.random.RandomState(0).randn(3, 8)
        out = NpFourierOp(axis=-1)(Sample(input=x))
        assert out.input.shape == (3, 8)
        # Each row transformed independently == the per-row 1-D FFT.
        for i in range(3):
            assert np.allclose(out.input[i], np.fft.fft(x[i]))

    def test_axis_zero(self) -> None:
        x = np.random.RandomState(1).randn(8, 3)
        out = NpFourierOp(axis=0)(Sample(input=x))
        assert np.allclose(out.input, np.fft.fft(x, axis=0))

    @pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
    def test_norm_modes_match_numpy(self, norm: FourierNorm) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        out = NpFourierOp(norm=norm)(Sample(input=x))
        assert np.allclose(out.input, np.fft.fft(x, norm=norm))

    def test_shift_flag_matches_manual_fftshift(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # odd length — fftshift is non-trivial
        shifted = NpFourierOp(shift=True)(Sample(input=x)).input
        assert np.allclose(shifted, np.fft.fftshift(NpFourierOp()(Sample(input=x)).input))

    def test_shift_defaults_off(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        plain = NpFourierOp()(Sample(input=x)).input
        assert np.allclose(NpFourierOp(shift=False)(Sample(input=x)).input, plain)
        assert not np.allclose(NpFourierOp(shift=True)(Sample(input=x)).input, plain)

    def test_preserves_target_and_metadata(self) -> None:
        out = NpFourierOp()(Sample(input=np.ones(4), target=5, metadata={"k": "v"}))
        assert out.target == 5
        assert out.meta == {"k": "v"}

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="FourierOp expects an np.ndarray"):
            NpFourierOp()(Sample(input=torch.zeros(4)))

    def test_zero_arg_construction(self) -> None:
        op = NpFourierOp()
        assert op.n is None and op.axis == -1 and op.norm == "backward"

    def test_invalid_norm_rejected_at_construction(self) -> None:
        # ``norm`` is a closed ``Literal`` — confluid's pydantic schema rejects an out-of-set value.
        with pytest.raises((ValueError, ValidationError)):
            NpFourierOp(norm="bogus")  # type: ignore[arg-type]

    def test_produces_contract_conforms_to_real_output(self) -> None:
        out = NpFourierOp()(Sample(input=np.ones(4, dtype=np.float32)))
        assert NpFourierOp.PRODUCES.accepts(infer_sample_type(out))

    def test_produces_contract_conforms_to_complex_output(self) -> None:
        x = np.array([1 + 2j, 3 - 1j], dtype=np.complex128)
        out = NpFourierOp()(Sample(input=x))
        assert NpFourierOp.PRODUCES.accepts(infer_sample_type(out))


# ---------------------------------------------------------------------------
# torch FourierOp
# ---------------------------------------------------------------------------


class TestTorchFourierOp:
    def test_real_float32_matches_torch_and_is_complex64(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = TorchFourierOp()(Sample(input=t))
        assert isinstance(out.input, torch.Tensor)
        assert out.input.dtype == torch.complex64
        assert torch.allclose(out.input, torch.fft.fft(t))

    def test_real_float64_promotes_to_complex128(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        out = TorchFourierOp()(Sample(input=t))
        assert out.input.dtype == torch.complex128
        assert torch.allclose(out.input, torch.fft.fft(t))

    def test_integer_input_handled_natively(self) -> None:
        # torch.fft.fft auto-promotes integer tensors to complex64 — no manual cast needed.
        t = torch.arange(8)
        out = TorchFourierOp()(Sample(input=t))
        assert out.input.dtype == torch.complex64
        assert torch.allclose(out.input, torch.fft.fft(t))

    def test_bool_input_handled_natively(self) -> None:
        t = torch.tensor([True, False, True, True])
        out = TorchFourierOp()(Sample(input=t))
        assert out.input.dtype == torch.complex64

    def test_complex64_input_stays_complex64(self) -> None:
        t = torch.tensor([1 + 2j, 3 - 1j, 0 + 0j, -2 + 1j], dtype=torch.complex64)
        out = TorchFourierOp()(Sample(input=t))
        assert out.input.dtype == torch.complex64
        assert torch.allclose(out.input, torch.fft.fft(t))

    def test_float16_promoted_to_float32_without_mutating_input(self) -> None:
        # torch.fft.fft rejects half precision; the op promotes to float32 first. The promotion
        # is local — the caller's tensor is untouched.
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
        out = TorchFourierOp()(Sample(input=t))
        assert out.input.dtype == torch.complex64
        assert t.dtype == torch.float16
        assert torch.allclose(out.input, torch.fft.fft(t.float()))

    def test_bfloat16_promoted_to_float32(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
        out = TorchFourierOp()(Sample(input=t))
        assert out.input.dtype == torch.complex64

    def test_constant_signal_has_only_dc_component(self) -> None:
        out = TorchFourierOp()(Sample(input=torch.ones(4)))
        assert torch.allclose(out.input, torch.tensor([4, 0, 0, 0], dtype=torch.complex64))

    def test_roundtrip_via_ifft_recovers_input(self) -> None:
        t = torch.tensor([1.0, -2.0, 3.5, 0.0, 7.0], dtype=torch.float64)
        out = TorchFourierOp()(Sample(input=t))
        recovered = torch.fft.ifft(out.input)
        assert torch.allclose(recovered.real, t, atol=1e-9)

    def test_n_zero_pads(self) -> None:
        t = torch.arange(4, dtype=torch.float64)
        out = TorchFourierOp(n=8)(Sample(input=t))
        assert out.input.shape == (8,)
        assert torch.allclose(out.input, torch.fft.fft(t, n=8))

    def test_n_truncates(self) -> None:
        t = torch.arange(8, dtype=torch.float64)
        out = TorchFourierOp(n=4)(Sample(input=t))
        assert out.input.shape == (4,)

    def test_dim_transforms_per_row_of_batch(self) -> None:
        t = torch.randn(3, 8)
        out = TorchFourierOp(dim=-1)(Sample(input=t))
        assert out.input.shape == (3, 8)
        for i in range(3):
            assert torch.allclose(out.input[i], torch.fft.fft(t[i]))

    def test_dim_zero(self) -> None:
        t = torch.randn(8, 3)
        out = TorchFourierOp(dim=0)(Sample(input=t))
        assert torch.allclose(out.input, torch.fft.fft(t, dim=0))

    @pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
    def test_norm_modes_match_torch(self, norm: FourierNorm) -> None:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        out = TorchFourierOp(norm=norm)(Sample(input=t))
        assert torch.allclose(out.input, torch.fft.fft(t, norm=norm))

    def test_shift_flag_matches_manual_fftshift(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # odd length — fftshift is non-trivial
        shifted = TorchFourierOp(shift=True)(Sample(input=t)).input
        assert torch.allclose(shifted, torch.fft.fftshift(TorchFourierOp()(Sample(input=t)).input))

    def test_shift_defaults_off(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        plain = TorchFourierOp()(Sample(input=t)).input
        assert torch.allclose(TorchFourierOp(shift=False)(Sample(input=t)).input, plain)
        assert not torch.allclose(TorchFourierOp(shift=True)(Sample(input=t)).input, plain)

    def test_preserves_target_and_metadata(self) -> None:
        out = TorchFourierOp()(Sample(input=torch.ones(4), target=7, metadata={"k": "v"}))
        assert out.target == 7
        assert out.meta == {"k": "v"}

    def test_raises_on_non_tensor(self) -> None:
        with pytest.raises(TypeError, match="FourierOp expects a torch.Tensor"):
            TorchFourierOp()(Sample(input=np.zeros(4)))

    def test_zero_arg_construction(self) -> None:
        op = TorchFourierOp()
        assert op.n is None and op.dim == -1 and op.norm == "backward"

    def test_invalid_norm_rejected_at_construction(self) -> None:
        with pytest.raises((ValueError, ValidationError)):
            TorchFourierOp(norm="bogus")  # type: ignore[arg-type]

    def test_produces_contract_conforms_to_real_output(self) -> None:
        out = TorchFourierOp()(Sample(input=torch.ones(4)))
        assert TorchFourierOp.PRODUCES.accepts(infer_sample_type(out))

    def test_produces_contract_conforms_to_complex_output(self) -> None:
        t = torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex128)
        out = TorchFourierOp()(Sample(input=t))
        assert TorchFourierOp.PRODUCES.accepts(infer_sample_type(out))


# ---------------------------------------------------------------------------
# numpy InverseFourierOp
# ---------------------------------------------------------------------------


class TestNumpyInverseFourierOp:
    def test_matches_numpy_ifft_and_is_complex(self) -> None:
        x = np.array([10.0, -2.0, 0.0, 4.0])
        out = NpInverseFourierOp()(Sample(input=x))
        assert out.input.dtype == np.complex128
        assert np.allclose(out.input, np.fft.ifft(x))

    def test_inverts_forward_transform(self) -> None:
        x = np.array([1.0, -2.0, 3.5, 0.0, 7.0], dtype=np.float64)  # odd length
        spectrum = NpFourierOp()(Sample(input=x))
        recovered = NpInverseFourierOp()(spectrum)
        assert np.allclose(recovered.input.real, x, atol=1e-9)

    def test_shift_inverts_forward_shift(self) -> None:
        # InverseFourierOp(shift=True) exactly undoes FourierOp(shift=True), odd length included.
        x = np.array([1.0, -2.0, 3.5, 0.0, 7.0], dtype=np.float64)
        centered = NpFourierOp(shift=True)(Sample(input=x))
        recovered = NpInverseFourierOp(shift=True)(centered)
        assert np.allclose(recovered.input.real, x, atol=1e-9)

    @pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
    def test_roundtrip_under_each_norm(self, norm: FourierNorm) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        spectrum = NpFourierOp(norm=norm)(Sample(input=x))
        recovered = NpInverseFourierOp(norm=norm)(spectrum)
        assert np.allclose(recovered.input.real, x, atol=1e-9)

    def test_n_truncates(self) -> None:
        x = np.arange(8, dtype=np.float64)
        out = NpInverseFourierOp(n=4)(Sample(input=x))
        assert out.input.shape == (4,)
        assert np.allclose(out.input, np.fft.ifft(x, n=4))

    def test_preserves_target_and_metadata(self) -> None:
        out = NpInverseFourierOp()(Sample(input=np.ones(4), target=5, metadata={"k": "v"}))
        assert out.target == 5
        assert out.meta == {"k": "v"}

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="InverseFourierOp expects an np.ndarray"):
            NpInverseFourierOp()(Sample(input=torch.zeros(4)))

    def test_zero_arg_construction(self) -> None:
        op = NpInverseFourierOp()
        assert op.n is None and op.axis == -1 and op.norm == "backward" and op.shift is False

    def test_invalid_norm_rejected_at_construction(self) -> None:
        with pytest.raises((ValueError, ValidationError)):
            NpInverseFourierOp(norm="bogus")  # type: ignore[arg-type]

    def test_produces_contract_conforms(self) -> None:
        out = NpInverseFourierOp()(Sample(input=np.ones(4)))
        assert NpInverseFourierOp.PRODUCES.accepts(infer_sample_type(out))


# ---------------------------------------------------------------------------
# torch InverseFourierOp
# ---------------------------------------------------------------------------


class TestTorchInverseFourierOp:
    def test_matches_torch_ifft_and_is_complex(self) -> None:
        t = torch.tensor([10.0, -2.0, 0.0, 4.0])
        out = TorchInverseFourierOp()(Sample(input=t))
        assert out.input.dtype == torch.complex64
        assert torch.allclose(out.input, torch.fft.ifft(t))

    def test_inverts_forward_transform(self) -> None:
        t = torch.tensor([1.0, -2.0, 3.5, 0.0, 7.0], dtype=torch.float64)
        spectrum = TorchFourierOp()(Sample(input=t))
        recovered = TorchInverseFourierOp()(spectrum)
        assert torch.allclose(recovered.input.real, t, atol=1e-9)

    def test_shift_inverts_forward_shift(self) -> None:
        t = torch.tensor([1.0, -2.0, 3.5, 0.0, 7.0], dtype=torch.float64)
        centered = TorchFourierOp(shift=True)(Sample(input=t))
        recovered = TorchInverseFourierOp(shift=True)(centered)
        assert torch.allclose(recovered.input.real, t, atol=1e-9)

    def test_integer_input_handled_natively(self) -> None:
        out = TorchInverseFourierOp()(Sample(input=torch.arange(8)))
        assert out.input.dtype == torch.complex64

    def test_float16_promoted_without_mutating_input(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
        out = TorchInverseFourierOp()(Sample(input=t))
        assert out.input.dtype == torch.complex64
        assert t.dtype == torch.float16

    def test_raises_on_non_tensor(self) -> None:
        with pytest.raises(TypeError, match="InverseFourierOp expects a torch.Tensor"):
            TorchInverseFourierOp()(Sample(input=np.zeros(4)))

    def test_zero_arg_construction(self) -> None:
        op = TorchInverseFourierOp()
        assert op.n is None and op.dim == -1 and op.norm == "backward" and op.shift is False

    def test_produces_contract_conforms(self) -> None:
        out = TorchInverseFourierOp()(Sample(input=torch.ones(4)))
        assert TorchInverseFourierOp.PRODUCES.accepts(infer_sample_type(out))


# ---------------------------------------------------------------------------
# fftshift / ifftshift ops (numpy + torch) — pure, dtype-preserving rearrangements
# ---------------------------------------------------------------------------


class TestNumpyShiftOps:
    def test_fftshift_matches_numpy(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(NpFftShiftOp()(Sample(input=x)).input, np.fft.fftshift(x))

    def test_ifftshift_matches_numpy(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(NpIfftShiftOp()(Sample(input=x)).input, np.fft.ifftshift(x))

    def test_ifftshift_inverts_fftshift_odd_length(self) -> None:
        x = np.arange(5)
        shifted = NpFftShiftOp()(Sample(input=x))
        restored = NpIfftShiftOp()(shifted)
        assert np.array_equal(restored.input, x)

    def test_preserves_dtype_integer_and_complex(self) -> None:
        assert NpFftShiftOp()(Sample(input=np.arange(5))).input.dtype == np.int64
        cx = np.array([1 + 1j, 2 - 2j, 3j], dtype=np.complex64)
        assert NpFftShiftOp()(Sample(input=cx)).input.dtype == np.complex64

    def test_axis_shifts_per_row(self) -> None:
        x = np.arange(15).reshape(3, 5)
        assert np.allclose(NpFftShiftOp(axis=-1)(Sample(input=x)).input, np.fft.fftshift(x, axes=-1))

    def test_axis_none_shifts_all_axes(self) -> None:
        x = np.arange(15).reshape(3, 5)
        assert np.allclose(NpFftShiftOp(axis=None)(Sample(input=x)).input, np.fft.fftshift(x))

    def test_preserves_target_and_metadata(self) -> None:
        out = NpFftShiftOp()(Sample(input=np.arange(4), target=9, metadata={"k": "v"}))
        assert out.target == 9
        assert out.meta == {"k": "v"}

    def test_raises_on_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match="FftShiftOp expects an np.ndarray"):
            NpFftShiftOp()(Sample(input=torch.zeros(4)))
        with pytest.raises(TypeError, match="IfftShiftOp expects an np.ndarray"):
            NpIfftShiftOp()(Sample(input=torch.zeros(4)))

    def test_zero_arg_construction(self) -> None:
        assert NpFftShiftOp().axis == -1
        assert NpIfftShiftOp().axis == -1

    def test_produces_contract_conforms(self) -> None:
        out = NpFftShiftOp()(Sample(input=np.arange(5)))
        assert NpFftShiftOp.PRODUCES.accepts(infer_sample_type(out))


class TestTorchShiftOps:
    def test_fftshift_matches_torch(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert torch.allclose(TorchFftShiftOp()(Sample(input=t)).input, torch.fft.fftshift(t))

    def test_ifftshift_matches_torch(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert torch.allclose(TorchIfftShiftOp()(Sample(input=t)).input, torch.fft.ifftshift(t))

    def test_ifftshift_inverts_fftshift_odd_length(self) -> None:
        t = torch.arange(5)
        restored = TorchIfftShiftOp()(TorchFftShiftOp()(Sample(input=t)))
        assert torch.equal(restored.input, t)

    def test_preserves_dtype_half_and_integer(self) -> None:
        # Pure rearrangement — no FFT — so half precision (which the FFT ops reject) passes through.
        assert TorchFftShiftOp()(Sample(input=torch.arange(5))).input.dtype == torch.int64
        half = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
        assert TorchFftShiftOp()(Sample(input=half)).input.dtype == torch.float16

    def test_dim_shifts_per_row(self) -> None:
        t = torch.arange(15).reshape(3, 5)
        assert torch.equal(TorchFftShiftOp(dim=-1)(Sample(input=t)).input, torch.fft.fftshift(t, dim=-1))

    def test_dim_none_shifts_all_dims(self) -> None:
        t = torch.arange(15).reshape(3, 5)
        assert torch.equal(TorchFftShiftOp(dim=None)(Sample(input=t)).input, torch.fft.fftshift(t))

    def test_raises_on_non_tensor(self) -> None:
        with pytest.raises(TypeError, match="FftShiftOp expects a torch.Tensor"):
            TorchFftShiftOp()(Sample(input=np.zeros(4)))
        with pytest.raises(TypeError, match="IfftShiftOp expects a torch.Tensor"):
            TorchIfftShiftOp()(Sample(input=np.zeros(4)))

    def test_zero_arg_construction(self) -> None:
        assert TorchFftShiftOp().dim == -1
        assert TorchIfftShiftOp().dim == -1

    def test_produces_contract_conforms(self) -> None:
        out = TorchFftShiftOp()(Sample(input=torch.arange(5)))
        assert TorchFftShiftOp.PRODUCES.accepts(infer_sample_type(out))


# ---------------------------------------------------------------------------
# Full pipeline round-trips combining the ops
# ---------------------------------------------------------------------------


class TestRoundTrips:
    def test_numpy_fft_shift_unshift_ifft_recovers(self) -> None:
        # FourierOp -> FftShiftOp -> IfftShiftOp -> InverseFourierOp == identity (real signal).
        x = np.array([1.0, -2.0, 3.5, 0.0, 7.0], dtype=np.float64)
        s = NpFourierOp()(Sample(input=x))
        s = NpFftShiftOp()(s)
        s = NpIfftShiftOp()(s)
        s = NpInverseFourierOp()(s)
        assert np.allclose(s.input.real, x, atol=1e-9)

    def test_torch_fft_shift_unshift_ifft_recovers(self) -> None:
        t = torch.tensor([1.0, -2.0, 3.5, 0.0, 7.0], dtype=torch.float64)
        s = TorchFourierOp()(Sample(input=t))
        s = TorchFftShiftOp()(s)
        s = TorchIfftShiftOp()(s)
        s = TorchInverseFourierOp()(s)
        assert torch.allclose(s.input.real, t, atol=1e-9)


# ---------------------------------------------------------------------------
# Windowing + unit scaling (WindowOp / SpectrumScalingOp + FourierOp options)
# ---------------------------------------------------------------------------


def _np_tone(n: int = 1024, bin_index: int = 64, amp: float = 1.0) -> np.ndarray:
    return np.asarray(amp * np.exp(2j * np.pi * bin_index * np.arange(n) / n), dtype=np.complex64)


class TestNumpyWindowAndScaling:
    def test_default_fourierop_unchanged_and_metadata_byte_identical(self) -> None:
        x = _np_tone()
        s = Sample(input=x, target=None, metadata={"k": 1})
        out = NpFourierOp()(s)
        assert np.allclose(out.input, np.fft.fft(x)) and out.input.dtype == np.complex64
        assert out.metadata == s.metadata  # boxcar/none stamps nothing

    def test_window_option_stashes_correction(self) -> None:
        out = NpFourierOp(window="hann")(Sample(input=_np_tone(), target=None, metadata={}))
        assert out.meta["window"] == "hann"
        assert out.meta["window_sum"] == pytest.approx(512.0)
        assert out.meta["window_enbw_bins"] == pytest.approx(1.5)

    def test_windowop_preserves_complex64_dtype(self) -> None:
        out = NpWindowOp(window="hann")(Sample(input=_np_tone(), target=None, metadata={}))
        assert out.input.dtype == np.complex64  # not upcast to complex128 by the float window
        assert out.meta["window_sum"] == pytest.approx(512.0)

    def test_amplitude_and_power_recover_tone(self) -> None:
        s = Sample(input=_np_tone(amp=1.0), target=None, metadata={})
        amp = NpFourierOp(window="hann", scaling="amplitude")(s).input
        pw = NpFourierOp(window="hann", scaling="power")(s).input
        assert np.abs(amp).max() == pytest.approx(1.0, abs=1e-4)
        assert pw.max() == pytest.approx(1.0, abs=1e-4)
        assert np.iscomplexobj(amp) and not np.iscomplexobj(pw)

    def test_one_node_equals_explicit_chain(self) -> None:
        s = Sample(input=_np_tone(), target=None, metadata={"samplerate": 1000.0})
        one = NpFourierOp(window="hann", scaling="density", sample_rate=1000.0)(s).input
        chain = NpSpectrumScalingOp(scaling="density", sample_rate=1000.0)(
            NpFourierOp()(NpWindowOp(window="hann")(s))
        ).input
        assert np.allclose(one, chain)

    def test_spectrumscaling_rectangular_fallback_without_metadata(self) -> None:
        s = Sample(input=_np_tone(), target=None, metadata={})
        raw = NpFourierOp()(s)  # boxcar default → no window correction stashed
        assert WINDOW_SUM_KEY not in raw.metadata
        pw = NpSpectrumScalingOp(scaling="power")(raw).input
        n = raw.input.shape[-1]
        assert np.allclose(pw, np.abs(raw.input) ** 2 / n**2)  # rectangular S1=S2=N fallback

    def test_scaling_requires_backward_norm(self) -> None:
        s = Sample(input=_np_tone(), target=None, metadata={})
        with pytest.raises(ValueError, match="norm='backward'"):
            NpFourierOp(scaling="power", norm="ortho")(s)

    def test_shift_and_one_sided_mutually_exclusive(self) -> None:
        s = Sample(input=_np_tone(), target=None, metadata={})
        with pytest.raises(ValueError, match="mutually exclusive"):
            NpFourierOp(scaling="power", shift=True, one_sided=True)(s)

    def test_one_sided_real_signal_amplitude(self) -> None:
        n = 1024
        t = np.arange(n)
        x = (2.0 * np.cos(2 * np.pi * 16 * t / n)).astype(np.float64)
        amp = NpFourierOp(scaling="amplitude", one_sided=True)(Sample(input=x, target=None, metadata={})).input
        assert amp.shape[0] == n // 2 + 1
        assert np.abs(amp).max() == pytest.approx(2.0, abs=1e-6)

    def test_produces_accepts_complex_and_real(self) -> None:
        s = Sample(input=_np_tone(), target=None, metadata={})
        assert NpFourierOp.PRODUCES.accepts(infer_sample_type(NpFourierOp()(s)))  # complex
        assert NpFourierOp.PRODUCES.accepts(infer_sample_type(NpFourierOp(scaling="power")(s)))  # real
        scaled = NpSpectrumScalingOp(scaling="power")(NpFourierOp()(s))
        assert NpSpectrumScalingOp.PRODUCES.accepts(infer_sample_type(scaled))

    def test_sample_rate_from_metadata(self) -> None:
        s = Sample(input=_np_tone(), target=None, metadata={"samplerate": 500.0})
        from_meta = NpFourierOp(window="hann", scaling="density")(s).input
        explicit = NpFourierOp(window="hann", scaling="density", sample_rate=500.0)(s).input
        assert np.allclose(from_meta, explicit)

    def test_spectrumscaling_density_no_rate_runs(self) -> None:
        # exercises the normalized-frequency fallback (Fs=1.0) + debug log branch
        s = Sample(input=_np_tone(), target=None, metadata={})
        out = NpSpectrumScalingOp(scaling="density")(NpFourierOp()(s)).input
        assert out.shape == (1024,) and not np.iscomplexobj(out)


class TestTorchWindowAndScaling:
    def test_default_unchanged(self) -> None:
        x = torch.view_as_complex(torch.randn(1024, 2))
        out = TorchFourierOp()(Sample(input=x, target=None, metadata={"k": 1}))
        assert torch.allclose(out.input, torch.fft.fft(x))
        assert out.metadata == {"k": 1}  # boxcar/none stamps nothing

    def test_amplitude_recovers_tone(self) -> None:
        n = 1024
        x = torch.exp(2j * torch.pi * 64 * torch.arange(n) / n).to(torch.complex64)
        amp = TorchFourierOp(window="hann", scaling="amplitude")(Sample(input=x, target=None, metadata={})).input
        assert amp.abs().max().item() == pytest.approx(1.0, abs=1e-3)

    def test_window_stashes_and_preserves_dtype(self) -> None:
        x = torch.exp(2j * torch.pi * 64 * torch.arange(1024) / 1024).to(torch.complex64)
        out = TorchWindowOp(window="hann")(Sample(input=x, target=None, metadata={}))
        assert out.input.dtype == torch.complex64
        assert out.meta["window_sum"] == pytest.approx(512.0)

    def test_power_is_real_and_scaling_requires_backward_norm(self) -> None:
        x = torch.view_as_complex(torch.randn(64, 2))
        pw = TorchFourierOp(window="hann", scaling="power")(Sample(input=x, target=None, metadata={})).input
        assert not pw.is_complex()
        with pytest.raises(ValueError, match="norm='backward'"):
            TorchFourierOp(scaling="power", norm="ortho")(Sample(input=x, target=None, metadata={}))

    def test_spectrumscaling_standalone_matches_one_node(self) -> None:
        n = 256
        x = torch.exp(2j * torch.pi * 20 * torch.arange(n) / n).to(torch.complex64)
        s = Sample(input=x, target=None, metadata={})
        one = TorchFourierOp(window="hamming", scaling="power")(s).input
        chain = TorchSpectrumScalingOp(scaling="power")(TorchFourierOp()(TorchWindowOp(window="hamming")(s))).input
        assert torch.allclose(one, chain, atol=1e-4)


def test_numpy_torch_parity_window_and_scaling() -> None:
    n = 512
    base = np.exp(2j * np.pi * 40 * np.arange(n) / n).astype(np.complex64)
    meta = {"samplerate": 2000.0}
    for scaling in ("none", "amplitude", "power", "density"):
        npo = NpFourierOp(window="blackmanharris", scaling=scaling, sample_rate=2000.0)(
            Sample(input=base.copy(), target=None, metadata=dict(meta))
        ).input
        to = TorchFourierOp(window="blackmanharris", scaling=scaling, sample_rate=2000.0)(
            Sample(input=torch.from_numpy(base.copy()), target=None, metadata=dict(meta))
        ).input
        assert np.allclose(npo, to.numpy(), rtol=1e-3, atol=1e-3), scaling
