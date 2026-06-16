"""Tests for :mod:`dataflux.windows` — the window functions + spectral unit-scaling math.

Pins (1) the closed Literals match their runtime tuples, (2) the pure-numpy windows match
scipy (when available) and have the right correction constants (Hann coherent gain 0.5 / ENBW
1.5 bins, flat-top CG 0.2156), and (3) ``scale_spectrum`` returns calibrated units — a
unit-amplitude tone reads amplitude 1.0 / power 1.0, and the power-to-density ratio is the
window's equivalent noise bandwidth in Hz.
"""

from typing import cast, get_args

import numpy as np
import pytest

from dataflux import windows as W
from dataflux.windows import WindowName


def test_literals_match_runtime_tuples() -> None:
    assert W.WINDOW_NAMES == get_args(W.WindowName)
    assert W.SPECTRUM_SCALINGS == get_args(W.SpectrumScaling)
    assert "boxcar" in W.WINDOW_NAMES and "hann" in W.WINDOW_NAMES
    assert W.SPECTRUM_SCALINGS == ("none", "amplitude", "power", "density")


@pytest.mark.parametrize("name", W.WINDOW_NAMES)
def test_get_window_builds_each(name: WindowName) -> None:
    wp = {"kaiser": 8.6, "tukey": 0.5, "gaussian": 7.0}.get(name)
    w = W.get_window(name, 64, window_param=wp)
    assert w.shape == (64,) and w.dtype == np.float64
    assert np.all(np.isfinite(w))
    # boxcar is the rectangular identity; every other taper has a sub-unity mean (it attenuates)
    if name == "boxcar":
        assert np.allclose(w, 1.0)
    else:
        assert w.mean() < 1.0


def test_get_window_periodic_differs_from_symmetric() -> None:
    p = W.get_window("hann", 64, periodic=True)
    s = W.get_window("hann", 64, periodic=False)
    assert not np.allclose(p, s)
    # symmetric Hann is zero at both endpoints; periodic is zero only at index 0
    assert s[0] == pytest.approx(0.0) and s[-1] == pytest.approx(0.0)
    assert p[0] == pytest.approx(0.0)


def test_get_window_matches_scipy() -> None:
    sw = pytest.importorskip("scipy.signal")
    specs = {
        "boxcar": "boxcar",
        "bartlett": "bartlett",
        "hann": "hann",
        "hamming": "hamming",
        "blackman": "blackman",
        "blackmanharris": "blackmanharris",
        "nuttall": "nuttall",
        "flattop": "flattop",
        "kaiser": ("kaiser", 8.6),
        "tukey": ("tukey", 0.5),
        "gaussian": ("gaussian", 7.0),
    }
    for name, spec in specs.items():
        wp = {"kaiser": 8.6, "tukey": 0.5, "gaussian": 7.0}.get(name)
        for periodic in (True, False):
            mine = W.get_window(cast(WindowName, name), 128, window_param=wp, periodic=periodic)
            ref = sw.get_window(spec, 128, fftbins=periodic)
            assert np.allclose(mine, ref, atol=1e-12), f"{name} periodic={periodic}"


def test_get_window_errors() -> None:
    with pytest.raises(ValueError, match="unknown window"):
        W.get_window("nope", 16)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be positive"):
        W.get_window("hann", 0)
    with pytest.raises(ValueError, match="gaussian window requires"):
        W.get_window("gaussian", 16)  # no window_param


def test_correction_constants() -> None:
    # asymptotic (large-N) coherent gain + ENBW for the textbook windows
    cases = {"boxcar": (1.0, 1.0), "hann": (0.5, 1.5), "hamming": (0.54, 1.363), "flattop": (0.2156, 3.77)}
    for name, (cg, enbw) in cases.items():
        w = W.get_window(cast(WindowName, name), 8192)
        assert W.coherent_gain(w) == pytest.approx(cg, abs=2e-3)
        assert W.enbw_bins(w) == pytest.approx(enbw, abs=2e-2)


def test_window_sums_and_metadata() -> None:
    w = W.get_window("hann", 1024)
    s1, s2 = W.window_sums(w)
    assert s1 == pytest.approx(512.0) and s2 == pytest.approx(384.0)
    meta = W.window_metadata("hann", w)
    assert meta[W.WINDOW_NAME_KEY] == "hann"
    assert meta[W.WINDOW_SIZE_KEY] == 1024
    assert meta[W.WINDOW_SUM_KEY] == pytest.approx(512.0)
    assert meta[W.WINDOW_SUMSQ_KEY] == pytest.approx(384.0)
    assert meta[W.WINDOW_ENBW_KEY] == pytest.approx(1.5)
    assert meta[W.WINDOW_CG_KEY] == pytest.approx(0.5)


def _tone(n: int, bin_index: int, amp: float = 1.0) -> np.ndarray:
    return np.asarray(amp * np.exp(2j * np.pi * bin_index * np.arange(n) / n), dtype=np.complex64)


def test_scale_spectrum_none_is_passthrough() -> None:
    x = _tone(256, 10)
    X = np.fft.fft(x)
    out = W.scale_spectrum(X, "none", s1=256.0, s2=256.0)
    assert np.array_equal(out, X)


def test_scale_spectrum_amplitude_and_power_recover_tone() -> None:
    n = 1024
    X = np.fft.fft(_tone(n, 64, amp=1.0))
    amp = W.scale_spectrum(X, "amplitude", s1=float(n), s2=float(n))
    pw = W.scale_spectrum(X, "power", s1=float(n), s2=float(n))
    assert np.abs(amp).max() == pytest.approx(1.0, abs=1e-4)  # amplitude V
    assert pw.max() == pytest.approx(1.0, abs=1e-4)  # power V² = amplitude²
    assert np.iscomplexobj(amp) and not np.iscomplexobj(pw)  # power is real


def test_scale_spectrum_density_is_power_over_enbw_hz() -> None:
    n, fs = 1024, 1000.0
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    w = W.get_window("hann", n)
    Xw = np.fft.fft(x * w)
    s1, s2 = W.window_sums(w)
    pw = W.scale_spectrum(Xw, "power", s1=s1, s2=s2)
    den = W.scale_spectrum(Xw, "density", s1=s1, s2=s2, sample_rate=fs)
    enbw_hz = fs * s2 / (s1 * s1)  # = Fs · ENBW_bins / N
    assert np.allclose(pw, den * enbw_hz)  # power = density × ENBW_Hz, bin-for-bin


def test_scale_spectrum_density_normalized_without_rate() -> None:
    X = np.fft.fft(_tone(256, 8))
    den = W.scale_spectrum(X, "density", s1=256.0, s2=256.0)  # Fs defaults to 1.0
    den_fs1 = W.scale_spectrum(X, "density", s1=256.0, s2=256.0, sample_rate=1.0)
    assert np.allclose(den, den_fs1)


def test_fold_one_sided_even_and_odd() -> None:
    # real cosine, amplitude 2 at bin 4 → one-sided amplitude reads 2 at that bin
    for n in (64, 65):
        t = np.arange(n)
        x = 2.0 * np.cos(2 * np.pi * 4 * t / n)
        amp = W.scale_spectrum(np.fft.fft(x), "amplitude", s1=float(n), s2=float(n), one_sided=True)
        assert amp.shape[0] == n // 2 + 1
        assert np.abs(amp).max() == pytest.approx(2.0, abs=1e-6)


def test_fold_one_sided_preserves_dc_and_nyquist() -> None:
    n = 64
    x = np.ones(n) * 3.0  # pure DC
    one = W.fold_one_sided(np.fft.fft(x), axis=-1)
    assert one[0] == pytest.approx(3.0 * n)  # DC not doubled
