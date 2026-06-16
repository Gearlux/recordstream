"""Window functions + spectral unit-scaling — the math home for the Fourier ops.

A raw FFT is *uncalibrated*: to read a spectrum in real units you must (1) taper the
signal with a window to control spectral leakage and (2) divide out the window's gain.
This module is the single, framework-neutral (pure-numpy — scipy is only an optional
dataflux dependency) source of both:

* :func:`get_window` builds the taper (``WindowName`` — Hann, Hamming, Blackman-Harris,
  flat-top, Kaiser, …).
* :func:`window_sums` / :func:`coherent_gain` / :func:`enbw_bins` give the correction
  factors — coherent gain ``S1 = Σw`` (amplitude) and ``S2 = Σw²`` with the equivalent
  noise bandwidth (power-spectral density).
* :func:`scale_spectrum` turns a windowed FFT into the chosen ``SpectrumScaling`` units
  (amplitude V, power V², density V²/Hz).

It is a library module (like :mod:`dataflux.labels` / :mod:`dataflux.projection`), **not**
``@configurable`` and not entry-pointed. The numpy ops in :mod:`dataflux.ops.numpy`
(``WindowOp`` / ``SpectrumScalingOp`` / ``FourierOp``) and their torch mirrors in
:mod:`dataflux.ops.torch` all reuse it — the torch ops take the numpy window coefficients
and the scalar ``S1``/``S2`` corrections, then do the array arithmetic with torch.

Calibration assumes the **unscaled forward transform** (``numpy.fft.fft`` /
``torch.fft.fft`` with ``norm="backward"`` — the default). The amplitude/power/density
formulas are only meaningful for that normalization, so the ops reject a non-``backward``
``norm`` combined with a unit ``scaling`` rather than emit silently-wrong units.
"""

from typing import Dict, Literal, Optional, Tuple, get_args

import numpy as np

# --- closed Literals (workspace "prefer closed Literals over bare strings" mandate) ---
# The supported window tapers. ``boxcar`` is the rectangular window (all ones) — i.e. *no*
# taper, the identity — and is the default for FourierOp so its behaviour is unchanged.
WindowName = Literal[
    "boxcar",
    "bartlett",
    "hann",
    "hamming",
    "blackman",
    "blackmanharris",
    "nuttall",
    "flattop",
    "kaiser",
    "tukey",
    "gaussian",
]
WINDOW_NAMES: Tuple[WindowName, ...] = get_args(WindowName)

# Spectral unit-scaling modes. ``none`` = raw FFT (complex, unchanged); ``amplitude`` =
# amplitude spectrum (V, complex); ``power`` = power spectrum (V², real); ``density`` =
# power spectral density (V²/Hz, real). NB this is the GENERAL scaling set — distinct from
# the narrower matplotlib-style ``waivefront.visualizers.SpectrumScaling`` (density/spectrum),
# which is a different module modelling matplotlib's ``scale_by_freq`` toggle.
SpectrumScaling = Literal["none", "amplitude", "power", "density"]
SPECTRUM_SCALINGS: Tuple[SpectrumScaling, ...] = get_args(SpectrumScaling)

# --- metadata keys: the window correction stashed by WindowOp / FourierOp (when a real
# window is applied) and read back by SpectrumScalingOp so a spectrum computed in one node
# can be scaled to units in another. ``window`` here is the FFT *taper* name — unrelated to
# waivefront's ``window_start_sample`` (a time-slice index). ---
WINDOW_NAME_KEY = "window"
WINDOW_SIZE_KEY = "window_size"  # N (number of taps)
WINDOW_SUM_KEY = "window_sum"  # S1 = Σw  (coherent-gain numerator)
WINDOW_SUMSQ_KEY = "window_sum_sq"  # S2 = Σw²
WINDOW_ENBW_KEY = "window_enbw_bins"  # equivalent noise bandwidth, N·S2/S1² (bins)
WINDOW_CG_KEY = "window_coherent_gain"  # S1/N

# Generalized-cosine coefficients (scipy / Harris-1978 convention): w[n] = Σ_k a_k·cos(k·φ)
# with φ ∈ [-π, π] over the taps. The alternating shape is carried by cos(k·φ), so the
# coefficients are all positive and sum to 1 at the centre (coherent gain ≈ a_0).
_COSINE_COEFFS: Dict[str, Tuple[float, ...]] = {
    "hann": (0.5, 0.5),
    "hamming": (0.54, 0.46),
    "blackman": (0.42, 0.5, 0.08),
    "blackmanharris": (0.35875, 0.48829, 0.14128, 0.01168),
    "nuttall": (0.3635819, 0.4891775, 0.1365995, 0.0106411),
    "flattop": (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368),
}


def _general_cosine(n: int, coeffs: Tuple[float, ...], periodic: bool) -> np.ndarray:
    """Generalized-cosine window of length ``n`` (the Hann/Hamming/Blackman/… family).

    ``periodic=True`` (the DFT-even form correct for FFT spectral analysis) builds the
    symmetric window of length ``n+1`` and drops the last sample; ``periodic=False`` is the
    plain symmetric window (zero — or near-zero — at both endpoints).
    """
    m = n + 1 if periodic else n
    fac = np.linspace(-np.pi, np.pi, m)
    w = np.zeros(m, dtype=np.float64)
    for k, a in enumerate(coeffs):
        w = w + a * np.cos(k * fac)
    return w[:-1] if periodic else w


def _tukey(n: int, alpha: float, periodic: bool) -> np.ndarray:
    """Tukey (tapered-cosine) window — ``alpha`` is the cosine-tapered fraction in [0, 1]."""
    if alpha <= 0:
        return np.ones(n, dtype=np.float64)
    if alpha >= 1:
        return _general_cosine(n, (0.5, 0.5), periodic)  # full cosine taper == Hann
    m = n + 1 if periodic else n
    idx = np.arange(0, m)
    width = int(np.floor(alpha * (m - 1) / 2.0))
    w = np.ones(m, dtype=np.float64)
    n1 = idx[: width + 1]
    n3 = idx[m - width - 1 :]
    w[: width + 1] = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0 * n1 / alpha / (m - 1))))
    w[m - width - 1 :] = 0.5 * (1 + np.cos(np.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (m - 1))))
    return w[:-1] if periodic else w


def _gaussian(n: int, std: float, periodic: bool) -> np.ndarray:
    """Gaussian window — ``std`` is the standard deviation in samples (must be > 0)."""
    if std <= 0:
        raise ValueError(f"gaussian window std must be > 0; got {std!r}")
    m = n + 1 if periodic else n
    k = np.arange(0, m) - (m - 1) / 2.0
    w = np.exp(-0.5 * (k / std) ** 2)
    return np.asarray(w[:-1] if periodic else w, dtype=np.float64)


def get_window(
    window: WindowName, n: int, *, window_param: Optional[float] = None, periodic: bool = True
) -> np.ndarray:
    """Build a length-``n`` window taper as a ``float64`` ndarray.

    Args:
        window: Which taper — one of ``WindowName`` (``boxcar`` is the rectangular identity).
        n: Number of taps (must be positive); normally the signal length being transformed.
        window_param: Shape parameter for the parametrized windows — Kaiser ``β`` (default 8.6),
            Tukey ``α`` taper fraction in [0, 1] (default 0.5), or Gaussian ``σ`` std in samples
            (required, no default). Ignored by the fixed windows.
        periodic: ``True`` (default) = DFT-even window (the correct form for FFT spectral
            analysis); ``False`` = symmetric window (zero at both endpoints).

    Returns:
        The window coefficients, ``float64``, shape ``(n,)``.

    Raises:
        ValueError: unknown ``window``, non-positive ``n``, or a Gaussian without ``window_param``.
    """
    if window not in WINDOW_NAMES:
        raise ValueError(f"unknown window {window!r}; valid: {WINDOW_NAMES}")
    if n <= 0:
        raise ValueError(f"window length n must be positive; got {n!r}")
    if n == 1:
        return np.ones(1, dtype=np.float64)
    if window == "boxcar":
        return np.ones(n, dtype=np.float64)
    if window in _COSINE_COEFFS:
        return _general_cosine(n, _COSINE_COEFFS[window], periodic)
    if window == "bartlett":
        m = n + 1 if periodic else n
        w = np.bartlett(m)
        return (w[:-1] if periodic else w).astype(np.float64)
    if window == "kaiser":
        beta = 8.6 if window_param is None else float(window_param)
        m = n + 1 if periodic else n
        w = np.kaiser(m, beta)
        return (w[:-1] if periodic else w).astype(np.float64)
    if window == "tukey":
        alpha = 0.5 if window_param is None else float(window_param)
        return _tukey(n, alpha, periodic)
    # window == "gaussian"
    if window_param is None:
        raise ValueError("gaussian window requires window_param (std in samples)")
    return _gaussian(n, float(window_param), periodic)


def window_sums(window: np.ndarray) -> Tuple[float, float]:
    """Return ``(S1, S2)`` = ``(Σw, Σw²)`` — the two sums the unit corrections need."""
    w = np.asarray(window, dtype=np.float64)
    return float(w.sum()), float(np.square(w).sum())


def coherent_gain(window: np.ndarray) -> float:
    """Coherent gain ``S1/N`` — the amplitude attenuation the window applies to a tone."""
    w = np.asarray(window, dtype=np.float64)
    return float(w.sum() / w.size)


def enbw_bins(window: np.ndarray) -> float:
    """Equivalent noise bandwidth ``N·S2/S1²`` in **bins** (e.g. ≈1.5 for Hann)."""
    s1, s2 = window_sums(window)
    return float(np.asarray(window).size * s2 / (s1 * s1))


def window_metadata(window_name: str, window: np.ndarray) -> Dict[str, object]:
    """Build the window-correction metadata dict (the keys ``SpectrumScalingOp`` reads)."""
    s1, s2 = window_sums(window)
    n = int(np.asarray(window).size)
    return {
        WINDOW_NAME_KEY: window_name,
        WINDOW_SIZE_KEY: n,
        WINDOW_SUM_KEY: s1,
        WINDOW_SUMSQ_KEY: s2,
        WINDOW_ENBW_KEY: float(n * s2 / (s1 * s1)),
        WINDOW_CG_KEY: float(s1 / n),
    }


def fold_one_sided(spectrum: np.ndarray, axis: int) -> np.ndarray:
    """Fold a two-sided spectrum (natural FFT order, DC at index 0) to one-sided.

    Keeps bins ``0 … N//2`` and doubles the interior bins (everything except DC and, for
    even ``N``, the Nyquist bin) so a real signal's one-sided amplitude/power reads its true
    value. Meaningful only for spectra of **real** inputs in natural (un-``fftshift``ed) order.
    """
    moved = np.swapaxes(np.asarray(spectrum), axis, -1)
    n = moved.shape[-1]
    out = moved[..., : n // 2 + 1].copy()
    if n % 2 == 0:
        out[..., 1:-1] = out[..., 1:-1] * 2  # exclude DC (0) and Nyquist (-1)
    else:
        out[..., 1:] = out[..., 1:] * 2  # no Nyquist bin for odd N
    return np.swapaxes(out, axis, -1)


def scale_spectrum(
    spectrum: np.ndarray,
    scaling: SpectrumScaling,
    *,
    s1: float,
    s2: float,
    sample_rate: Optional[float] = None,
    one_sided: bool = False,
    axis: int = -1,
) -> np.ndarray:
    """Scale a windowed FFT spectrum to the chosen units (assumes ``norm="backward"``).

    Args:
        spectrum: The complex FFT output (windowed, unscaled forward transform).
        scaling: ``none`` (complex, unchanged) · ``amplitude`` (V, complex, ``X/S1``) ·
            ``power`` (V², real, ``|X|²/S1²``) · ``density`` (V²/Hz, real, ``|X|²/(Fs·S2)``).
        s1: Window coherent-gain sum ``Σw`` (use ``N`` for an unwindowed / boxcar spectrum).
        s2: Window squared sum ``Σw²`` (use ``N`` for boxcar).
        sample_rate: ``Fs`` in Hz for ``density`` (V²/Hz). ``None`` / ≤0 → ``1.0`` (density
            per normalized frequency, V² per cycle/sample). Ignored by other modes.
        one_sided: Fold to a one-sided spectrum (real-input convention) after scaling.
        axis: Transform axis (for ``one_sided`` folding and the bin count).

    Returns:
        The scaled spectrum — complex for ``none``/``amplitude``, real for ``power``/``density``.
    """
    x = np.asarray(spectrum)
    if scaling == "none":
        out = x
    elif scaling == "amplitude":
        out = x / s1
    elif scaling == "power":
        out = np.square(np.abs(x)) / (s1 * s1)
    elif scaling == "density":
        fs = float(sample_rate) if (sample_rate is not None and sample_rate > 0) else 1.0
        out = np.square(np.abs(x)) / (fs * s2)
    else:
        raise ValueError(f"unknown scaling {scaling!r}; valid: {SPECTRUM_SCALINGS}")
    if one_sided:
        out = fold_one_sided(out, axis)
    return out
