"""ECG, HRV, and waveform feature extraction."""

import numpy as np
from scipy.signal import find_peaks, welch


SELECTED_ECG_FEATURES = [
    "std_amplitude",
    "rms_amplitude",
    "amplitude_difference_r_q",
    "amplitude_difference_r_s",
    "r_peak_mean_amplitude",
    "rmssd_ms",
    "pnn50_percent",
    "lf_hf_ratio",
]


def _wave_points(signal: np.ndarray, r_peaks: np.ndarray, sampling_rate: float) -> dict[str, np.ndarray]:
    radius = max(1, int(0.1 * sampling_rate))
    points: dict[str, list[int]] = {"p": [], "q": [], "r": [], "s": [], "t": []}
    for r_idx in r_peaks:
        q_start = max(0, r_idx - radius)
        s_end = min(len(signal), r_idx + radius)
        q_idx = q_start + int(np.argmin(signal[q_start:r_idx])) if r_idx > q_start else r_idx
        s_idx = r_idx + int(np.argmin(signal[r_idx:s_end])) if s_end > r_idx else r_idx

        p_start = max(0, q_idx - radius)
        t_end = min(len(signal), s_idx + radius)
        p_idx = p_start + int(np.argmax(signal[p_start:q_idx])) if q_idx > p_start else q_idx
        t_idx = s_idx + int(np.argmax(signal[s_idx:t_end])) if t_end > s_idx else s_idx

        for name, index in (("p", p_idx), ("q", q_idx), ("r", r_idx), ("s", s_idx), ("t", t_idx)):
            points[name].append(index)
    return {name: np.asarray(indices, dtype=int) for name, indices in points.items()}


def _band_power(frequency: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (frequency >= low) & (frequency < high)
    return float(np.trapz(power[mask], frequency[mask])) if mask.sum() > 1 else 0.0


def extract_ecg_features(
    time: np.ndarray,
    signal: np.ndarray,
    sampling_rate: float = 1000.0,
) -> dict[str, float]:
    """Extract the ECG descriptors used by the analysis pipeline."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time) != len(signal) or len(signal) < sampling_rate:
        raise ValueError("Time and ECG arrays must have equal length and at least one second of data.")

    threshold = float(np.mean(signal) + 0.2 * np.std(signal))
    r_peaks, _ = find_peaks(signal, height=threshold, distance=int(0.5 * sampling_rate))
    if len(r_peaks) < 3:
        raise ValueError("At least three R-peaks are required for feature extraction.")

    rr_seconds = np.diff(time[r_peaks])
    rr_seconds = rr_seconds[np.isfinite(rr_seconds) & (rr_seconds > 0)]
    rr_differences = np.diff(rr_seconds)
    if len(rr_seconds) < 2:
        raise ValueError("Insufficient valid R-R intervals.")

    frequency, power = welch(
        rr_seconds,
        fs=1.0 / np.mean(rr_seconds),
        nperseg=min(len(rr_seconds), 256),
    )
    lf_power = _band_power(frequency, power, 0.04, 0.15)
    hf_power = _band_power(frequency, power, 0.15, 0.40)

    points = _wave_points(signal, r_peaks, sampling_rate)
    p, q, r, s, t = (points[name] for name in ("p", "q", "r", "s", "t"))
    rmssd_ms = float(np.sqrt(np.mean(rr_differences**2)) * 1000) if len(rr_differences) else 0.0
    nn50 = int(np.sum(np.abs(rr_differences) > 0.05))

    return {
        "heart_rate_bpm": float(60.0 / np.mean(rr_seconds)),
        "mean_rr_seconds": float(np.mean(rr_seconds)),
        "std_rr_seconds": float(np.std(rr_seconds)),
        "rmssd_ms": rmssd_ms,
        "nn50_count": float(nn50),
        "pnn50_percent": float(100.0 * nn50 / len(rr_differences)) if len(rr_differences) else 0.0,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": float(lf_power / hf_power) if hf_power > 0 else np.nan,
        "mean_amplitude": float(np.mean(signal)),
        "std_amplitude": float(np.std(signal)),
        "rms_amplitude": float(np.sqrt(np.mean(signal**2))),
        "r_peak_mean_amplitude": float(np.mean(signal[r])),
        "r_peak_std_amplitude": float(np.std(signal[r])),
        "amplitude_difference_r_q": float(np.mean(signal[r] - signal[q])),
        "amplitude_difference_r_s": float(np.mean(signal[r] - signal[s])),
        "mean_qt_interval_ms": float(np.mean((t - p) / sampling_rate) * 1000),
        "mean_pr_interval_ms": float(np.mean((r - p) / sampling_rate) * 1000),
        "mean_st_elevation": float(np.mean(signal[s] - signal[p])),
        "t_wave_mean_amplitude": float(np.mean(signal[t])),
        "t_wave_std_amplitude": float(np.std(signal[t])),
        "qrs_duration_ms": float(np.mean((s - q) / sampling_rate) * 1000),
    }
