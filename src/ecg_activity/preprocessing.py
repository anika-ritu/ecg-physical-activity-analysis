"""ECG loading and filtering."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float,
    lowcut: float = 0.5,
    highcut: float = 50.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter."""
    nyquist = 0.5 * sampling_rate
    if not 0 < lowcut < highcut < nyquist:
        raise ValueError("Filter cutoffs must satisfy 0 < lowcut < highcut < Nyquist.")
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, np.asarray(signal, dtype=float))


def notch_filter(
    signal: np.ndarray,
    sampling_rate: float,
    notch_frequency: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Remove power-line interference with an IIR notch filter."""
    if not 0 < notch_frequency < 0.5 * sampling_rate:
        raise ValueError("Notch frequency must be below the Nyquist frequency.")
    b, a = iirnotch(notch_frequency, quality_factor, fs=sampling_rate)
    return filtfilt(b, a, np.asarray(signal, dtype=float))


def clean_ecg(
    signal: np.ndarray,
    sampling_rate: float = 1000.0,
    lowcut: float = 0.5,
    highcut: float = 50.0,
    notch_frequency: float = 50.0,
) -> np.ndarray:
    """Apply the bandpass and notch filters used in the study."""
    filtered = bandpass_filter(signal, sampling_rate, lowcut, highcut)
    return notch_filter(filtered, sampling_rate, notch_frequency)


def read_ecg_excel(path: str | Path, skiprows: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Read time and ECG amplitude from the first two numeric Excel columns."""
    path = Path(path)
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    frame = pd.read_excel(path, sheet_name=0, engine=engine, skiprows=skiprows)
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if numeric.shape[1] < 2:
        raise ValueError(f"{path.name} must contain time and ECG columns.")
    time = numeric.iloc[:, 0].to_numpy(dtype=float)
    signal = numeric.iloc[:, 1].to_numpy(dtype=float)
    valid = np.isfinite(time) & np.isfinite(signal)
    if valid.sum() < 20:
        raise ValueError(f"{path.name} does not contain enough valid ECG samples.")
    return time[valid], signal[valid]
