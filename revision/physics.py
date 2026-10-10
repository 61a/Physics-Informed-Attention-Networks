"""NumPy reference for GHz/ms cold-plasma alignment of intensity spectra."""
import numpy as np


def delays_in_samples(frequencies_ghz, dm_values, sampling_ms):
    frequencies = np.asarray(frequencies_ghz, dtype=np.float64)
    dm = np.asarray(dm_values, dtype=np.float64)
    if frequencies.ndim != 1 or len(frequencies) < 2 or not np.isfinite(frequencies).all() or (frequencies <= 0).any():
        raise ValueError('Supply a finite positive one-dimensional frequency vector in GHz.')
    if len(np.unique(frequencies)) != len(frequencies):
        raise ValueError('Channel frequencies must be distinct.')
    if dm.ndim != 1 or len(dm) < 1 or not np.isfinite(dm).all():
        raise ValueError('Supply a finite one-dimensional DM grid.')
    if sampling_ms is None or not np.isfinite(sampling_ms) or sampling_ms <= 0:
        raise ValueError('sampling_ms must be a measured positive interval in ms.')
    return 4.149 * dm[:, None] * (frequencies[None, :]**-2 - frequencies.max()**-2) / sampling_ms


def align_numpy(x, delays):
    x = np.asarray(x)
    if x.ndim != 4 or not np.isfinite(x).all():
        raise ValueError('Input must be a finite (batch,channel,time,frequency) array.')
    delays = np.asarray(delays, dtype=np.float64)
    if delays.ndim != 2 or delays.shape[1] != x.shape[-1] or not np.isfinite(delays).all():
        raise ValueError('Delay bank must have shape (hypothesis,frequency).')
    time = x.shape[-2]
    maximum = float(np.abs(delays).max())
    if maximum >= time:
        raise ValueError('Trial displacement exceeds window support; supply a longer window or residual-DM grid.')
    guard = int(np.ceil(maximum)) + time
    n = time + 2*guard
    padded = np.pad(x, ((0,0),(0,0),(guard,guard),(0,0)))
    spectrum = np.fft.rfft(padded, axis=2)
    cycles = np.fft.rfftfreq(n)
    bank = []
    for delay in delays:
        phase = np.exp(2j*np.pi*cycles[:,None]*delay[None,:])
        shifted = np.fft.irfft(spectrum*phase[None,None,:,:], n=n, axis=2)
        bank.append(shifted[:,:,guard:guard+time,:])
    return np.stack(bank,axis=1)
