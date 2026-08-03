"""Signal post-processing for result time-histories.

SAE J211 CFC filters, a general Butterworth, cumulative integrate/differentiate,
and resampling — for channels read from a ``Binout`` / ``D3plot``.

    from dynars import signal

    ax = signal.cfc(raw, 1000.0, dt)     # CFC1000
    vel = signal.integrate(ax, dt)
    low = signal.butterworth(ax, 4, 300.0, 1 / dt, "low")
"""

from dynars._dynars import (
    butterworth,
    cfc,
    decimate,
    differentiate,
    filtfilt,
    integrate,
    resample_linear,
)

__all__ = [
    "butterworth",
    "cfc",
    "decimate",
    "differentiate",
    "filtfilt",
    "integrate",
    "resample_linear",
]
