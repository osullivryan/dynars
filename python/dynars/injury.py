"""Occupant injury criteria.

Head (HIC15 / HIC36 / HIC), neck (Nij, NIC), chest (VC, 3 ms clip), brain
(BrIC / UBrIC), tibia index, and the Gadd severity index — computed from
acceleration / force time-histories.

    from dynars import injury

    a_res = injury.resultant(ax, ay, az)   # sqrt(x^2 + y^2 + z^2)
    hic = injury.hic36(a_res, dt)          # also hic15, hic
    a3ms = injury.clip(a_res, dt)          # 3 ms clip
    csi = injury.severity_index(a_res, dt) # Gadd severity index
"""

from dynars._dynars import (
    bric,
    clip,
    hic,
    hic15,
    hic36,
    nic,
    nij,
    resultant,
    severity_index,
    tibia_index,
    ubric,
    vc,
)

__all__ = [
    "bric",
    "clip",
    "hic",
    "hic15",
    "hic36",
    "nic",
    "nij",
    "resultant",
    "severity_index",
    "tibia_index",
    "ubric",
    "vc",
]
