import numpy as np
import astropy.units as u

from pathlib import Path
from typing import Optional, Tuple, Union
from astropy.constants import au
from support import import_data, create_distance_map, build_mask
from pathlib import Path
from typing import Optional, Union
from astropy.constants import au

_ALLOWED_KEYS = {
    "r_front", "r_back",
    "l_front", "l_back",
    "x_front", "x_back",
    "s_front", "s_back",
    "tau_front", "tau_back",
    "PR",
    "epsilon_map",
    "d_map",
}

def radial_position_ps(
    tb_path: Union[str, Path],
    pb_path: Union[str, Path],
    *,
    return_key: str = "r_front",
    data_type: Optional[str] = None,
    use_mask: bool = True,
    use_cdelt: bool = False,
    subtract_base_image: bool = False,
    base_tb_path: Optional[Union[str, Path]] = None,
    base_pb_path: Optional[Union[str, Path]] = None,
    dist_obs_to_source_km: float = au.to_value(u.km),
) -> np.ndarray:
    """
    TS-based PR localization that returns ONLY one requested output map.

    Valid return_key values:
      r_front, r_back, l_front, l_back, x_front, x_back,
      s_front, s_back, tau_front, tau_back, PR, epsilon_map, d_map
    """
    if return_key not in _ALLOWED_KEYS:
        raise ValueError(f"return_key must be one of {sorted(_ALLOWED_KEYS)}; got '{return_key}'")

    tb_path = Path(tb_path)
    pb_path = Path(pb_path)
    r_obs = float(dist_obs_to_source_km)

    # ---- Load images WITHOUT internal masking; apply one shared mask if requested ----
    tB, _ = import_data(
        tb_path,
        data_type=data_type,
        use_mask=False,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_file_path=base_tb_path,
    )
    pB, _ = import_data(
        pb_path,
        data_type=data_type,
        use_mask=False,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_file_path=base_pb_path,
    )

    tB = np.asarray(tB, dtype=float)
    pB = np.asarray(pB, dtype=float)

    if tB.shape != pB.shape:
        raise ValueError("tB and pB must have the same shape")

    if use_mask:
        mask = build_mask(str(pb_path))
        if mask.shape != tB.shape:
            raise ValueError("Mask shape does not match image shape.")
        tB = tB * mask
        pB = pB * mask

    # ---- Geometry ----
    rpos_map = create_distance_map(pb_path, data_type=data_type, use_cdelt=use_cdelt)  # km
    epsilon_map = np.arctan2(rpos_map, r_obs)  # rad
    d_map = r_obs * np.sin(epsilon_map)        # km

    # ---- PR (Eq 13) ----
    denom = (tB + pB)
    good = np.isfinite(tB) & np.isfinite(pB) & np.isfinite(denom) & (denom != 0.0)

    PR = np.full_like(tB, np.nan, dtype=float)
    PR[good] = (tB[good] - pB[good]) / denom[good]
    PR = np.clip(PR, 0.0, 1.0)

    # ---- tau solutions ----
    tau_front =  np.arcsin(np.sqrt(PR))
    tau_back  = -np.arcsin(np.sqrt(PR))

    # ---- s solutions (from TS) ----
    s_front = d_map * np.tan(tau_front)
    s_back  = d_map * np.tan(tau_back)

    # ---- LOS distance observer→feature ----
    l_front = r_obs * np.cos(epsilon_map) - s_front
    l_back  = r_obs * np.cos(epsilon_map) - s_back

    # ---- distance from POS ----
    x_front = d_map * np.sin(epsilon_map) + s_front * np.cos(epsilon_map)
    x_back  = d_map * np.sin(epsilon_map) + s_back  * np.cos(epsilon_map)

    # ---- heliocentric distance Sun→feature ----
    r_front = np.hypot(d_map, s_front)
    r_back  = np.hypot(d_map, s_back)

    # ---- Ensure invalid pixels are NaN everywhere (including derived products) ----
    invalid = ~good
    for arr in (tau_front, tau_back, s_front, s_back, l_front, l_back, x_front, x_back, r_front, r_back):
        arr[invalid] = np.nan

    # ---- Return only what was requested ----
    outputs = {
        "r_front": r_front,
        "r_back": r_back,
        "l_front": l_front,
        "l_back": l_back,
        "x_front": x_front,
        "x_back": x_back,
        "s_front": s_front,
        "s_back": s_back,
        "tau_front": tau_front,
        "tau_back": tau_back,
        "PR": PR,
        "epsilon_map": epsilon_map,
        "d_map": d_map,
    }

    return outputs[return_key]


def radial_position_scatter():
    return None