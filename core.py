import warnings
import numpy as np

import numpy as np

def radial_position_ps(tB, pB, dist_image_plane, dist_obs_to_source):
    """
    Polarization-ratio line-of-sight localization (point-source approximation).

    Parameters
    ----------
    tB : array_like
        Total white-light brightness B for each pixel.
    pB : array_like
        Polarized brightness pB for each pixel (same shape as tB).
    dist_image_plane : array_like
        Projected distance from Sun centre in the image plane for each pixel
        (impact parameter r_pos, same shape as tB), in the same units as
        dist_obs_to_source (e.g. R_sun or km).
    dist_obs_to_source : float
        Distance from observer to the Sun (e.g. 215 R_sun for 1 AU).

    Returns
    -------
    r_plus, r_minus : ndarray
        Heliocentric radial distances of the foreground (+) and background (-)
        solutions, same shape as tB.
    l_plus, l_minus : ndarray
        Line-of-sight distances from the observer to the feature (foreground
        and background).
    tau_plus, tau_minus : ndarray
        LOS angles from the Thomson surface for the + and - solutions.
    x_plus, x_minus : ndarray
        Distance from the plane of the sky along the Sun–observer axis for the
        + and - solutions (x = 0 at the POS; x > 0 in front of the POS).

    Invalid pixels are returned as np.nan.
    """

    # Cast and check shapes
    tB   = np.asarray(tB, dtype=float)
    pB   = np.asarray(pB, dtype=float)
    rpos = np.asarray(dist_image_plane, dtype=float)

    if tB.shape != pB.shape or tB.shape != rpos.shape:
        raise ValueError("tB, pB, and dist_image_plane must have the same shape")

    # Elongation epsilon = arctan(r_pos / R_obs)
    epsilon = np.arctan2(rpos, dist_obs_to_source)

    # Fractional polarization p = pB / B
    pol = np.zeros_like(tB, dtype=float)
    valid = (tB > 0) & np.isfinite(tB) & np.isfinite(pB)
    pol[valid] = pB[valid] / tB[valid]

    # Require 0 <= p <= 1 and mark others invalid
    valid &= (pol >= 0.0) & (pol <= 1.0)

    # Polarization ratio PR = (1 - p) / (1 + p)
    PR = np.full_like(tB, np.nan, dtype=float)
    PR[valid] = (1.0 - pol[valid]) / (1.0 + pol[valid])

    # Guard against numerical overshoots
    valid &= (PR >= 0.0) & (PR <= 1.0)
    PR[~valid] = np.nan

    # Scattering angle chi from PR (point-source: PR = sin^2 chi)
    chi_plus  = np.full_like(tB, np.nan, dtype=float)
    chi_plus[valid] = np.arccos(np.sqrt(PR[valid]))
    chi_minus = np.pi - chi_plus

    # Angle xi between Sun–feature and observer–feature rays
    xi_plus  = epsilon - chi_plus  + 0.5 * np.pi
    xi_minus = epsilon - chi_minus + 0.5 * np.pi

    # LOS distances (observer → feature)
    l_plus  = np.full_like(tB, np.nan, dtype=float)
    l_minus = np.full_like(tB, np.nan, dtype=float)

    denom_plus  = np.sin(np.pi - chi_plus)   # = sin(chi_plus)
    denom_minus = np.sin(np.pi - chi_minus)  # = sin(chi_minus)

    good_plus  = valid & (denom_plus  != 0)
    good_minus = valid & (denom_minus != 0)

    l_plus[good_plus] = (
        dist_obs_to_source
        * np.sin(0.5 * np.pi - xi_plus[good_plus])
        / denom_plus[good_plus]
    )
    l_minus[good_minus] = (
        dist_obs_to_source
        * np.sin(0.5 * np.pi - xi_minus[good_minus])
        / denom_minus[good_minus]
    )

    # Radial distances (Sun → feature)
    r_plus  = np.full_like(tB, np.nan, dtype=float)
    r_minus = np.full_like(tB, np.nan, dtype=float)

    r_plus[good_plus] = (
        dist_obs_to_source
        * np.sin(epsilon[good_plus])
        / denom_plus[good_plus]
    )
    r_minus[good_minus] = (
        dist_obs_to_source
        * np.sin(epsilon[good_minus])
        / denom_minus[good_minus]
    )

    # LOS angle from TS: tau = xi - epsilon
    tau_plus  = xi_plus  - epsilon
    tau_minus = xi_minus - epsilon

    # Distance from plane of sky along Sun–observer axis: x = r sin(xi)
    x_plus  = np.full_like(tB, np.nan, dtype=float)
    x_minus = np.full_like(tB, np.nan, dtype=float)

    x_plus[good_plus]  = r_plus[good_plus]  * np.sin(xi_plus[good_plus])
    x_minus[good_minus] = r_minus[good_minus] * np.sin(xi_minus[good_minus])

    return r_plus, r_minus, l_plus, l_minus, tau_plus, tau_minus, x_plus, x_minus
