import warnings
import numpy as np
from support import import_data, create_distance_map
from astropy.constants import R_sun, au
import astropy.units as u


# Solar radius as a Quantity in km
R_SUN = R_sun.to_value(u.km)              # Quantity, e.g. <Quantity 695700. km>

# Observer distance as a Quantity in km
DEFAULT_R_OBS = au.to_value(u.km)         # <Quantity 149597870.7 km>


def radial_position_ps(file_list, 
                       data_type=None,
                       use_mask=True,
                       use_cdelt=False,
                       subtract_base_image=False,
                       base_file_list=None,
                       dist_obs_to_source_km=DEFAULT_R_OBS):
    """
    Polarization-ratio line-of-sight localization (point-source approximation).

    This routine takes total and polarized brightness in each image pixel and,
    under the point-source approximation for the Sun, inverts the polarization
    ratio to recover where along the line of sight the scattering feature could
    be located.

    There are two geometric solutions for each pixel:

    - The foreground (+) solution: feature between the Sun and the observer.
    - The background (–) solution: feature on the far side of the Sun,
      behind the plane of the sky, but still along the same line of sight.

    Distances and angles are defined as follows:

    - `r_plus`, `r_minus`  (Sun → feature):
        Heliocentric radial distance of the scattering point from the Sun
        for the foreground (+) and background (–) solutions.

    - `l_plus`, `l_minus`  (observer → feature):
        Line-of-sight distance from the observer to the scattering point
        along the ray corresponding to each pixel, for the + and – solutions.
        `l = 0` at the observer; `l ≈ dist_obs_to_source` near the Sun.

    - `x_plus`, `x_minus`  (distance from the plane of the sky):
        Signed distance of the scattering point from the plane of the sky
        (POS) along the Sun–observer axis:
            * x = 0   : point lies in the POS,
            * x > 0   : point in front of the POS (towards the observer),
            * x < 0   : point behind the POS (far side of the Sun).

    All distance outputs (`r_*`, `l_*`, `x_*`) are returned in the same units
    as `dist_image_plane` and `dist_obs_to_source` (e.g. km or R_sun). Angles
    (`epsilon`, `chi_*`, `xi_*`) are in radians.

    Parameters
    ----------
    tB : array_like
        Total white-light brightness B for each pixel.
    pB : array_like
        Polarized brightness pB for each pixel (same shape as tB).
    dist_image_plane : array_like
        Projected distance from Sun centre in the image plane for each pixel
        (impact parameter r_pos, same shape as tB), in the same units as
        `dist_obs_to_source` (e.g. km or R_sun).
    dist_obs_to_source : float
        Distance from observer to the Sun (e.g. 1 AU in km, or ~215 R_sun).

    Returns
    -------
    r_plus, r_minus : ndarray
        Heliocentric radial distance from the Sun to the scattering point for
        the foreground (+) and background (–) solutions.
    l_plus, l_minus : ndarray
        Line-of-sight distance from the observer to the scattering point for
        the foreground (+) and background (–) solutions.
    x_plus, x_minus : ndarray
        Signed distance from the plane of the sky along the Sun–observer axis
        for the + and – solutions.

    Notes
    -----
    - Inputs and outputs must be in consistent length units (km with km, or
      R_sun with R_sun).
    - Pixels with unphysical polarization (p < 0 or p > 1) or numerically
      invalid PR are returned as NaN in all outputs.
    """

    # ------------------------------------------------------------------
    # 1. Load data & distance map
    # ------------------------------------------------------------------
    tB, pB, tB_hdr, pB_hdr = import_data(
        file_list,
        data_type=data_type,
        use_mask=use_mask,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_file_list=base_file_list,
    )

    dist_image_plane = create_distance_map(
        file_list,
        data_type=data_type,
        use_mask=use_mask,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_file_list=base_file_list,
    )  # km

    # Cast and check shapes
    tB   = np.asarray(tB, dtype=float)
    pB   = np.asarray(pB, dtype=float)
    rpos = np.asarray(dist_image_plane, dtype=float)

    if tB.shape != pB.shape or tB.shape != rpos.shape:
        raise ValueError("tB, pB, and dist_image_plane must have the same shape")

    # Elongation epsilon = arctan(r_pos / R_obs)
    epsilon = np.arctan2(rpos, dist_obs_to_source_km)

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

    # Scattering angle chi from PR (point-source: PR = cos^2 chi)
    chi = np.full_like(tB, np.nan, dtype=float)
    chi[valid] = np.arccos(np.sqrt(PR[valid]))

    chi_plus  = chi
    chi_minus = np.pi - chi_plus

    # Angle xi between Sun–feature and observer–feature rays
    xi_plus  = epsilon - chi_plus  + 0.5 * np.pi
    xi_minus = epsilon - chi_minus + 0.5 * np.pi

    # Common denominator: sin(chi)
    denom = np.sin(chi_plus)

    # LOS distances (observer → feature)
    l_plus  = np.full_like(tB, np.nan, dtype=float)
    l_minus = np.full_like(tB, np.nan, dtype=float)

    good = valid & (denom != 0)

    l_plus[good] = (
        dist_obs_to_source_km
        * np.sin(0.5 * np.pi - xi_plus[good])
        / denom[good]
    )
    l_minus[good] = (
        dist_obs_to_source_km
        * np.sin(0.5 * np.pi - xi_minus[good])
        / denom[good]
    )

    # Radial distances (Sun → feature)
    r_plus  = np.full_like(tB, np.nan, dtype=float)
    r_minus = np.full_like(tB, np.nan, dtype=float)

    r_plus[good] = (
        dist_obs_to_source_km
        * np.sin(epsilon[good])
        / denom[good]
    )
    r_minus[good] = (
        dist_obs_to_source_km
        * np.sin(epsilon[good])
        / denom[good]
    )

    # Distance from plane of sky along Sun–observer axis: x = r sin(xi)
    x_plus  = np.full_like(tB, np.nan, dtype=float)
    x_minus = np.full_like(tB, np.nan, dtype=float)

    x_plus[good]  = r_plus[good]  * np.sin(xi_plus[good])
    x_minus[good] = r_minus[good] * np.sin(xi_minus[good])

    return r_plus, r_minus, l_plus, l_minus, x_plus, x_minus


import numpy as np






def radial_position_scatter(file_list, 
                       data_type=None,
                       use_mask=True,
                       use_cdelt=False,
                       subtract_base_image=False,
                       base_file_list=None,
                       dist_obs_to_source_km=DEFAULT_R_OBS,
                       scattering_fn=1.0):
    """
    Generalized polarization-ratio line-of-sight localization using a modified
    scattering inversion:
    
        chi = arcsin( scattering_fn * sqrt(1 - PR) )
    
    where PR = (1 - p) / (1 + p), p = pB/tB.
    
    This reduces to the standard Thomson point-source inversion when
    scattering_fn = 1.0.

    Parameters
    ----------
    tB : array_like
        Total brightness.
    pB : array_like
        Polarized brightness (same shape as tB).
    dist_image_plane : array_like
        Impact parameter (projected distance from Sun centre) per pixel, same
        shape as tB, in the same units as dist_obs_to_source.
    dist_obs_to_source : float
        Distance from observer to the Sun (1 AU in km, or 215 R_sun).
    scattering_fn : float, optional
        Scaling factor applied to sin(chi). Must satisfy:
            |scattering_fn * sqrt(1 - PR)| <= 1
        Default = 1 (Thomson scattering).

    Returns
    -------
    r_plus, r_minus : ndarray
        Radial distance Sun → feature for + and – solutions.
    l_plus, l_minus : ndarray
        LOS distance observer → feature for + and – solutions.
    x_plus, x_minus : ndarray
        Signed distance from plane of sky for + and – solutions.

    Notes
    -----
    - This function is only physically self-consistent when scattering_fn = 1.
    - For other values, inversion is mathematically valid but does NOT follow
      the real Thomson scattering kernel.
    """

    # ------------------------------------------------------------------
    # 1. Load data & distance map
    # ------------------------------------------------------------------
    tB, pB, tB_hdr, pB_hdr = import_data(
        file_list,
        data_type=data_type,
        use_mask=use_mask,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_file_list=base_file_list,
    )

    dist_image_plane = create_distance_map(
        file_list,
        data_type=data_type,
        use_mask=use_mask,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_file_list=base_file_list,
    )  # km

    # Cast inputs
    tB   = np.asarray(tB, dtype=float)
    pB   = np.asarray(pB, dtype=float)
    rpos = np.asarray(dist_image_plane, dtype=float)
    
    if tB.shape != pB.shape or tB.shape != rpos.shape:
        raise ValueError("tB, pB, and dist_image_plane must have the same shape")

    # Elongation angle
    epsilon = np.arctan2(rpos, dist_obs_to_source_km)

    # Fractional polarization p = pB/tB
    pol = np.zeros_like(tB)
    valid = (tB > 0) & np.isfinite(tB) & np.isfinite(pB)
    pol[valid] = pB[valid] / tB[valid]
    valid &= (pol >= 0.0) & (pol <= 1.0)

    # Polarization ratio PR = cos^2 chi (Thomson)
    PR = np.full_like(tB, np.nan)
    PR[valid] = (1.0 - pol[valid]) / (1.0 + pol[valid])
    valid &= (PR >= 0.0) & (PR <= 1.0)

    # ---- Modified χ inversion: chi = arcsin( scattering_fn * sqrt(1 - PR) ) ----
    arg = scattering_fn * np.sqrt(np.maximum(0.0, 1.0 - PR))
    good_chi = valid & (np.abs(arg) <= 1.0)

    chi_plus = np.full_like(tB, np.nan)
    chi_plus[good_chi] = np.arcsin(arg[good_chi])
    chi_minus = np.pi - chi_plus

    # Geometry angle xi
    xi_plus  = epsilon - chi_plus  + 0.5*np.pi
    xi_minus = epsilon - chi_minus + 0.5*np.pi

    # Denominator (sin chi)
    denom = np.sin(chi_plus)

    # Good pixels are those where inversion & denom are valid
    good = good_chi & (denom != 0)

    # LOS distance (observer → feature)
    l_plus  = np.full_like(tB, np.nan)
    l_minus = np.full_like(tB, np.nan)

    l_plus[good] = (
        dist_obs_to_source_km * np.sin(0.5*np.pi - xi_plus[good]) / denom[good]
    )
    l_minus[good] = (
        dist_obs_to_source_km * np.sin(0.5*np.pi - xi_minus[good]) / denom[good]
    )

    # Radial distance (Sun → feature)
    r_plus  = np.full_like(tB, np.nan)
    r_minus = np.full_like(tB, np.nan)

    r_plus[good] = (
        dist_obs_to_source_km * np.sin(epsilon[good]) / denom[good]
    )
    r_minus[good] = (
        dist_obs_to_source_km * np.sin(epsilon[good]) / denom[good]
    )

    # POS distance x = r sin(xi)
    x_plus  = np.full_like(tB, np.nan)
    x_minus = np.full_like(tB, np.nan)

    x_plus[good]  = r_plus[good]  * np.sin(xi_plus[good])
    x_minus[good] = r_minus[good] * np.sin(xi_minus[good])

    return r_plus, r_minus, l_plus, l_minus, x_plus, x_minus
