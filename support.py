import numpy as np
import matplotlib.pyplot as plt
import warnings
import cv2
from pathlib import Path
from astropy.io import fits
from typing import Iterable, Tuple, Optional, Union
from astropy import units as u, constants as c
from astropy.constants import R_sun, au

def replace_value_with_nan_inplace(arr, bad_value, out_value=np.nan):
    """
    Replace values in the given array *in place*.
    """
    mask = (arr == bad_value)
    arr[mask] = out_value
    return arr


def _read_fits_image(path: str) -> Tuple[np.ndarray, fits.Header]:
    """Read primary HDU image + header from a FITS file."""
    with fits.open(path, memmap=False) as hdul:
        # Prefer first IMAGE-like HDU if primary has no data
        hdu = hdul[0]
        if hdu.data is None:
            # find first extension with image data
            img_hdus = [h for h in hdul[1:] if getattr(h, "data", None) is not None]
            if not img_hdus:
                raise ValueError(f"No image data found in FITS file: {path}")
            hdu = img_hdus[0]
        return np.array(hdu.data), hdu.header



def build_mask(
    file_path: str,
    *,
    inner_radius_solar: Optional[float] = None,
    inner_radius_pix: Optional[float] = None,
    outer_radius_pix: Optional[float] = None,
) -> np.ndarray:
    """
    Build a circular binary mask (1=keep, 0=mask) for coronagraph images.

    By default, the inner occulted radius is chosen from instrument/detector
    lookups (R_i in solar radii), scaled by the current image size relative
    to the instrument's default native size. The outer radius defaults to a
    conservative circle that stays within the frame.

    Overrides:
      - `inner_radius_solar`: set inner radius in solar radii (overrides lookup).
      - `inner_radius_pix`:   set inner radius directly in pixels (mutually exclusive with `inner_radius_solar`).
      - `outer_radius_pix`:   set outer radius directly in pixels.

    Instrument-specific defaults (R_i in solar radii; default native size):
      - SOHO/LASCO C2:  R_i = 2.5, default_size = 1024
      - SOHO/LASCO C3:  R_i = 4.0, default_size = 1024
      - COSMO K-Coronagraph: R_i = 1.15, default_size = 1024
      - STEREO/SECCHI COR1: R_i = 1.57, default_size = 2048
      - STEREO/SECCHI COR2: R_i = 3.0,  default_size = 2048

    Solar-radius-to-pixels conversion uses:
        R_sun_pix = 960.0 / CDELT1
    assuming CDELT1 is in arcsec/pixel.

    Parameters
    ----------
    file_path : str
        Path to the FITS file.
    inner_radius_solar : float, optional
        Override inner radius in solar radii (mutually exclusive with `inner_radius_pix`).
    inner_radius_pix : float, optional
        Override inner radius in pixels (mutually exclusive with `inner_radius_solar`).
    outer_radius_pix : float, optional
        Override outer radius in pixels.

    Returns
    -------
    mask : ndarray, shape (ny, nx), dtype=float
        Binary mask where 1 indicates valid pixels and 0 indicates masked pixels.

    Notes
    -----
    - Requires header keys: INSTRUME, CRPIX1, CRPIX2, NAXIS1, NAXIS2, CDELT1.
    - FITS CRPIX are 1-based; this function converts them to 0-based by subtracting 1
    - If `inner_radius_solar` and `inner_radius_pix` are both provided, a ValueError is raised.

    Examples
    --------
    >>> # Default behavior from headers (instrument lookup + safe outer radius)
    >>> mask = build_mask("LASCO_C2_example.fits")

    >>> # Force inner radius to 2.8 solar radii, keep default outer radius
    >>> mask = build_mask("LASCO_C2_example.fits", inner_radius_solar=2.8)

    >>> # Force inner radius to 500 px and outer radius to 980 px
    >>> mask = build_mask("LASCO_C2_example.fits",
    ...                   inner_radius_pix=500,
    ...                   outer_radius_pix=980)
    """
    # --- Read header + image once ---
    with fits.open(file_path, memmap=False) as hdul:
        hdu = hdul[0]
        if hdu.data is None:
            for ext in hdul[1:]:
                if getattr(ext, "data", None) is not None:
                    hdu = ext
                    break
            if hdu.data is None:
                raise ValueError(f"No image data found in FITS file: {file_path}")

        hdr = hdu.header
        img = np.asarray(hdu.data)

    # --- Required header keys ---
    required = ["INSTRUME", "CRPIX1", "CRPIX2", "NAXIS1", "NAXIS2", "CDELT1"]
    missing = [k for k in required if k not in hdr]
    if missing:
        raise ValueError(f"Missing required FITS header keys: {missing} in {file_path}")

    instrume = str(hdr["INSTRUME"]).strip().upper()
    detector = str(hdr.get("DETECTOR", "")).strip().upper()
    cx = float(hdr["CRPIX1"]) -1.0 # 1-based; keep as-is
    cy = float(hdr["CRPIX2"]) -1.0
    nx = int(hdr["NAXIS1"])
    ny = int(hdr["NAXIS2"])
    cdelt1 = float(hdr["CDELT1"])  # arcsec / pixel

    # --- Mutually exclusive inner-radius overrides ---
    if inner_radius_solar is not None and inner_radius_pix is not None:
        raise ValueError(
            "Provide only one of inner_radius_solar or inner_radius_pix, not both."
        )

    # --- Instrument lookup for defaults ---
    R_i = None
    default_image_size = None

    if instrume == "LASCO":
        default_image_size = 1024
        if "C2" in detector:
            R_i = 2.5
        elif "C3" in detector:
            R_i = 4.0
    elif instrume == "COSMO K-CORONAGRAPH":
        default_image_size = 1024
        R_i = 1.15
    elif instrume == "SECCHI":
        default_image_size = 2048
        if "COR1" in detector:
            R_i = 1.57
        elif "COR2" in detector:
            R_i = 3.0

    if inner_radius_solar is None and inner_radius_pix is None:
        if R_i is None or default_image_size is None:
            raise ValueError(
                f"Unrecognized instrument/detector: INSTRUME='{instrume}', DETECTOR='{detector}'. "
                "Either extend the lookup or pass an inner radius override."
            )

    # --- Pixel scale / solar radius in pixels ---
    if cdelt1 == 0:
        raise ValueError("CDELT1 is zero; cannot compute solar radius in pixels.")
    R_sun_pix = 960.0 / cdelt1

    # --- Compute inner radius in pixels ---
    if inner_radius_pix is not None:
        inner_pix = float(inner_radius_pix)
    else:
        # choose solar radii value (override or lookup), then scale to pixels
        Ri_solar = float(inner_radius_solar) if inner_radius_solar is not None else float(R_i)
        scale_factor = float(nx) / float(default_image_size if default_image_size else nx)
        inner_pix = Ri_solar * R_sun_pix * scale_factor

    # --- Coordinate grid (0-based indices), compare to 1-based CRPIX ---
    yy, xx = np.ogrid[0:ny, 0:nx]
    r2_pix = (xx - cx) ** 2 + (yy - cy) ** 2

    # --- Outer radius: default conservative value inside frame unless overridden ---
    if outer_radius_pix is not None:
        outer_pix = float(outer_radius_pix)
    else:
        dx_left = cx
        dx_right = (nx - 1) - cx
        dy_top = cy
        dy_bottom = (ny - 1) - cy
        outer_pix = float(np.nanmin([dx_left, dx_right, dy_top, dy_bottom]))
        if outer_pix <= 0:
            outer_pix = float(min(nx, ny)) / 2.0
            warnings.warn(
                "CRPIX near/outside frame; using fallback outer radius.",
                RuntimeWarning,
            )

    # --- Compose mask: 1 inside [inner_pix, outer_pix], else 0 ---
    mask = np.ones((ny, nx), dtype=float)
    mask[r2_pix < inner_pix**2] = 0.0
    mask[r2_pix > outer_pix**2] = 0.0

    return mask






def import_data(
    file_path: Union[str, Path],
    data_type: Optional[str] = None,
    use_mask: bool = True,
    use_cdelt: bool = False,
    subtract_base_image: bool = False,
    base_file_path: Optional[Union[str, Path]] = None,
    bad_pixel_value: Optional[int] = -8888,
) -> Tuple[np.ndarray, fits.Header]:
    """
    Load a single FITS image, optionally mask it, optionally replace bad pixels,
    and optionally base-difference it.

    Parameters
    ----------
    file_path : str or Path
        FITS file path.
    data_type : {'forward', None}, optional
        If 'forward', replace pixels equal to `bad_pixel_value` with NaN.
    use_mask : bool, default True
        If True, multiply the image by a mask derived from `build_mask(file_path)`.
    use_cdelt : bool, default False
        Placeholder for future pixel-scale handling. Currently **not used**;
        a warning is issued if True.
    subtract_base_image : bool, default False
        If True, subtract the base image before returning.
    base_file_path : str or Path, optional
        FITS file path of the base image. Required if subtract_base_image=True.
    bad_pixel_value : int or None, optional
        Value used to mark bad pixels (default -8888). If None, no replacement.

    Returns
    -------
    data : ndarray
        Image data (possibly masked and/or base-differenced).
    header : astropy.io.fits.Header
        FITS header associated with the returned image.
    """
    file_path = Path(file_path)

    if use_cdelt:
        warnings.warn(
            "'use_cdelt' is currently not implemented and is ignored.",
            RuntimeWarning,
            stacklevel=2,
        )

    data, header = _read_fits_image(str(file_path))

    # Optional mask
    mask = None
    if use_mask:
        try:
            mask = build_mask(str(file_path))
            if mask.shape != data.shape:
                raise ValueError("Mask shape does not match image shape.")
            data = data * mask
        except NameError:
            warnings.warn("build_mask is not defined; skipping masking.",
                          RuntimeWarning, stacklevel=2)

    # Optional bad-pixel replacement
    if data_type == "forward" and bad_pixel_value is not None:
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(float, copy=False)
        data[data == bad_pixel_value] = np.nan
        data=replace_value_with_nan_inplace(data, -9999, out_value=np.nan)
        data=replace_value_with_nan_inplace(data, -8888, out_value=np.nan)


    # Optional base differencing
    if subtract_base_image:
        if base_file_path is None:
            warnings.warn(
                "subtract_base_image=True but no base_file_path provided; skipping base subtraction.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            base_file_path = Path(base_file_path)
            base_data, _ = _read_fits_image(str(base_file_path))

            if use_mask and mask is not None:
                base_data = base_data * mask

            if base_data.shape != data.shape:
                raise ValueError("Base image shape does not match image shape.")

            data = cv2.GaussianBlur(data - base_data, (7,7),0)

    return data, header




def create_distance_map(
    file_path: Union[str, Path],
    data_type: Optional[str] = None,
    use_cdelt: bool = False,
) -> np.ndarray:
    """
    Create a 2D map of projected image-plane distances r_pos (km) from Sun centre
    for each pixel, using ONLY the specified FITS file (tB/pB agnostic).

    Parameters
    ----------
    file_path : str or Path
        Path to the FITS file (geometry taken from its header).
    data_type : str, optional
        If "noise_gate_data", interpret CDELT as R_sun per pixel.
    use_cdelt : bool
        If True, use CRPIX/CDELT/RSUN in header. If False, assume 64 R_sun FOV.

    Returns
    -------
    rpos_km : ndarray
        2D array (ny, nx) of r_pos in km.
    """
    file_path = Path(file_path)
    data, hdr = _read_fits_image(str(file_path))

    solar_radii_in_km = c.R_sun.to(u.kilometer).value
    ny, nx = data.shape

    if use_cdelt:
        x_center = hdr["CRPIX1"] - 1.0
        y_center = hdr["CRPIX2"] - 1.0

        if data_type == "noise_gate_data":
            x_pix_km = solar_radii_in_km * hdr["CDELT1"]
            y_pix_km = solar_radii_in_km * hdr["CDELT2"]
        else:
            x_pix_km = solar_radii_in_km * hdr["CDELT1"] / hdr["RSUN"]
            y_pix_km = solar_radii_in_km * hdr["CDELT2"] / hdr["RSUN"]
    else:
        x_center = (nx - 1) / 2.0
        y_center = (ny - 1) / 2.0

        fov_solar_radii = 64.0
        x_pix_km = fov_solar_radii * solar_radii_in_km / nx
        y_pix_km = fov_solar_radii * solar_radii_in_km / ny

    yy, xx = np.indices((ny, nx))
    dx = (xx - x_center) * x_pix_km
    dy = (yy - y_center) * y_pix_km

    return np.hypot(dx, dy)




def create_epsilon_map(
    file_path: Union[str, Path],
    data_type: Optional[str] = None,
    use_cdelt: bool = False,
    robs_km: float = au.to_value(u.km),
    degrees: bool = False,
) -> np.ndarray:
    """
    Create a 2D map of elongation angle epsilon for each pixel (tB/pB agnostic).

    Computes:
        epsilon = arctan(r_pos / r_obs)
    where r_pos is the projected distance from Sun center in the image plane (km)
    and r_obs is the Sun–observer distance (km).

    Parameters
    ----------
    file_path : str or Path
        Path to the FITS file (geometry taken from its header).
    data_type : str, optional
        Passed to create_distance_map (used for e.g. "noise_gate_data").
    use_cdelt : bool
        Passed to create_distance_map.
    robs_km : float
        Sun–observer distance in km (default 1 AU).
    degrees : bool
        If True, return epsilon in degrees. Otherwise radians.

    Returns
    -------
    epsilon_map : ndarray
        2D array (ny, nx) of elongation angle epsilon.
    """
    file_path = Path(file_path)

    # r_pos map (km) from the same file's geometry
    rpos_km = create_distance_map(
        file_path=file_path,
        data_type=data_type,
        use_cdelt=use_cdelt,
    )

    # epsilon in radians (use arctan2 for numerical stability)
    eps = np.arctan2(rpos_km, robs_km)

    if degrees:
        eps = np.degrees(eps)

    return eps









