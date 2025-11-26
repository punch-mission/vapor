import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy import signal
from astropy.io import fits
from typing import Iterable, Tuple, Optional
from astropy import constants as c
from astropy import units as u

def gauss_kern(size: int, size_y: int | None = None) -> np.ndarray:
    """
    Generate a normalized 2D Gaussian kernel for convolution.

    Parameters
    ----------
    size : int
        The half-size of the kernel in the x-direction. The full kernel
        will span from -size to +size.
    size_y : int, optional
        The half-size in the y-direction. If None, it defaults to `size`.

    Returns
    -------
    numpy.ndarray
        A 2D Gaussian kernel normalized so that its sum equals 1.

    Notes
    -----
    The kernel uses a Gaussian of the form:
        exp(-(x^2 / size^2 + y^2 / size_y^2))
    where `size` and `size_y` act as scale parameters, not standard deviations.
    """
    size = int(size)
    size_y = int(size_y) if size_y is not None else size

    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2 / float(size)**2 + y**2 / float(size_y)**2))

    return g / g.sum()


def blur_image(image: np.ndarray, n: int, n_y: int | None = None) -> np.ndarray:
    """
    Blur an image by convolving it with a 2D Gaussian kernel.

    Parameters
    ----------
    image : numpy.ndarray
        The 2D input image array to be blurred.
    n : int
        The Gaussian kernel half-size in the x-direction.
    n_y : int, optional
        The Gaussian kernel half-size in the y-direction. If None, it defaults to `n`.

    Returns
    -------
    numpy.ndarray
        The blurred image, computed using `scipy.signal.convolve`
        with `mode='valid'`.

    Notes
    -----
    The output will be smaller than the input by `n` (and `n_y`) pixels
    on each side because `mode='valid'` is used.
    """
    kernel = gauss_kern(n, size_y=n_y)
    blurred = signal.convolve(image, kernel, mode="valid")
    return blurred


def open_fits(filepath):
    """
    Load a FITS file and display the primary HDU image using matplotlib.
    """
    # Open FITS file
    with fits.open(filepath) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    
    if data is None:
        raise ValueError("No image data found in primary HDU.")
    
    return data


def generate_square_test_pattern(size=256, square_size=32):
    """
    Generate a 2D test array containing non-overlapping squares of increasing values.

    Parameters
    ----------
    size : int
        Size of the full image (size x size).
    square_size : int
        Size of each square.

    Returns
    -------
    numpy.ndarray
        A 2D array with square patterns.
    """
    data = np.zeros((size, size), dtype=float)
    value = 1

    # Number of squares along each dimension
    n = size // square_size

    for i in range(n):
        for j in range(n):
            y0 = i * square_size
            y1 = y0 + square_size
            x0 = j * square_size
            x1 = x0 + square_size
            data[y0:y1, x0:x1] = value
            value += 1

    return data


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


def import_data(
    file_list: Iterable[str],
    data_type: Optional[str] = None,
    use_mask: bool = True,
    use_cdelt: bool = True,
    subtract_base_image: bool = False,
    base_file_list: Optional[Iterable[str]] = None,
    bad_pixel_value: Optional[int] = -8888,
) -> Tuple[np.ndarray, np.ndarray, fits.Header, fits.Header]:
    """
    Load, optionally mask, and optionally base-difference a total-brightness (tB)
    and polarized-brightness (pB) FITS pair.

    Parameters
    ----------
    file_list : iterable of str
        List/tuple of exactly two FITS file paths:
        [tB_path, pB_path].
    data_type : {'forward', None}, optional
        If 'forward', replace pixels equal to `bad_pixel_value` with NaN.
    use_mask : bool, default True
        If True, multiply each image by a mask derived from its file via
        `build_mask(path)`. Assumes `build_mask` returns a numeric mask with
        the same shape as the image.
    use_cdelt : bool, default True
        Placeholder for future pixel-scale handling. Currently **not used**;
        a warning is issued if True.
    subtract_base_image : bool, default False
        If True, subtract base images (tB_base, pB_base) before returning.
    base_file_list : iterable of str, optional
        List/tuple of exactly two FITS file paths for base images:
        [tB_base_path, pB_base_path]. Required if `subtract_base_image=True`.
    bad_pixel_value : int or None, optional
        Value used to mark bad pixels (default -8888). If `None`, no replacement
        is performed even if `data_type='forward'`.

    Returns
    -------
    tB_data : ndarray
        Total brightness image (possibly masked and/or base-differenced).
    pB_data : ndarray
        Polarized brightness image (possibly masked and/or base-differenced).
    tB_header : astropy.io.fits.Header
        Header associated with the returned tB image.
    pB_header : astropy.io.fits.Header
        Header associated with the returned pB image.
    """
    file_list = list(file_list)
    if len(file_list) != 2:
        raise ValueError("file_list must contain exactly two paths: [tB_path, pB_path].")

    if use_cdelt:
        warnings.warn(
            "'use_cdelt' is currently not implemented and is ignored.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Load primary images
    tB_data, tB_header = _read_fits_image(file_list[0])
    pB_data, pB_header = _read_fits_image(file_list[1])

    # Optional masks
    if use_mask:
        try:
            tB_mask = build_mask(file_list[0])
            pB_mask = build_mask(file_list[1])
            if tB_mask.shape != tB_data.shape:
                raise ValueError("tB mask shape does not match tB image shape.")
            if pB_mask.shape != pB_data.shape:
                raise ValueError("pB mask shape does not match pB image shape.")
            tB_data = tB_data * tB_mask
            pB_data = pB_data * pB_mask
        except NameError:
            warnings.warn("build_mask is not defined; skipping masking.",
                          RuntimeWarning, stacklevel=2)

    # Optional bad-pixel replacement
    if data_type == "forward" and bad_pixel_value is not None:
        if not np.issubdtype(tB_data.dtype, np.floating):
            tB_data = tB_data.astype(float, copy=False)
        if not np.issubdtype(pB_data.dtype, np.floating):
            pB_data = pB_data.astype(float, copy=False)

        tB_data[tB_data == bad_pixel_value] = np.nan
        pB_data[pB_data == bad_pixel_value] = np.nan

    # Optional base differencing
    if subtract_base_image:
        if base_file_list is None:
            warnings.warn(
                "subtract_base_image=True but no base_file_list provided; skipping base subtraction.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            base_file_list = list(base_file_list)
            if len(base_file_list) != 2:
                raise ValueError("base_file_list must contain exactly two paths.")

            base_tB, _ = _read_fits_image(base_file_list[0])
            base_pB, _ = _read_fits_image(base_file_list[1])

            if use_mask:
                try:
                    base_tB = base_tB * tB_mask
                    base_pB = base_pB * pB_mask
                except NameError:
                    pass

            if base_tB.shape != tB_data.shape:
                raise ValueError("Base tB image shape does not match tB image shape.")
            if base_pB.shape != pB_data.shape:
                raise ValueError("Base pB image shape does not match pB image shape.")

            tB_data = tB_data - base_tB
            pB_data = pB_data - base_pB

    return tB_data, pB_data, tB_header, pB_header



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
    - FITS CRPIX are 1-based; this function uses them as-is to match typical solar conventions.
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
    cx = float(hdr["CRPIX1"])  # 1-based; keep as-is
    cy = float(hdr["CRPIX2"])
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

def create_distance_map(file_list, 
                        data_type=None, 
                        use_mask=1, 
                        use_cdelt=0,
                        subtract_base_image=0,
                        base_file_list=None):
    # Load one image + header (we only need geometry here)
    tB_data, pB_data, tB_hdr, pB_hdr = import_data(
        file_list,
        data_type=data_type,
        use_mask=use_mask,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_file_list=base_file_list,
    )

    # calculate constants
    solar_radii=c.R_sun
    solar_radii_in_km=solar_radii.to(u.kilometer)
    solar_radii_in_km=solar_radii_in_km.value

    sun_obs_dist = c.au# * u.kilometer   # km
    sun_obs_dist=sun_obs_dist.to(u.kilometer)
    sun_obs_dist=sun_obs_dist.value

    sun_obs_dist_rs=sun_obs_dist/solar_radii_in_km

    # Allocate
    spatial_plane_distance = np.zeros_like(pB_data, dtype=float)

    # Geometry
    if use_cdelt:
        # FITS is 1-based; Python is 0-based
        x_center = pB_hdr['CRPIX1'] - 1.0
        y_center = pB_hdr['CRPIX2'] - 1.0

        if data_type == 'noise_gate_data':
            x_pix_km = solar_radii_in_km * pB_hdr['CDELT1']
            y_pix_km = solar_radii_in_km * pB_hdr['CDELT2']
        else:
            x_pix_km = solar_radii_in_km * pB_hdr['CDELT1'] / pB_hdr['RSUN']
            y_pix_km = solar_radii_in_km * pB_hdr['CDELT2'] / pB_hdr['RSUN']

        ydim, xdim = spatial_plane_distance.shape

    else:
        ydim, xdim = spatial_plane_distance.shape
        x_center = (xdim - 1) / 2.0
        y_center = (ydim - 1) / 2.0

        fov_solar_radii = 64.0
        x_pix_km = fov_solar_radii * solar_radii_in_km / xdim
        y_pix_km = fov_solar_radii * solar_radii_in_km / ydim

    # Compute distances in the image plane
    for y in range(ydim):
        dy = (y - y_center) * y_pix_km
        for x in range(xdim):
            dx = (x - x_center) * x_pix_km
            spatial_plane_distance[y, x] = np.hypot(dx, dy)

    # This is r_pos in km; good to feed straight into radial_position_ps
    return spatial_plane_distance



def print_minmax(arr, label="array"):
    """
    Prints the min/max of an array, ignoring NaN values.
    """
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)

    if not np.any(finite):
        print(f"{label}: no finite values (all NaN or inf)")
        return

    print(f"{label}: min = {np.nanmin(arr):.6g}, max = {np.nanmax(arr):.6g}")



def clean_distance(dist_km, dist_obs_to_source_km, max_factor=2.0):
    """
    Clean a distance array (in km):
    - remove NaN/inf
    - remove negative values
    - remove values larger than max_factor * dist_obs_to_source_km

    Returns a new array with bad entries set to np.nan.
    """
    dist = np.asarray(dist_km, dtype=float).copy()

    # mask non-finite
    bad = ~np.isfinite(dist)
    # mask negative
    bad |= (dist < 0)
    # mask crazy-large values (beyond some multiple of observer distance)
    bad |= (dist > max_factor * dist_obs_to_source_km)

    dist[bad] = np.nan
    return dist


def to_solar_radii(dist_km):
    """
    Convert distance in km to solar radii.
    """
    R_SUN_KM = c.R_sun.to(u.kilometer).value
    dist = np.asarray(dist_km, dtype=float)
    return dist / R_SUN_KM

def depth_from_radial(r_km, dist_obs_to_source_km):
    r = np.asarray(r_km, dtype=float)
    bad = ~np.isfinite(r) | (r <= 0) | (r > dist_obs_to_source_km)
    D = np.full_like(r, np.nan, dtype=float)
    good = ~bad

    D[good] = 1.0 - (r[good] / dist_obs_to_source_km)
    D[good] = np.clip(D[good], 0.0, 1.0)
    return D

def depth_map_from_los(l_plus_km, dist_obs_to_source_km, vmin=0.0, vmax=1.0):
    """
    Create a normalized depth map from l_plus (LOS distance, km).

    D = 1 - l / dist_obs_to_source

    - D ~ 0   → near the Sun (dark)
    - D ~ 1   → near the observer (bright)

    Returns
    -------
    D : ndarray
        Normalized depth in [0, 1], with NaNs in invalid pixels.
    """
    l = np.asarray(l_plus_km, dtype=float).copy()

    # Remove non-finite and outside [0, dist_obs]
    bad = ~np.isfinite(l)
    bad |= (l < 0)
    bad |= (l > dist_obs_to_source_km)

    D = np.full_like(l, np.nan, dtype=float)
    good = ~bad

    # normalized depth
    D[good] = 1.0 - (l[good] / dist_obs_to_source_km)

    # clip to [vmin, vmax] just for safety
    D[good] = np.clip(D[good], vmin, vmax)

    return D