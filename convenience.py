from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import sys

from pathlib import Path
from astropy.io import fits
from scipy import signal
from astropy import units as u, constants as c
from scipy.ndimage import gaussian_filter
from typing import Iterable, Optional, Union

PathLike = Union[str, Path]

# generate simple output images
def show_1_image(img, title1="", cmap="gray",extent=None):
    """
    Display a single 2D data array as an image.

    Parameters
    ----------
    img : 2D array-like
        The image or data array to display.
    title : str, optional
        Title shown above the image.

    Notes
    -----
    Displays the image using a grayscale colormap with a colorbar.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap, origin="lower",
    extent=extent)
    plt.title(title1)
    plt.colorbar(shrink=0.7)
    plt.axis("on")
    plt.tight_layout()
    plt.show()


def show_2_images(img1, 
                  img2, 
                  title1="", 
                  title2="",
                  cmap1="gray",
                  cmap2="gray",
                  extent1=None,
                  extent2=None):
    """
    Display two 2D data arrays side-by-side for comparison.

    Parameters
    ----------
    img1, img2 : 2D array-like
        Images or data arrays to display.
    title1, title2 : str, optional
        Titles shown above each subplot.

    Notes
    -----
    Uses a grayscale colormap with colorbars and matched axes.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap=cmap1, origin="lower",
    extent=extent1)

    plt.title(title1)
    plt.colorbar(shrink=0.7)
    plt.axis("on")

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap=cmap2, origin="lower",
    extent=extent2)
    plt.title(title2)
    plt.colorbar(shrink=0.7)
    plt.axis("on")

    plt.tight_layout()
    plt.show()


def show_3_images(img1,
                  img2, 
                  img3, 
                  title1="",
                  title2="",
                  title3="",
                  cmap1="gray",
                  cmap2="gray",
                  cmap3="gray",
                  extent1=None,
                  extent2=None,
                  extent3=None):
    """
    Display three 2D data arrays side-by-side for visual comparison.

    Parameters
    ----------
    img1, img2, img3 : 2D array-like
        Images or data arrays to display.
    title1, title2, title3 : str, optional
        Titles shown above each subplot.

    Notes
    -----
    Each subplot uses a grayscale colormap with its own colorbar.
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap=cmap1, origin="lower",
    extent=extent1)
    plt.title(title1)
    plt.colorbar(shrink=0.7)
    plt.axis("on")

    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap=cmap2, origin="lower",
    extent=extent2)
    plt.title(title2)
    plt.colorbar(shrink=0.7)
    plt.axis("on")

    plt.subplot(1, 3, 3)
    plt.imshow(img3, cmap=cmap3, origin="lower",
    extent=extent3)
    plt.title(title3)
    plt.colorbar(shrink=0.7)
    plt.axis("on")

    plt.tight_layout()
    plt.show()

def generate_square_test_pattern(size=256, square_size=32, noise_level=0):
    """
    Generate a 2D test array containing non-overlapping squares of increasing values,
    with optional adjustable noise.

    Parameters
    ----------
    size : int
        Size of the full image (size x size).
    square_size : int
        Size of each square.
    noise_level : float
        Noise strength from 0 to 100:
            0   -> no noise
            100 -> image replaced by pure random noise
            (intermediate values blend between clean pattern and noise)

    Returns
    -------
    numpy.ndarray
        A 2D array with square patterns and optional noise.
    """

    # 1. Generate clean pattern
    data = np.zeros((size, size), dtype=float)
    value = 1

    n = size // square_size  # number of squares per dimension

    for i in range(n):
        for j in range(n):
            y0 = i * square_size
            y1 = y0 + square_size
            x0 = j * square_size
            x1 = x0 + square_size
            data[y0:y1, x0:x1] = value
            value += 1

    # 2. Add controllable noise
    noise_level = np.clip(noise_level, 0, 100)   # enforce range

    if noise_level > 0:
        # Normalise noise_level to 0–1 scale
        alpha = noise_level / 100.0

        # Generate noise with the same dynamic range as the image
        noise = np.random.normal(
            loc=np.mean(data),
            scale=np.std(data),
            size=data.shape
        )

        # Blend: (1-alpha) * clean + alpha * noise
        data = (1 - alpha) * data + alpha * noise

    return data


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
    
    return data, header


def minmax(arr, label="array"):
    """
    Prints the min/max of an array, ignoring NaN values.
    """
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)

    if not np.any(finite):
        print(f"{label}: no finite values (all NaN or inf)")
        return

    print(f"{label}: min = {np.nanmin(arr):.6g}, max = {np.nanmax(arr):.6g}")


def to_solar_radii(dist_km):
    """
    Convert distance in km to solar radii.
    """
    R_SUN_KM = c.R_sun.to(u.kilometer).value
    dist = np.asarray(dist_km, dtype=float)
    return dist / R_SUN_KM




def to_km(
    dist,
    unit="Rsun",
    R_sun_km=None,
    AU_km=None,
    L1_km=None
):
    """
    Convert heliophysics distance units into kilometers.

    Parameters
    ----------
    dist : float or array_like
        Input distance value(s) in the specified unit.
    unit : str, optional
        One of:
        - "Rsun" : solar radii → km
        - "AU"   : astronomical units → km
        - "L1"   : multiples of Sun→L1 distance → km
        - "L5"   : multiples of Sun→L5 distance → km (≈1 AU)
    R_sun_km : float, optional
        Solar radius in km. Defaults to IAU value if None.
    AU_km : float, optional
        Distance of 1 AU in km. Defaults to astropy if None.
    L1_km : float, optional
        Sun→L1 distance in km.
        If None, computed as 1 AU − 1.5e6 km.

    Returns
    -------
    float or ndarray
        Distance(s) converted to kilometers.
    """
    dist = np.asarray(dist, dtype=float)

    # Defaults
    if R_sun_km is None:
        R_sun_km = c.R_sun.to(u.km).value

    if AU_km is None:
        AU_km = c.au.to(u.km).value

    # Sun to L1, not Earth to L1
    if L1_km is None:
        EARTH_L1 = 1.5e6  # km
        L1_km = AU_km - EARTH_L1  # ~1.481e8 km

    # Normalize unit
    unit = unit.lower()

    if unit in ["rsun", "r_sun", "solar", "solar_radii"]:
        return dist * R_sun_km

    elif unit == "au":
        return dist * AU_km

    elif unit in ["l1", "sun-l1", "sun_l1"]:
        return dist * L1_km

    else:
        raise ValueError(f"Unrecognized unit '{unit}'.")



def _gauss_kern(size: int, size_y: int | None = None) -> np.ndarray:
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
    kernel = _gauss_kern(n, size_y=n_y)
    blurred = signal.convolve(image, kernel, mode="valid")
    return blurred



def clean_distance(dist_km, dist_obs_to_source_km, max_factor=2.0):
    """
    Clean and sanity-filter a distance array expressed in kilometers.

    This is useful for post-processing LOS or radial distance solutions
    where numerical instabilities can produce unphysical values.

    The function performs the following clean-ups:
      1. Converts the input to a float numpy array.
      2. Removes non-finite values (NaN, +inf, -inf).
      3. Removes negative distances.
      4. Removes distances that exceed `max_factor * dist_obs_to_source_km`
         (e.g., default = 2× observer–Sun distance).

    Parameters
    ----------
    dist_km : array_like
        Distance array in kilometers (e.g., r_plus, l_plus, etc.).
    dist_obs_to_source_km : float
        Physical distance from the observer to the Sun (in km).
        For 1 AU use: ``dist_obs_to_source_km = c.au.to(u.km).value``.
    max_factor : float, optional
        Maximum allowed physical range before values are considered invalid.
        Default=2.0 means values > 2 × (Sun–observer distance) are set to NaN.

    Returns
    -------
    cleaned : numpy.ndarray
        A new array where all invalid entries are replaced by ``np.nan``.
        Shape matches the input.

    Examples
    --------
    >>> from astropy.constants import au
    >>> dist_obs = au.to('km').value
    >>> dirty = np.array([1e5, -5, np.inf, 3*dist_obs, 1e8])
    >>> clean_distance(dirty, dist_obs)
    array([1.0000e+05,        nan,        nan,        nan, 1.0000e+08])
    """
    # Convert to array
    dist = np.asarray(dist_km, dtype=float).copy()

    # Mask invalid entries
    bad = ~np.isfinite(dist)                    # NaN / inf
    bad |= (dist < 0)                           # negative distances
    bad |= (dist > max_factor * dist_obs_to_source_km)   # too large

    # Replace with NaN
    dist[bad] = np.nan
    return dist





def copy_metadata(stereo_path, synthetic_input, output_path,
                  preserve_synthetic_keys=None):
    """
    Copy headers from stereo_path onto synthetic_input data and save to output_path.

    Parameters
    ----------
    stereo_path : str or Path
        Path to the STEREO FITS (metadata template).

    synthetic_input : str, Path, np.ndarray, or list of np.ndarray
        - Path to synthetic FITS (original behaviour), OR
        - numpy array (single-image), OR
        - list of numpy arrays (for multiple HDUs).

    output_path : str or Path
        Output FITS file path.

    preserve_synthetic_keys : list of str, optional
        List of header keys to preserve from the synthetic file (e.g. ["HISTORY"]).
    """

    stereo_path = Path(stereo_path)
    output_path = Path(output_path)

    if preserve_synthetic_keys is None:
        preserve_synthetic_keys = []

    # ----------------------------------------------------------------------
    # Load stereo FITS (metadata template)
    # ----------------------------------------------------------------------
    with fits.open(stereo_path, mode="readonly") as hdu_stereo:

        # ------------------------------------------------------------------
        # Determine the synthetic input type
        # ------------------------------------------------------------------
        if isinstance(synthetic_input, (str, Path)):
            # Case A: synthetic_input is a FITS file path
            syn_hdul = fits.open(synthetic_input, mode="readonly")
            synthetic_is_fits = True

        elif isinstance(synthetic_input, np.ndarray):
            # Case B: a single numpy array → create minimal HDU list
            syn_hdul = fits.HDUList([
                fits.PrimaryHDU(data=synthetic_input)
            ])
            synthetic_is_fits = False

        elif isinstance(synthetic_input, list) and all(isinstance(x, np.ndarray) for x in synthetic_input):
            # Case C: list of numpy arrays → multi-HDU synthetic
            hdus = [fits.PrimaryHDU(data=synthetic_input[0])]
            for arr in synthetic_input[1:]:
                hdus.append(fits.ImageHDU(data=arr))
            syn_hdul = fits.HDUList(hdus)
            synthetic_is_fits = False

        else:
            raise TypeError(
                "synthetic_input must be a FITS path, a numpy array, "
                "or a list of numpy arrays."
            )

        # ------------------------------------------------------------------
        # Build new HDUs
        # ------------------------------------------------------------------
        new_hdus = []

        n_hdus = min(len(hdu_stereo), len(syn_hdul))

        for i in range(n_hdus):
            stereo_hdu = hdu_stereo[i]
            syn_hdu = syn_hdul[i]

            # Copy stereo header and synthetic data
            new_header = stereo_hdu.header.copy()
            new_data = syn_hdu.data

            # Optionally preserve synthetic header keys (only if synthetic was FITS)
            if synthetic_is_fits:
                for key in preserve_synthetic_keys:
                    if key in syn_hdu.header:
                        new_header[key] = syn_hdu.header[key]

            # Construct HDU
            if i == 0:
                new_hdu = fits.PrimaryHDU(data=new_data, header=new_header)
            else:
                new_hdu = fits.ImageHDU(data=new_data, header=new_header)

            new_hdus.append(new_hdu)

        # Append any extra synthetic HDUs (mostly relevant for FITS input)
        if len(syn_hdul) > n_hdus:
            for extra in syn_hdul[n_hdus:]:
                new_hdus.append(extra.copy())

        # ------------------------------------------------------------------
        # Write output
        # ------------------------------------------------------------------
        new_hdul = fits.HDUList(new_hdus)
        new_hdul.writeto(output_path, overwrite=True)
        print(f"Wrote: {output_path}")

        # Clean up FITS handles if opened from disk
        if synthetic_is_fits:
            syn_hdul.close()




def upscale_and_smooth(data, factor=4, sigma=1.0):
    """
    Upscale a 2D array by an integer factor using nearest-neighbour replication
    and apply a Gaussian smoothing.

    Parameters
    ----------
    data : ndarray (ny, nx)
        Input 2D array.
    factor : int
        Integer upscaling factor (e.g. 4: 512 → 2048).
    sigma : float
        Gaussian sigma in pixels (on the upscaled grid).

    Returns
    -------
    upscaled : ndarray
        Upscaled and smoothed array.
    """
    data = np.asarray(data)

    # Nearest-neighbour upscale
    upscaled = np.repeat(np.repeat(data, factor, axis=0), factor, axis=1)

    # Smooth
    if sigma > 0:
        upscaled = gaussian_filter(upscaled, sigma=sigma)

    return upscaled





def downsample_2x(image: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Downsample a 2D image by factor 2 in each axis.

    Parameters
    ----------
    image : np.ndarray
        2D array with even dimensions.
    method : {"mean", "sum", "decimate"}
        - "mean": 2x2 binning mean (recommended; output typically float)
        - "sum":  2x2 binning sum
        - "decimate": take every other pixel (fastest)

    Returns
    -------
    np.ndarray
        Downsampled image of shape (ny/2, nx/2).
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    ny, nx = image.shape
    if (ny % 2) or (nx % 2):
        raise ValueError(f"Image shape must be even, got {image.shape}")

    method = method.lower()
    if method == "decimate":
        return image[::2, ::2]

    b = image.reshape(ny // 2, 2, nx // 2, 2)
    if method == "mean":
        return b.mean(axis=(1, 3))
    if method == "sum":
        return b.sum(axis=(1, 3))

    raise ValueError(f"Unknown method: {method}")


def update_header_for_2x_downsample(hdr: fits.Header) -> fits.Header:
    """
    Update common WCS/axis keywords for a 2x downsample.

    Applies:
      - CRPIX1/2 -> CRPIX1/2 / 2
      - CDi_j (if present) -> CDi_j * 2
      - else CDELT1/2 (if present) -> CDELT1/2 * 2
      - NAXIS1/2 -> NAXIS1/2 / 2 (if present)

    Notes
    -----
    This assumes you are binning/decimating by exactly 2 in both axes.
    """
    hdr2 = hdr.copy()

    for k in ("CRPIX1", "CRPIX2"):
        if k in hdr2:
            hdr2[k] = hdr2[k] / 2.0

    cd_keys = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
    has_cd = any(k in hdr2 for k in cd_keys)

    if has_cd:
        for k in cd_keys:
            if k in hdr2:
                hdr2[k] = hdr2[k] * 2.0
    else:
        for k in ("CDELT1", "CDELT2"):
            if k in hdr2:
                hdr2[k] = hdr2[k] * 2.0

    if "NAXIS1" in hdr2:
        hdr2["NAXIS1"] = int(hdr2["NAXIS1"] // 2)
    if "NAXIS2" in hdr2:
        hdr2["NAXIS2"] = int(hdr2["NAXIS2"] // 2)

    hdr2.add_history("Downsampled 2x (2048->1024) and updated CRPIX/CDELT/CD accordingly.")
    return hdr2


def shrink_fits_file_inplace(
    file_path: PathLike,
    method: str = "mean",
    backup: bool = True,
    all_image_hdus: bool = False,
    overwrite: bool = True,
) -> Path:
    """
    Downsample FITS image data in-place (2048->1024 if input is 2k square),
    updating header WCS/axis keywords.

    Parameters
    ----------
    file_path : str | Path
        Path to a FITS file.
    method : {"mean","sum","decimate"}
        Downsampling method.
    backup : bool
        If True, create a sibling backup "<file>.bak" if it doesn't already exist.
    all_image_hdus : bool
        If False, only process HDU 0 (PrimaryHDU) if it is a 2D image.
        If True, process every HDU that contains a 2D image array.
    overwrite : bool
        If False, raises if output would overwrite (only relevant if you adapt
        this later to a different output path).

    Returns
    -------
    Path
        The path that was written (same as input).
    """
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    if backup:
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            bak.write_bytes(p.read_bytes())

    # Read, build a new HDUList, then atomically replace original
    with fits.open(p, mode="readonly", memmap=False) as hdul:
        new_hdul = fits.HDUList()

        for i, hdu in enumerate(hdul):
            data = hdu.data
            hdr = hdu.header

            do_this = False
            if all_image_hdus:
                do_this = isinstance(data, np.ndarray) and data.ndim == 2
            else:
                do_this = (i == 0) and isinstance(data, np.ndarray) and data.ndim == 2

            if do_this:
                new_data = downsample_2x(data, method=method)
                new_hdr = update_header_for_2x_downsample(hdr)

                if isinstance(hdu, fits.PrimaryHDU):
                    new_hdu = fits.PrimaryHDU(data=new_data, header=new_hdr)
                else:
                    new_hdu = fits.ImageHDU(data=new_data, header=new_hdr, name=hdu.name)

                new_hdul.append(new_hdu)
            else:
                new_hdul.append(hdu.copy())

    # Write to temp then replace for safety
    tmp = p.with_suffix(p.suffix + ".tmp")
    if tmp.exists() and not overwrite:
        raise FileExistsError(tmp)

    new_hdul.writeto(tmp, overwrite=True, output_verify="fix")
    tmp.replace(p)

    return p


def shrink_fits_many_inplace(
    paths: Iterable[PathLike],
    method: str = "mean",
    backup: bool = True,
    all_image_hdus: bool = False,
) -> list[Path]:
    """
    Convenience wrapper to process multiple FITS files.

    Returns list of processed Paths.
    """
    out: list[Path] = []
    for fp in paths:
        out.append(
            shrink_fits_file_inplace(
                fp, method=method, backup=backup, all_image_hdus=all_image_hdus
            )
        )
    return out


def list_fits_in_dir(directory: PathLike, pattern: str = "*.fits") -> list[Path]:
    """
    Return sorted FITS paths in a directory matching a glob pattern.
    """
    d = Path(directory).expanduser().resolve()
    return sorted(d.glob(pattern))
