import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy import signal
from astropy import units as u, constants as c

# generate simple output images
def show_1_image(img, title1=""):
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
    plt.imshow(img, cmap="gray")
    plt.title(title1)
    plt.colorbar(shrink=0.7)
    plt.axis("on")
    plt.tight_layout()
    plt.show()


def show_2_images(img1, img2, title1="", title2=""):
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
    plt.imshow(img1, cmap="gray")
    plt.title(title1)
    plt.colorbar(shrink=0.7)
    plt.axis("on")

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap="gray")
    plt.title(title2)
    plt.colorbar(shrink=0.7)
    plt.axis("on")

    plt.tight_layout()
    plt.show()


def show_3_images(img1, img2, img3, title1="", title2="", title3=""):
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
    plt.imshow(img1, cmap="gray")
    plt.title(title1)
    plt.colorbar(shrink=0.7)
    plt.axis("on")

    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap="gray")
    plt.title(title2)
    plt.colorbar(shrink=0.7)
    plt.axis("on")

    plt.subplot(1, 3, 3)
    plt.imshow(img3, cmap="gray")
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


import numpy as np
from astropy import units as u, constants as c

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


import numpy as np


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
