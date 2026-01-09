"""
plotter.py — plotting utilities for VAPOR

This module provides:
- create_figure_core: save a single 2D array as a PNG with basic scaling rules
- create_figure: convenience wrapper to save tB and pB PNGs from file paths
- create_triple_stereo_plot: the “triple plot” (central map + x/y profiles)

IMPORTANT (API alignment):
- support.import_data() loads ONE file at a time and accepts base_file_path=...
- core.radial_position_ps(tb_path, pb_path, return_key=...) returns ONE map at a time
  (front/back solutions are accessed via different return_key values)
"""

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from astropy.constants import R_sun, au
import astropy.units as u

from core import radial_position_ps, radial_position_scatter
from support import import_data, create_distance_map


# -------------------------------------------------------------------
# Matplotlib defaults
# -------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.size": 9,
})

R_SUN_KM = R_sun.to_value(u.km)
DEFAULT_R_OBS = au.to_value(u.km)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _split_paths(
    file_list: Sequence[Union[str, Path]],
    base_file_list: Optional[Sequence[Union[str, Path]]] = None,
) -> Tuple[Path, Path, Optional[Path], Optional[Path]]:
    """
    Split [tb_path, pb_path] and optional [base_tb_path, base_pb_path].
    """
    if file_list is None or len(file_list) != 2:
        raise ValueError("file_list must be a 2-element sequence: [tb_path, pb_path]")

    tb_path = Path(file_list[0])
    pb_path = Path(file_list[1])

    base_tb = base_pb = None
    if base_file_list is not None:
        if len(base_file_list) != 2:
            raise ValueError("base_file_list must be a 2-element sequence: [base_tb_path, base_pb_path]")
        base_tb = Path(base_file_list[0]) if base_file_list[0] is not None else None
        base_pb = Path(base_file_list[1]) if base_file_list[1] is not None else None

    return tb_path, pb_path, base_tb, base_pb


def _load_tb_pb(
    tb_path: Path,
    pb_path: Path,
    *,
    data_type: Optional[str],
    use_mask: bool,
    use_cdelt: bool,
    subtract_base_image: bool,
    base_tb_path: Optional[Path],
    base_pb_path: Optional[Path],
):
    """
    Load tB and pB with consistent options using support.import_data() (single-file API).
    """
    tB, _ = import_data(
        tb_path,
        data_type=data_type,
        use_mask=use_mask,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_file_path=str(base_tb_path) if base_tb_path is not None else None,
    )

    pB, _ = import_data(
        pb_path,
        data_type=data_type,
        use_mask=use_mask,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_file_path=str(base_pb_path) if base_pb_path is not None else None,
    )

    tB = np.asarray(tB, dtype=float)
    pB = np.asarray(pB, dtype=float)

    if tB.shape != pB.shape:
        raise ValueError(f"tB and pB must have the same shape; got {tB.shape} vs {pB.shape}")

    return tB, pB


def _get_minmax_fov(data_type: Optional[str], dist_image_plane_km: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Determine (minxy, maxxy) in R_sun units for plotting.
    """
    if data_type in ("stereo_dif", "stereo", "noise"):
        return -16.0, 16.0
    if data_type == "forward":
        return -32.0, 32.0
    if dist_image_plane_km is not None and np.isfinite(dist_image_plane_km).any():
        max_r_rs = float(np.nanmax(dist_image_plane_km) / R_SUN_KM)
        maxxy = float(np.ceil(max_r_rs))
        return -maxxy, maxxy
    return -16.0, 16.0


# -------------------------------------------------------------------
# Simple figure creators
# -------------------------------------------------------------------
def create_figure_core(image_data, image_name=None, data_type=None):
    """
    Save a simple image of the data with basic scaling per data_type.
    """
    if image_name is None:
        image_name = "Test.png"

    figure = plt.figure(frameon=False)
    axes = plt.Axes(figure, [0.0, 0.0, 1.0, 1.0])
    axes.set_axis_off()
    figure.add_axes(axes)
    figure.set_size_inches(15, 15)

    if data_type in ("stereo_dif", "stereo"):
        plt.imshow(image_data, cmap="gray", vmin=-20, vmax=20, origin="lower", axes=axes)
    elif data_type == "noise":
        plt.imshow(image_data, cmap="gray", vmin=-15, vmax=15, origin="lower", axes=axes)
    elif data_type == "forward":
        minxy, maxxy = -32, 32
        plt.imshow(image_data, extent=[minxy, maxxy, minxy, maxxy], cmap="gray", origin="lower", axes=axes)
    else:
        plt.imshow(image_data, cmap="gray", origin="lower", axes=axes)

    plt.savefig(image_name, format="png")
    plt.close()


def create_figure(
    file_list,
    data_type=None,
    use_mask=True,
    use_cdelt=True,
    subtract_base_image=False,
    base_file_list=None,
    image_name=None,
):
    """
    Convenience wrapper: load tB, pB and save individual PNGs.
    """
    tb_path, pb_path, base_tb, base_pb = _split_paths(file_list, base_file_list)

    tB, pB = _load_tb_pb(
        tb_path,
        pb_path,
        data_type=data_type,
        use_mask=use_mask,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_tb_path=base_tb,
        base_pb_path=base_pb,
    )

    if image_name is None:
        image_name = data_type or "image"

    create_figure_core(pB, f"pB - {image_name}.png", data_type)
    create_figure_core(tB, f"tB - {image_name}.png", data_type)


# -------------------------------------------------------------------
# Aggregator and triple-plot helpers
# -------------------------------------------------------------------
def data_aggregator(indata, aggregator_type="max"):
    """
    Aggregate 1D array indata according to aggregator_type.
    """
    if aggregator_type == "mean":
        return np.mean(indata)
    if aggregator_type == "std":
        return np.std(indata)
    if aggregator_type == "med":
        return np.median(indata)
    if aggregator_type == "max":
        return np.max(indata)
    if aggregator_type == "min":
        return np.min(indata)
    if aggregator_type == "sum":
        return np.sum(indata)
    raise ValueError(f"Unknown aggregator_type: {aggregator_type}")


def make_triple_plot_data(data, aggregator="max", minvalue=None, maxvalue=None, verbose=False):
    """
    Reduce a 2D map into 1D x- and y-profiles by aggregating along rows/cols,
    with optional clipping in value-space [minvalue, maxvalue].
    """
    dim = data.shape

    if minvalue is None:
        minvalue = np.nanmin(data)
    if maxvalue is None:
        maxvalue = np.nanmax(data)

    yPlot = np.zeros(dim[0])
    for iStep in range(dim[0]):
        row = data[iStep, :]
        outrow = row[(row >= minvalue) & (row <= maxvalue)]
        yPlot[iStep] = data_aggregator(outrow, aggregator_type=aggregator) if outrow.size > 0 else 0.0

    xPlot = np.zeros(dim[1])
    for jStep in range(dim[1]):
        col = data[:, jStep]
        outcol = col[(col >= minvalue) & (col <= maxvalue)]
        xPlot[jStep] = data_aggregator(outcol, aggregator_type=aggregator) if outcol.size > 0 else 0.0

    return yPlot, xPlot


# -------------------------------------------------------------------
# Triple plot (updated to match new core + support APIs)
# -------------------------------------------------------------------
def create_triple_stereo_plot(
    file_list,
    data_type=None,
    use_mask=True,
    use_cdelt=False,
    subtract_base_image=False,
    base_file_list=None,
    output_method="ps",          # "ps" or "scattered"
    image_name=None,
    plot_max=None,
    plot_min=None,
    apply_filter="Gauss",
    gauss_filter_sig_x=1.5,
    gauss_filter_sig_y=1.5,
    side_panel_filter="savgol",
    savgol_window_size=25,
    savgol_poly_order=2,
    show_time=False,
    verbose=False,
    color_map="YlOrRd",
    include_name=True,
    dist_obs_to_source_km=DEFAULT_R_OBS,
    solution="minus",            # "plus" or "minus" branch
    quantity="los",              # "los" (default), "pos", or "radial"
    min_cut_off_value=None,
    max_cut_off_value=None,
):
    """
    Create a "triple" plot:
      - central 2D map (LOS, POS distance, or radial distance) in R_sun,
      - top panel: X-profile,
      - right panel: Y-profile.

    This version is API-consistent with:
      - support.import_data(file_path, ..., base_file_path=...)
      - core.radial_position_ps(tb_path, pb_path, return_key=...)
    """
    # --------------------------------------------------------------
    # 0) Parse inputs
    # --------------------------------------------------------------
    tb_path, pb_path, base_tb, base_pb = _split_paths(file_list, base_file_list)

    # --------------------------------------------------------------
    # 1) Load data
    # --------------------------------------------------------------
    _tB, _pB = _load_tb_pb(
        tb_path,
        pb_path,
        data_type=data_type,
        use_mask=use_mask,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_tb_path=base_tb,
        base_pb_path=base_pb,
    )

    # Distance map for FOV fallback (create_distance_map expects a file path)
    dist_image_plane_km = create_distance_map(pb_path, data_type=data_type, use_cdelt=use_cdelt)

    R_obs_km = float(dist_obs_to_source_km)
    R_obs_rs = R_obs_km / R_SUN_KM

    # --------------------------------------------------------------
    # 2) Run inversion (ps or scattered)
    # --------------------------------------------------------------
    if output_method == "ps":
        if quantity == "radial":
            front_key, back_key = "r_front", "r_back"
        elif quantity == "pos":
            front_key, back_key = "x_front", "x_back"
        elif quantity == "los":
            front_key, back_key = "l_front", "l_back"
        else:
            raise ValueError("quantity must be 'los', 'pos', or 'radial'")

        front_map = radial_position_ps(
            tb_path, pb_path,
            return_key=front_key,
            data_type=data_type,
            use_mask=use_mask,
            use_cdelt=use_cdelt,
            subtract_base_image=subtract_base_image,
            base_tb_path=str(base_tb) if base_tb is not None else None,
            base_pb_path=str(base_pb) if base_pb is not None else None,
            dist_obs_to_source_km=R_obs_km,
        )
        back_map = radial_position_ps(
            tb_path, pb_path,
            return_key=back_key,
            data_type=data_type,
            use_mask=use_mask,
            use_cdelt=use_cdelt,
            subtract_base_image=subtract_base_image,
            base_tb_path=str(base_tb) if base_tb is not None else None,
            base_pb_path=str(base_pb) if base_pb is not None else None,
            dist_obs_to_source_km=R_obs_km,
        )

    elif output_method == "scattered":
        # Placeholder: only wire this if/when radial_position_scatter matches the new API.
        raise NotImplementedError(
            "output_method='scattered' is not wired yet: radial_position_scatter must be updated "
            "to accept (tb_path, pb_path, return_key=...) like radial_position_ps."
        )
    else:
        raise ValueError("output_method must be 'ps' or 'scattered'")

    # Choose branch
    if solution == "plus":
        use_map = front_map
    elif solution == "minus":
        use_map = back_map
    else:
        raise ValueError("solution must be 'plus' or 'minus'")

    # --------------------------------------------------------------
    # 3) Convert to requested plotted quantity in R_sun
    # --------------------------------------------------------------
    if quantity == "los":
        # use_map is l_front/l_back in km (observer->feature)
        l_use_rs = use_map / R_SUN_KM
        rad_out = R_obs_rs - l_use_rs
        cb_label = "Line of Sight (R$_\\odot$)"
        side_label = "LOS (R$_\\odot$)"

    elif quantity == "pos":
        # use_map is x_front/x_back in km
        rad_out = use_map / R_SUN_KM
        cb_label = "Distance from POS (R$_\\odot$)"
        side_label = "POS x (R$_\\odot$)"

    elif quantity == "radial":
        # use_map is r_front/r_back in km
        rad_out = use_map / R_SUN_KM
        cb_label = "Radial distance (R$_\\odot$)"
        side_label = "Radial (R$_\\odot$)"
    else:
        raise ValueError("quantity must be 'los', 'pos', or 'radial'")

    # Cutoffs
    if max_cut_off_value is not None:
        rad_out = np.where(rad_out > max_cut_off_value, np.nan, rad_out)
    if min_cut_off_value is not None:
        rad_out = np.where(rad_out < min_cut_off_value, np.nan, rad_out)

    rad_out_orig = rad_out.copy()

    # --------------------------------------------------------------
    # 4) Optional smoothing and side profiles
    # --------------------------------------------------------------
    if apply_filter == "Gauss":
        sigma = [gauss_filter_sig_y, gauss_filter_sig_x]
        rad_out = gaussian_filter(rad_out, sigma=sigma, mode="constant")

    if plot_min is None:
        plot_min = float(np.nanmin(rad_out))
    if plot_max is None:
        plot_max = float(np.nanmax(rad_out))

    y_rad_out, x_rad_out = make_triple_plot_data(
        rad_out, aggregator="max", minvalue=plot_min, maxvalue=plot_max
    )

    if side_panel_filter == "savgol":
        # guard against too-large windows
        n = x_rad_out.size
        win = int(savgol_window_size)
        if win >= n:
            win = n - 1 if (n % 2 == 0) else n
        if win < 3:
            win = 3
        if win % 2 == 0:
            win += 1
        poly = int(min(savgol_poly_order, win - 1))

        x_rad_out = savgol_filter(x_rad_out, win, poly)
        y_rad_out = savgol_filter(y_rad_out, win, poly)

    if image_name is None:
        image_name = "XTRIPLE_plot.png"

    if verbose:
        try:
            smooth_annotate_image_name = (
                image_name[9:11] + ":" + image_name[11:13] + ":" + image_name[13:15]
            )
        except Exception:
            smooth_annotate_image_name = image_name
        print("x max:", smooth_annotate_image_name, np.nanmax(x_rad_out))
        print("y max:", smooth_annotate_image_name, np.nanmax(y_rad_out))

    # --------------------------------------------------------------
    # 5) Plot the triple mosaic
    # --------------------------------------------------------------
    minxy, maxxy = _get_minmax_fov(data_type, dist_image_plane_km)

    y_out_array = np.array(y_rad_out, copy=True)
    x_out_array = np.array(x_rad_out, copy=True)

    # Replace NaNs in side profiles with zero (for plotting only)
    x_out_array = np.nan_to_num(x_out_array, nan=0.0)
    y_out_array = np.nan_to_num(y_out_array, nan=0.0)

    #y_out_array[y_out_array < 0.1] = np.nan
    #x_out_array[x_out_array < 0.1] = np.nan

    yvalues = np.flip(y_out_array, axis=0)
    n = x_out_array.size
    xvalues = np.linspace(minxy, maxxy, n)
    
    # this accounts for the origin being 'lower' in the central image
    yvalues = np.flip(yvalues, axis=0)
    #print(yvalues)

    fig = plt.figure(constrained_layout=True, figsize=(7.0, 6.0))
    axd = fig.subplot_mosaic(
        [["plotx", "corner"], ["image", "ploty"]],
        gridspec_kw={"width_ratios": [6, 1.3], "height_ratios": [1.3, 6]},
    )

    im = axd["image"].imshow(
        rad_out_orig,
        extent=[minxy, maxxy, minxy, maxxy],
        vmax=plot_max,
        vmin=plot_min,
        aspect="auto",
        cmap=None if color_map == "no_color" else color_map,
        origin="lower",
    )

    axd["image"].yaxis.set_label_position("left")
    axd["image"].yaxis.tick_left()
    axd["image"].set_ylabel("(R$_\\odot$)")
    axd["image"].set_xlabel("(R$_\\odot$)")

    if show_time:
        try:
            annotate_image_name = f"{image_name[0:4]}-{image_name[4:6]}-{image_name[6:8]}"
            annotate_image_name_2 = f"{image_name[9:11]}:{image_name[11:13]}:{image_name[13:15]} UT"
        except Exception:
            annotate_image_name = image_name
            annotate_image_name_2 = ""
        axd["image"].annotate(annotate_image_name, xy=(0, 0), xytext=(maxxy - 6.3, minxy + 2))
        if annotate_image_name_2:
            axd["image"].annotate(
                annotate_image_name_2, xy=(0, 0), xytext=(maxxy - 6.2, minxy + 0.8), fontsize=8
            )

    fig.colorbar(im, orientation="vertical", ax=axd["image"], label=cb_label, location="left")

    # Top panel
    if color_map == "no_color":
        axd["plotx"].plot(xvalues, x_out_array, linewidth=1.0, color="black")
    else:
        cmap = plt.cm.get_cmap(color_map)
        for i in range(0, n - 1):
            axd["plotx"].plot(
                xvalues[i:i + 2],
                x_out_array[i:i + 2],
                color=cmap(x_out_array[i] / plot_max) if np.isfinite(x_out_array[i]) else "black",
            )

    axd["plotx"].get_xaxis().set_visible(False)
    axd["plotx"].yaxis.set_label_position("left")
    axd["plotx"].yaxis.tick_left()
    axd["plotx"].set_ylabel(side_label)
    axd["plotx"].set_yticks([plot_min, plot_min + 0.5 * (plot_max - plot_min), plot_max])
    axd["plotx"].set_ylim([plot_min, plot_max])


    # Right panel
    if color_map == "no_color":
        axd["ploty"].plot(yvalues, xvalues, linewidth=1.0, color="black")
    else:
        cmap = plt.cm.get_cmap(color_map)
        for i in range(0, n - 1):
            axd["ploty"].plot(
                yvalues[i:i + 2],
                xvalues[i:i + 2],
                color=cmap(yvalues[i] / plot_max) if np.isfinite(yvalues[i]) else "black",
            )


    axd["ploty"].get_yaxis().set_visible(False)
    axd["ploty"].set_xlabel(side_label)
    axd["ploty"].set_xticks([plot_min, plot_min + 0.5 * (plot_max - plot_min), plot_max])
    axd["ploty"].set_xlim([plot_min, plot_max])

    # Corner logo
    if include_name:
        try:
            py_file_path = os.path.dirname(__file__)
            img = mpimg.imread(os.path.join(py_file_path, "images", "VAPOR.png"))
            axd["corner"].imshow(img)
            axd["corner"].get_xaxis().set_visible(False)
            axd["corner"].get_yaxis().set_visible(False)
            axd["corner"].axis("off")
        except Exception:
            axd["corner"].set_visible(False)
    else:
        axd["corner"].set_visible(False)

    plt.savefig(image_name, dpi=300)
    plt.close(fig)
