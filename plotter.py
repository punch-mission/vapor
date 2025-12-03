import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from astropy.constants import R_sun

from core import radial_position_ps, radial_position_scatter
from support import import_data, create_distance_map
from astropy.constants import R_sun, au
import astropy.units as u



# -------------------------------------------------------------------
# Matplotlib defaults
# -------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.size": 9,
})

# Solar radius as a Quantity in km
R_SUN_KM = R_sun.to_value(u.km)              # Quantity, e.g. <Quantity 695700. km>

# Observer distance as a Quantity in km
DEFAULT_R_OBS = au.to_value(u.km)         # <Quantity 149597870.7 km>

# -------------------------------------------------------------------
# Simple figure creators (unchanged logic, just imports)
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
        imgplot = plt.imshow(
            image_data,
            cmap="gray",
            vmin=-20,
            vmax=20,
            origin="lower",
            axes=axes,
        )
    elif data_type == "noise":
        imgplot = plt.imshow(
            image_data,
            cmap="gray",
            vmin=-15,
            vmax=15,
            origin="lower",
            axes=axes,
        )
    elif data_type == "forward":
        minxy, maxxy = -32, 32
        imgplot = plt.imshow(
            image_data,
            extent=[minxy, maxxy, minxy, maxxy],
            cmap="gray",
            origin="lower",
            axes=axes,
        )
    else:
        imgplot = plt.imshow(image_data, cmap="gray", origin="lower", axes=axes)

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
    tB, pB, tB_hdr, pB_hdr = import_data(
        file_list,
        data_type=data_type,
        use_mask=use_mask,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_file_list=base_file_list,
    )

    if image_name is None:
        image_name = data_type or "image"

    pB_name = f"pB - {image_name}.png"
    tB_name = f"tB - {image_name}.png"

    create_figure_core(pB, pB_name, data_type)
    create_figure_core(tB, tB_name, data_type)

# -------------------------------------------------------------------
# Aggregator and triple-plot helpers (your old logic)
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


def make_triple_plot_data(data, 
                        aggregator="max", 
                          minvalue=None, 
                          maxvalue=None, 
                          verbose=False):
    """
    Reduce a 2D map into 1D x- and y-profiles by aggregating along rows/cols,
    with optional clipping in value-space [minvalue, maxvalue].
    """
    dim = data.shape

    if minvalue is None:
        minvalue = np.nanmin(data)
    if maxvalue is None:
        maxvalue = np.nanmax(data)

    # Y profile (aggregate along columns for each row)
    yPlot = np.zeros(dim[0])
    for iStep in range(dim[0]):
        row = data[iStep, :]
        outrow = row[(row >= minvalue) & (row <= maxvalue)]
        if outrow.size > 0:
            out_value = data_aggregator(outrow, aggregator_type=aggregator)
        else:
            out_value = 0.0
        yPlot[iStep] = out_value

    # X profile (aggregate along rows for each column)
    xPlot = np.zeros(dim[1])
    for jStep in range(dim[1]):
        col = data[:, jStep]
        outcol = col[(col >= minvalue) & (col <= maxvalue)]
        if outcol.size > 0:
            out_value = data_aggregator(outcol, aggregator_type=aggregator)
        else:
            out_value = 0.0
        xPlot[jStep] = out_value

    return yPlot, xPlot

# -------------------------------------------------------------------
# Triple plot – NEW back-end, old visuals
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
    solution="minus",            # "plus" or "minus" LOS branch
    quantity="los",              # "los" (default), "pos", or "radial"
):

    """
    Create a "triple" plot:
      - central 2D LOS depth map in R_sun,
      - top panel: X-profile,
      - right panel: Y-profile.

    This is the old visual style but using the new geometry code.
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

    R_obs_km = dist_obs_to_source_km
    R_obs_rs = R_obs_km / R_SUN_KM

    # ------------------------------------------------------------------
    # 2. Run PR inversion (ps or scattered)
    # ------------------------------------------------------------------
    if output_method == "ps":
        r_plus, r_minus, l_plus, l_minus, x_plus, x_minus = radial_position_ps(
            file_list
        )
    elif output_method == "scattered":
        r_plus, r_minus, l_plus, l_minus, x_plus, x_minus = radial_position_scatter(
            file_list
        )
    else:
        raise ValueError("output_method must be 'ps' or 'scattered'")

    # Choose + / - branch
    if solution == "plus":
        r_use = r_plus
        l_use = l_plus
        x_use = x_plus
    elif solution == "minus":
        r_use = r_minus
        l_use = l_minus
        x_use = x_minus
    else:
        raise ValueError("solution must be 'plus' or 'minus'")

    # ------------------------------------------------------------------
    # 3. Convert to LOS coordinate in R_sun (zero at Sun)
    # ------------------------------------------------------------------
    l_use_rs = l_use / R_SUN_KM
    rad_out = R_obs_rs - l_use_rs      # R_sun; 0 at Sun

    if quantity == "los":
        # LOS distance relative to the Sun (0 at Sun, positive toward observer)
        l_use_rs = l_use / R_SUN_KM
        rad_out = R_obs_rs - l_use_rs
        cb_label = "Line of Sight (R$_\\odot$)"
        side_label = "LOS (R$_\\odot$)"

    elif quantity == "pos":
        # Distance from plane of sky: x = r sin(xi), already from core
        rad_out = x_use / R_SUN_KM
        cb_label = "Distance from POS (R$_\\odot$)"
        side_label = "POS x (R$_\\odot$)"

    elif quantity == "radial":
        # Heliocentric radial distance
        rad_out = r_use / R_SUN_KM
        cb_label = "Radial distance (R$_\\odot$)"
        side_label = "Radial (R$_\\odot$)"

    else:
        raise ValueError("quantity must be 'los', 'pos', or 'radial'")



    # Remove obviously bogus values
    rad_out = np.where(rad_out > 100.0, np.nan, rad_out)
    rad_out_orig = rad_out.copy()

    # ------------------------------------------------------------------
    # 4. Optional smoothing
    # ------------------------------------------------------------------
    if apply_filter == "Gauss":
        sigma = [gauss_filter_sig_y, gauss_filter_sig_x]
        rad_out = gaussian_filter(rad_out, sigma=sigma, mode="constant")

    if plot_min is None:
        plot_min = np.nanmin(rad_out)
    if plot_max is None:
        plot_max = np.nanmax(rad_out)

    # Profiles
    y_rad_out, x_rad_out = make_triple_plot_data(
        rad_out, aggregator="max", minvalue=plot_min, maxvalue=plot_max
    )

    # Smooth side panels
    if side_panel_filter == "savgol":
        x_rad_out = savgol_filter(x_rad_out, savgol_window_size, savgol_poly_order)
        y_rad_out = savgol_filter(y_rad_out, savgol_window_size, savgol_poly_order)

    if image_name is None:
        image_name = "XTRIPLE_plot.png"

    if verbose:
        # crude timestamp assumptions as before (YYYYMMDD_HHMMSS... in name)
        try:
            smooth_annotate_image_name = (
                image_name[9:11] + ":" + image_name[11:13] + ":" + image_name[13:15]
            )
        except Exception:
            smooth_annotate_image_name = image_name
        print("x max:", smooth_annotate_image_name, np.nanmax(x_rad_out))
        print("y max:", smooth_annotate_image_name, np.nanmax(y_rad_out))

    # ------------------------------------------------------------------
    # 5. Plot the triple mosaic (old style)
    # ------------------------------------------------------------------
    # Set FOV extents in R_sun (keep the old magic numbers by data_type)
    if data_type in ("stereo_dif", "stereo", "noise"):
        minxy, maxxy = -16.0, 16.0
    elif data_type == "forward":
        minxy, maxxy = -32.0, 32.0
    else:
        # fallback: estimate from image plane distance (km → R_sun)
        max_r_rs = np.nanmax(dist_image_plane) / R_SUN_KM
        maxxy = float(np.ceil(max_r_rs))
        minxy = -maxxy

    y_out_array = np.array(y_rad_out, copy=True)
    x_out_array = np.array(x_rad_out, copy=True)

    y_out_array[y_out_array < 0.1] = np.nan
    x_out_array[x_out_array < 0.1] = np.nan

    yvalues = np.flip(y_out_array, axis=0)
    n = x_out_array.size
    xvalues = np.linspace(minxy, maxxy, n)

    # Build figure
    fig = plt.figure(constrained_layout=True, figsize=(7.0, 6.0))
    axd = fig.subplot_mosaic(
        [["plotx", "corner"], ["image", "ploty"]],
        gridspec_kw={"width_ratios": [6, 1.3], "height_ratios": [1.3, 6]},
    )

    # Central image
    im = axd["image"].imshow(
        rad_out_orig,
        extent=[minxy, maxxy, minxy, maxxy],
        vmax=plot_max,
        aspect="auto",
        cmap=None if color_map == "no_color" else color_map,
    )

    axd["image"].yaxis.set_label_position("left")
    axd["image"].yaxis.tick_left()
    axd["image"].set_ylabel("(R$_\\odot$)")
    axd["image"].set_xlabel("(R$_\\odot$)")

    # Annotation based on filename
    if show_time:
        try:
            annotate_image_name = (
                image_name[0:4] + "-" + image_name[4:6] + "-" + image_name[6:8]
            )
            annotate_image_name_2 = (
                image_name[9:11]
                + ":"
                + image_name[11:13]
                + ":"
                + image_name[13:15]
                + " UT"
            )
        except Exception:
            annotate_image_name = image_name
            annotate_image_name_2 = ""
        axd["image"].annotate(
            annotate_image_name, xy=(0, 0), xytext=(maxxy - 6.3, minxy + 2)
        )
        if annotate_image_name_2:
            axd["image"].annotate(
                annotate_image_name_2,
                xy=(0, 0),
                xytext=(maxxy - 6.2, minxy + 0.8),
                fontsize=8,
            )

    fig.colorbar(
        im,
        orientation="vertical",
        ax=axd["image"],
        label=cb_label,
        location="left",
    )

    # Top panel (x-profile)
    annotate_plotx_name = (
        "max=" + "{:.2f}".format(float(np.nanmax(x_out_array))) + " R$_\\odot$"
    )

    if color_map == "no_color":
        axd["plotx"].plot(xvalues, x_out_array, linewidth=1.0, color="black")
    else:
        cmap = plt.cm.get_cmap(color_map)
        for i in range(0, n - 1):
            axd["plotx"].plot(
                xvalues[i : i + 2],
                x_out_array[i : i + 2],
                color=cmap(x_out_array[i] / plot_max),
            )

    axd["plotx"].get_xaxis().set_visible(False)
    axd["plotx"].yaxis.set_label_position("left")
    axd["plotx"].yaxis.tick_left()
    axd["plotx"].set_ylabel(side_label)
    axd["plotx"].set_yticks([plot_min, plot_min+0.5 * (plot_max - plot_min), plot_max])
    axd["plotx"].set_ylim([plot_min, plot_max])
    if verbose:
        axd["plotx"].annotate(
            annotate_plotx_name,
            xy=(0, 0),
            xytext=(maxxy - 7, plot_max - 2.5),
        )

    # Right panel (y-profile)
    annotate_ploty_name = (
        "max=" + "{:.2f}".format(float(np.nanmax(y_out_array))) + " R$_\\odot$"
    )

    if color_map == "no_color":
        axd["ploty"].plot(yvalues, xvalues, linewidth=1.0, color="black")
    else:
        cmap = plt.cm.get_cmap(color_map)
        for i in range(0, n - 1):
            axd["ploty"].plot(
                yvalues[i : i + 2],
                xvalues[i : i + 2],
                color=cmap(yvalues[i] / plot_max),
            )

    axd["ploty"].get_yaxis().set_visible(False)
    axd["ploty"].set_xlabel(side_label)
    axd["ploty"].set_xticks([plot_min, plot_min+0.5 * (plot_max - plot_min), plot_max])
    axd["ploty"].set_xlim([plot_min, plot_max])
    if verbose:
        axd["ploty"].annotate(
            annotate_ploty_name,
            xy=(0, 0),
            xytext=(0.5, minxy + 0.5),
        )

    # Corner: optional logo
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
        axd["corner"].get_xaxis().set_visible(False)
        axd["corner"].get_yaxis().set_visible(False)
        axd["corner"].set_visible(False)

    plt.savefig(image_name, dpi=300)
    plt.close(fig)
