import matplotlib.pyplot as plt
import numpy as np
import core

from support import import_data
from typing import List, Optional, Tuple, Literal
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable  # if you use it elsewhere
from astropy import units as u, constants as c

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.size": 9
})


def _create_figure_core(
    image_data: np.ndarray,
    *,
    image_name: Optional[str] = None,
    data_type: Optional[str] = None,
) -> None:
    """
    Render a 2D image array as a grayscale PNG with no axes or margins.

    Parameters
    ----------
    image_data : np.ndarray
        2D array representing the image to display.
    image_name : str, optional
        Output filename (e.g., "image.png"). If None, defaults to "Test.png".
    data_type : {"stereo_dif", "stereo", "noise", "forward", None}, optional
        Determines default scaling and field-of-view handling for specific
        instruments or synthetic data types.

        - "stereo_dif", "stereo":  vmin=-20, vmax=20, extent [-16, 16] Rs
        - "noise":                  vmin=-15, vmax=15, extent [-16, 16] Rs
        - "forward":                extent [-32, 32] Rs (synthetic models)
        - None:                     automatic scaling
    """

    # Default filename
    if image_name is None:
        image_name = "Test.png"

    # Create borderless figure
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    fig.set_size_inches(15, 15)

    # Set defaults based on type
    if data_type in ("stereo", "stereo_dif"):
        vmin, vmax = -20, 20
        extent = [-16, 16, -16, 16]
    elif data_type == "noise":
        vmin, vmax = -15, 15
        extent = [-16, 16, -16, 16]
    elif data_type == "forward":
        vmin, vmax = None, None
        extent = [-32, 32, -32, 32]
    else:
        vmin, vmax = None, None
        extent = None

    # Plot
    if extent is None:
        ax.imshow(image_data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    else:
        ax.imshow(
            image_data,
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )

    plt.savefig(image_name, format="png", dpi=300)
    plt.close(fig)



# create figures
def create_figure(
    file_list: List[str],
    *,
    data_type: Optional[str] = None,
    use_mask: bool = True,
    use_cdelt: bool = False,
    subtract_base_image: bool = False,
    base_file_list: Optional[List[str]] = None,
    image_name: Optional[str] = None,
) -> None:
    """
    High-level wrapper: loads tB/pB images from FITS files and writes
    two PNG images ("tB" and "pB") using `_create_figure_core`.

    Parameters
    ----------
    file_list : list of str
        Paths to input FITS files. Passed to `import_data`.
    data_type : str, optional
        Instrument/data category (e.g., "stereo", "noise"). Passed through to
        `_create_figure_core` for correct display scaling.
    use_mask : bool, default True
        Whether to apply instrument masks when loading data.
    use_cdelt : bool, default True
        Whether to use CDELT-based coordinate scaling.
    subtract_base_image : bool, default False
        Whether to subtract a base image (for running-difference).
    base_file_list : list of str, optional
        Files used as base for subtraction if enabled.
    image_name : str, optional
        Base name for output files. If None, defaults to `data_type`.
    """

    # Load data from FITS through your existing pipeline
    tB_data, pB_data, tB_hdr, pB_hdr = import_data(
        file_list,
        data_type=data_type,
        use_mask=use_mask,
        use_cdelt=use_cdelt,
        subtract_base_image=subtract_base_image,
        base_file_list=base_file_list,
    )

    # Pick a base name if none given
    if image_name is None:
        image_name = data_type if data_type is not None else "figure"

    # Output filenames
    pB_name = f"pB_{image_name}.png"
    tB_name = f"tB_{image_name}.png"

    # Produce the PNGs
    _create_figure_core(pB_data, image_name=pB_name, data_type=data_type)
    _create_figure_core(tB_data, image_name=tB_name, data_type=data_type)


R_SUN_KM: float = c.R_sun.to(u.kilometer).value
AggregatorType = Literal["mean", "std", "med", "max", "min", "sum"]
SolutionType   = Literal["plus", "minus"]

def _make_triple_plot_data(
    data: np.ndarray,
    aggregator: AggregatorType,
    *,
    minvalue: Optional[float] = None,
    maxvalue: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate a 2D array along rows and columns, within a value range.

    For each row and column in the input 2D array, this function selects all
    values within [minvalue, maxvalue] and applies the requested aggregation
    function (e.g., mean, max, sum). It returns the aggregated profiles along
    the two axes, together with the number of contributing pixels in each row
    and column.

    Parameters
    ----------
    data : np.ndarray
        2D input data array of shape (ny, nx).
    aggregator : {"mean", "std", "med", "max", "min", "sum"}
        Aggregation function to apply along each row and column.
    minvalue : float, optional
        Lower bound on values to include. If None, uses np.nanmin(data).
    maxvalue : float, optional
        Upper bound on values to include. If None, uses np.nanmax(data).

    Returns
    -------
    y_profile : np.ndarray
        1D array of length ny with aggregated values along each row.
    x_profile : np.ndarray
        1D array of length nx with aggregated values along each column.
    y_counts : np.ndarray
        1D array of length ny with the number of data points used in each row.
    x_counts : np.ndarray
        1D array of length nx with the number of data points used in each column.
    """
    if data.ndim != 2:
        raise ValueError("data must be a 2D array")

    ny, nx = data.shape

    if minvalue is None:
        minvalue = np.nanmin(data)
    if maxvalue is None:
        maxvalue = np.nanmax(data)

    # Map aggregator string to numpy function
    agg_map = {
        "mean": np.mean,
        "std":  np.std,
        "med":  np.median,
        "max":  np.max,
        "min":  np.min,
        "sum":  np.sum,
    }

    if aggregator not in agg_map:
        raise ValueError(f"Invalid aggregator '{aggregator}'. "
                         f"Must be one of {list(agg_map.keys())}.")

    agg_func = agg_map[aggregator]

    # --- Row-wise aggregation (y) ---
    y_profile: list[float] = []
    y_counts: list[int] = []

    for i in range(ny):
        row = data[i, :]
        mask = (row >= minvalue) & (row <= maxvalue) & np.isfinite(row)
        selected = row[mask]

        if selected.size == 0:
            # No valid points in this row; use 0 as in the original implementation
            out_value = 0.0
            count = 0
        else:
            out_value = float(agg_func(selected))
            count = selected.size

        y_profile.append(out_value)
        y_counts.append(count)

    # --- Column-wise aggregation (x) ---
    x_profile: list[float] = []
    x_counts: list[int] = []

    for j in range(nx):
        col = data[:, j]
        mask = (col >= minvalue) & (col <= maxvalue) & np.isfinite(col)
        selected = col[mask]

        if selected.size == 0:
            out_value = 0.0
            count = 0
        else:
            out_value = float(agg_func(selected))
            count = selected.size

        x_profile.append(out_value)
        x_counts.append(count)

    return (
        np.asarray(y_profile, dtype=float),
        np.asarray(x_profile, dtype=float),
        np.asarray(y_counts, dtype=int),
        np.asarray(x_counts, dtype=int),
    )


def create_triple_stereo_plot_work(
    tB: np.ndarray,
    pB: np.ndarray,
    dist_image_plane: np.ndarray,  # impact parameter map (km), same shape as tB/pB
    dist_obs_to_source: float,     # observer–Sun distance (km), e.g. 1 AU
    *,
    solution: SolutionType = "plus",   # "plus" → foreground, "minus" → background
    image_name: Optional[str] = None,
    artificial_max: Optional[float] = None,
) -> None:
    """
    Create a triple-panel plot using polarization-ratio LOS distances
    (point-source approximation).

    The function:
    1. Uses `core.radial_position_ps` to compute LOS distances from tB/pB.
    2. Selects either the foreground ("plus") or background ("minus") solution.
    3. Constructs a LOS "depth" map in solar radii.
    4. Derives 1D aggregated profiles along x and y.
    5. Produces a figure with:
       - central image of the LOS map,
       - top panel: horizontal LOS profile,
       - right panel: vertical LOS profile.

    Parameters
    ----------
    tB, pB : np.ndarray
        2D arrays of total and polarized brightness, respectively.
    dist_image_plane : np.ndarray
        2D array of projected distance from Sun centre in the image plane (km).
        Must have the same shape as `tB` and `pB`.
    dist_obs_to_source : float
        Distance from observer to Sun (km), e.g. 1 AU in km.
    solution : {"plus", "minus"}, keyword-only
        Which polarization-ratio solution to use:
        - "plus"  → foreground LOS distance (l_plus)
        - "minus" → background LOS distance (l_minus)
    image_name : str, optional
        Output filename for the resulting PNG. If None, defaults to
        "triple_plot_<solution>.png".
    artificial_max : float, optional
        Maximum LOS distance (in R_sun) used for color scaling and
        profile plotting. If None, it is estimated from the data
        (95th percentile of finite values).
    """

    if tB.shape != pB.shape or tB.shape != dist_image_plane.shape:
        raise ValueError("tB, pB, and dist_image_plane must have the same shape")

    if image_name is None:
        image_name = f"triple_plot_{solution}.png"


    # 1. Get LOS distances from polarization ratio

    out = core.radial_position_ps(tB, pB, dist_image_plane, dist_obs_to_source)

    if len(out) == 4:
        r_plus_km, r_minus_km, l_plus_km, l_minus_km = out
    else:
        (r_plus_km,
         r_minus_km,
         l_plus_km,
         l_minus_km,
         tau_plus,
         tau_minus,
         x_plus,
         x_minus) = out

    l_plus_km = np.asarray(l_plus_km, dtype=float)
    l_minus_km = np.asarray(l_minus_km, dtype=float)

    if solution == "plus":
        l_use_km = l_plus_km
    elif solution == "minus":
        l_use_km = l_minus_km
    else:
        raise ValueError("solution must be 'plus' or 'minus'")


    # 2. Build a LOS "depth" map in solar radii
    #    rad_out = R_obs - l_use (in R_sun)

    sun_obs_dist_rs = dist_obs_to_source / R_SUN_KM  # observer distance in R_sun

    l_use_rs = l_use_km / R_SUN_KM
    rad_out = sun_obs_dist_rs - l_use_rs  # in R_sun

    # Clean obvious garbage
    bad_mask = (
        ~np.isfinite(rad_out)
        | (rad_out < 0)
        | (rad_out > sun_obs_dist_rs * 2.0)
    )
    rad_out[bad_mask] = np.nan

    # Determine plotting upper limit
    if artificial_max is None:
        finite_vals = rad_out[np.isfinite(rad_out)]
        if finite_vals.size > 0:
            artificial_max = float(np.nanpercentile(finite_vals, 95))
        else:
            artificial_max = 10.0  # fallback
    rad_out[rad_out > artificial_max] = np.nan

    # Keep a copy before smoothing for the image
    rad_out_orig = rad_out.copy()

    # 3. Smooth the LOS map
    sigma = [1.5, 1.5]  # (sigma_y, sigma_x)
    rad_out_smooth = gaussian_filter(rad_out, sigma=sigma, mode="constant")

    # 4. Extract slices with _make_triple_plot_data
    y_profile, x_profile, y_counts, x_counts = _make_triple_plot_data(
        rad_out_smooth,
        aggregator="max",
        minvalue=0.0,
        maxvalue=artificial_max,
    )

    # Smooth 1D profiles with Savitzky–Golay filter
    window_size = 35
    poly_order = 3

    # Ensure window_size is valid for x_profile
    if window_size >= len(x_profile):
        window_size = max(3, (len(x_profile) // 2) * 2 + 1)
    x_profile_smooth = savgol_filter(x_profile, window_size, poly_order, mode="interp")

    # Ensure window_size is valid for y_profile
    if window_size >= len(y_profile):
        window_size = max(3, (len(y_profile) // 2) * 2 + 1)
    y_profile_smooth = savgol_filter(y_profile, window_size, poly_order, mode="interp")

    # Mask tiny values (mostly background)
    x_out_array = x_profile_smooth[::-1].copy()
    y_out_array = y_profile_smooth.copy()
    x_out_array[x_out_array < 0.1] = np.nan
    y_out_array[y_out_array < 0.1] = np.nan

    # 5. Determine POS extent from dist_image_plane (in R_sun)
    rpos_rs = dist_image_plane / R_SUN_KM
    rpos_max = float(np.nanmax(rpos_rs))
    margin = 0.05 * rpos_max
    minxy = -(rpos_max + margin)
    maxxy = +(rpos_max + margin)

    # X-axis (horizontal POS) for x-profile
    xvalues = np.linspace(minxy, maxxy, x_out_array.size)
    # Y-axis (vertical POS) for y-profile; flip so top of array is top of plot
    #yvalues = np.flip(y_out_array, axis=0)
    yvalues = y_out_array
    ypos = np.linspace(minxy, maxxy, yvalues.size)


    # 6. Triple plot: image + horizontal + vertical slices

    fig = plt.figure(constrained_layout=True, figsize=(7.0, 6))
    axd = fig.subplot_mosaic(
        [["plotx", "."],
         ["image", "ploty"]],
        gridspec_kw={
            "width_ratios": [6, 1.3],
            "height_ratios": [1.3, 6],
        },
    )

    # Main image: LOS depth map in R_sun
    im = axd["image"].imshow(
        rad_out_orig,
        extent=[minxy, maxxy, minxy, maxxy],
        vmin=0.0,
        vmax=artificial_max,
        cmap="gray",   # dark near Sun, bright near observer
        origin="lower",
        aspect="auto",
    )

    axd["image"].yaxis.set_label_position("left")
    axd["image"].yaxis.tick_left()
    axd["image"].set_ylabel("POS / R$_\\odot$")
    axd["image"].set_xlabel("POS / R$_\\odot$")

    fig.colorbar(
        im,
        orientation="vertical",
        ax=axd["image"],
        label=f"LOS (R$_\\odot$), {solution} solution",
        location="left",
    )

    # Top plot: horizontal LOS profile vs POS x
    axd["plotx"].plot(xvalues, x_out_array, linewidth=1.0, color="black")
    axd["plotx"].get_xaxis().set_visible(False)
    axd["plotx"].yaxis.set_label_position("left")
    axd["plotx"].yaxis.tick_left()
    axd["plotx"].set_ylabel("LOS (R$_\\odot$)")
    axd["plotx"].set_ylim([0.0, artificial_max])
    axd["plotx"].set_yticks([0.0, artificial_max / 2.0, artificial_max])

    # Right plot: vertical LOS profile vs POS y
    axd["ploty"].plot(yvalues, ypos, linewidth=1.0, color="black")
    axd["ploty"].get_yaxis().set_visible(False)
    axd["ploty"].set_xlabel("LOS (R$_\\odot$)")
    axd["ploty"].set_xlim([0.0, artificial_max])
    axd["ploty"].set_xticks([0.0, artificial_max / 2.0, artificial_max])

    plt.suptitle(f"Polarization-ratio LOS ({solution} solution)")
    plt.savefig(image_name, dpi=300)
    plt.close(fig)



# creates the triple plot
def create_triple_stereo_plot(file_list, 
                data_type=None, 
                use_mask=1, 
                use_cdelt=0,
                subtract_base_image=0,
                base_file_list=None, 
                output_method='ps',
                image_name=None,
                plot_max=None,
                plot_min=None,
                apply_filter='Gauss',
                gauss_filter_sig_x=1.5,
                gauss_filter_sig_y=1.5,
                side_panel_filter='savgol',
                savgol_window_size=25,
                savgol_poly_order=2,
                show_time=True,
                verbose=True,
                color_map='YlOrRd',
                include_name=False):

    if image_name==None:
        image_name="TRIPLE plot.png"


    #from vapor.reconstruction import radial_position
    distance_array=radial_position(file_list, 
                data_type=data_type, 
                use_mask=use_mask, 
                use_cdelt=use_cdelt,
                subtract_base_image=subtract_base_image,
                base_file_list=base_file_list, 
                output_method='ps'
                )


    solar_radii_in_km = 695660
    sun_obs_dist = 1.495979e8# * u.kilometer   # km
    sun_obs_dist_rs=sun_obs_dist/solar_radii_in_km

    rad_out=sun_obs_dist_rs-(distance_array/solar_radii_in_km)
    rad_out[rad_out>100]=np.nan
    rad_out_orig=rad_out.copy()
    
    # Apply gaussian filter
    if apply_filter=='Gauss':
        sigma = [gauss_filter_sig_y, gauss_filter_sig_x]
        rad_out = sp.ndimage.filters.gaussian_filter(rad_out, sigma, mode='constant')

    # if min and max values are not supplied then make data min and max
    if plot_min==None: plot_min=np.min(rad_out)
    if plot_max==None: plot_max=np.max(rad_out)

    tripledata=make_triple_plot_data(rad_out, aggregator='max', minvalue=plot_min, maxvalue=plot_max)
    
    x_rad_out=tripledata[1]
    y_rad_out=tripledata[0]
    
    # to make more fine grain reduce window size and increase polynomial
    if side_panel_filter=='savgol':
        '''
        appl the Savitzky-Golay filter to the side and top bar
        for smooth keep as default, for coarse
        use 
        savgol_window_size=5
        savgol_poly_order=3
        '''
        x_rad_out = savgol_filter(x_rad_out, savgol_window_size, savgol_poly_order)
        y_rad_out = savgol_filter(y_rad_out, savgol_window_size, savgol_poly_order)

    if verbose:
        smooth_annotate_image_name=image_name[9:11]+':'+image_name[11:13]+':'+image_name[13:15]
        print("x max: "+ smooth_annotate_image_name +" " +str(np.max(x_rad_out)))
        print("y max: "+ smooth_annotate_image_name +" " +str(np.max(y_rad_out)))

    
    def plot_triple2():

        # annotate seems to be in relation to ploty
        annotate_image_name=image_name[0:4]+'-'+image_name[4:6]+'-'+image_name[6:8]
        annotate_image_name_2=image_name[9:11]+':'+image_name[11:13]+':'+image_name[13:15]+' UT'
        annotate_plotx_name=("max="+"{:.2f}".format(round(np.max(x_rad_out),2))+" R$_\odot$")
        annotate_ploty_name=("max="+"{:.2f}".format(round(np.max(y_rad_out),2))+" R$_\odot$")        
        #plt.imshow(image_data, cmap='gray',axes=axes, origin="lower")
        if data_type=='stereo_dif':
            minxy=-16
            maxxy=16

        if data_type=='stereo':
            minxy=-16
            maxxy=16
    
        elif data_type=='noise':
            minxy=-16
            maxxy=16
    
        elif data_type=='forward':
            minxy=-32
            maxxy=32

        y_out_array=np.array(y_rad_out)
        x_out_array=np.array(x_rad_out)
        
        y_out_array[y_out_array<0.1]=np.nan
        x_out_array[x_out_array<0.1]=np.nan

        yvalues=np.flip(y_out_array, axis=0)
        xvalues=np.arange(minxy,maxxy, (maxxy-minxy)/np.size(x_out_array))

        # below create the image

        fig=plt.figure(constrained_layout=True, figsize=(7.0, 6))
        axd=fig.subplot_mosaic([['plotx','corner'],
                                ['image','ploty']],
                                gridspec_kw={'width_ratios':[6, 1.3],
                                            'height_ratios':[1.3, 6]})
        axd['ploty'].invert_xaxis
        axd['plotx'].sharex(axd['image'])
        axd['ploty'].sharey(axd['image'])
        #axd['corner'].sharey(axd['ploty'])
        #axd['corner'].sharex(axd['plotx'])
        im=axd['image'].imshow(rad_out_orig, 
                               extent=[minxy,maxxy,minxy,maxxy], 
                               vmax=plot_max,
                               aspect='auto',
                               cmap=color_map)

        axd['image'].yaxis.set_label_position("left")
        axd['image'].yaxis.tick_left()
        #axd['image'].set_ylabel('Plane of Sky (R$_\odot$)')
        #axd['image'].set_xlabel('Plane of Sky (R$_\odot$)')

        axd['image'].set_ylabel('Plane of Sky (R$_\odot$)')
        axd['image'].set_xlabel('Plane of Sky (R$_\odot$)')

        if show_time:
            axd['image'].annotate(annotate_image_name, xy=(0, 0), xytext=(maxxy-6.3, minxy+2))
            axd['image'].annotate(annotate_image_name_2, xy=(0, 0), xytext=(maxxy-6.2, minxy+0.8), fontsize=8)

        fig.colorbar(im, orientation='vertical', ax=axd['image'], label='Line of Sight (R$_\odot$)', location='left')

        if color_map == 'no_color':
            axd['plotx'].plot(xvalues,x_out_array, linewidth=1.0, color='black')
        else:
            dim=rad_out.shape
            n=dim[0]
            x=np.linspace(minxy,maxxy,n)
            y=x_out_array
            # Segment plot and color depending on T
            s = 1 # Segment length
            cmap = plt.cm.get_cmap(color_map)
            for i in range(0,n-s,s):
                axd['plotx'].plot(x[i:i+s+1],y[i:i+s+1],color=cmap(x_out_array[i]/plot_max))

        axd['plotx'].get_xaxis().set_visible(False)
        axd['plotx'].yaxis.set_label_position("left")
        axd['plotx'].yaxis.tick_left()
        axd['plotx'].set_ylabel('LOS (R$_\odot$)')
        axd['plotx'].set_yticks([plot_min, 0.5*(plot_max-plot_min), plot_max])
        axd['plotx'].set_ylim([plot_min, plot_max])
        if verbose:
            axd['plotx'].annotate(annotate_plotx_name, xy=(0, 0), xytext=(maxxy-7, plot_max-2.5))
        
        
        if color_map == 'no_color':
            axd['ploty'].plot(yvalues,xvalues, linewidth=1.0, color='black')
        else:
            dim=rad_out.shape
            n=dim[0]
            x=np.linspace(minxy,maxxy,n)
            y=xvalues
            # Segment plot and color depending on T
            s = 1 # Segment length
            cmap = plt.cm.get_cmap(color_map)
            for i in range(0,n-s,s):
                axd['ploty'].plot(yvalues[i:i+s+1],y[i:i+s+1],color=cmap(yvalues[i]/plot_max))

        axd['ploty'].get_yaxis().set_visible(False)
        axd['ploty'].set_xlabel('LOS (R$_\odot$)')
        axd['ploty'].set_xticks([plot_min, 0.5*(plot_max-plot_min), plot_max])
        axd['ploty'].set_xlim([plot_min, plot_max])
        if verbose:
            axd['ploty'].annotate(annotate_ploty_name, xy=(0, 0), xytext=(0.5, minxy+0.5))
        
        if include_name:
            py_file_path = os.path.dirname(__file__)
            #os.path.join(py_file_path, 'images', 'VAPOR.png')
            img = mpimg.imread(os.path.join(py_file_path, 'images', 'VAPOR.png'))
            axd['corner'].imshow(img)
            axd['corner'].get_xaxis().set_visible(False)
            axd['corner'].get_yaxis().set_visible(False)
            axd['corner'].axis('off')
            #axd['corner'].set_visible(True)
        else: 
            axd['corner'].get_xaxis().set_visible(False)
            axd['corner'].get_yaxis().set_visible(False)
            axd['corner'].set_visible(False)


        plt.savefig(image_name,  dpi=300)
        plt.close()


    plot_triple2()




def plot_depth_map(D, title="LOS depth (Sun dark, observer bright)"):
    """
    Plot a depth map D (0–1) with a dark-to-bright grayscale.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(D, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, label="Normalized depth")
    ax.set_title(title)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    plt.tight_layout()
    return fig, ax



