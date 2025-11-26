from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import support 
import core

from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable

def show_fits_image(filepath):
    """
    Load a FITS file and display the primary HDU image using matplotlib.
    """
    # Open FITS file
    with fits.open(filepath) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    
    if data is None:
        raise ValueError("No image data found in primary HDU.")

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap="gray", origin="lower")
    plt.colorbar(label="Pixel Value")
    plt.title(f"{filepath}\nDATE-OBS: {header.get('DATE-OBS', 'N/A')}")
    plt.tight_layout()
    plt.show()


def show_two_fits_image(filepath1, filepath2, title1="image 1", title2="image 2"):
    """
    Load a FITS file and display the primary HDU image using matplotlib.
    """
    # Open FITS file
    with fits.open(filepath1) as hdul:
        data1 = hdul[0].data
        header1 = hdul[0].header
    
    if data1 is None:
        raise ValueError("No image data found in primary HDU.")

    with fits.open(filepath2) as hdul:
        data2 = hdul[0].data
        header2 = hdul[0].header
    
    if data2 is None:
        raise ValueError("No image data found in primary HDU.")


    # Plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(title1)
    plt.imshow(data1, cmap="gray")
    plt.colorbar(shrink=0.7)
    plt.axis("on")

    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.imshow(data2, cmap="gray")
    plt.colorbar(shrink=0.7)
    plt.axis("on")

    plt.tight_layout()
    plt.show()


def display_fits(fits_image, display=False):
    """
    Load and optionally display a FITS image using matplotlib.

    Parameters
    ----------
    fits_image : str
        Path to the FITS file to be loaded.
    display : bool, optional
        If True (default), the image will be displayed with matplotlib.
        If False, only the data will be returned.

    Returns
    -------
    data : ndarray
        2D array containing the image data from the primary HDU of the FITS file.

    Notes
    -----
    - The function assumes that the image data is stored in the primary HDU
      (Header Data Unit) of the FITS file.
    - A grayscale colormap is used by default, with the origin set to 'lower'
      so the image is displayed in the standard orientation (bottom-left corner
      is (0,0)).
    - The function also prints information about the FITS file structure
      (extensions, types, sizes) for reference.
    """
    # Open the FITS file
    with fits.open(fits_image) as hdul:
        hdul.info()  # Print structure of the FITS file
        data = hdul[0].data  # Extract image data from the primary HDU

    # Optionally display the image
    if display:
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap='gray', origin='lower')
        plt.colorbar(label="Pixel value")
        plt.title("FITS Image")
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")
        plt.show()

    return data


def display_img(
    img: np.ndarray,
    display: bool = True,
    title: str | None = None
) -> np.ndarray:
    """
    Optionally display a 2D numpy array as an image using matplotlib,
    with an optional title.

    Parameters
    ----------
    img : ndarray
        2D numpy array containing image data.
    display : bool, optional
        If True (default), the image will be displayed with matplotlib.
        If False, only the array will be returned.
    title : str, optional
        Title to display above the image. If None, defaults to "Image".

    Returns
    -------
    img : ndarray
        The input image array (returned unchanged).

    Notes
    -----
    - A grayscale colormap is used by default, with the origin set to 'lower'
      so the image is displayed in the standard orientation (bottom-left corner
      is (0,0)).
    - A colorbar is shown when display=True.
    """
    if display:
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap='gray', origin='lower')
        plt.colorbar(label="Pixel value")
        plt.title(title if title is not None else "Image")
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")
        plt.show()


#===prep specific figures
def data_type_figure_prep(data_type):
    #plt.imshow(image_data, cmap='gray',axes=axes, origin="lower")
    if data_type=='stereo_dif':
        out_array=[-16, 16, -20, 20]
        imgplot = plt.imshow(image_data, cmap='gray', vmin=-20, vmax=20, origin="lower",axes=axes)


    if data_type=='stereo':
        out_array=[-16, 16, -20, 20]
        minxy=-16
        maxxy=16
        imgplot = plt.imshow(image_data, cmap='gray', vmin=-20, vmax=20, origin="lower",axes=axes)
    
    elif data_type=='noise':
        out_array=[-16, 16, -15, 15]
        minxy=-16
        maxxy=16
        imgplot = plt.imshow(image_data, cmap='gray', vmin=-15, vmax=15, origin="lower",axes=axes)
    
    elif data_type=='forward':
        out_array=[-32, 32, None, None]
        minxy=-32
        maxxy=32
        imgplot = plt.imshow(image_data, extent=[minxy,maxxy,minxy,maxxy], cmap='gray', origin="lower",axes=axes)

    else:
        out_array=[None, None, None, None]
        imgplot = plt.imshow(image_data, cmap='gray', origin="lower",axes=axes) 
    
    # TODO: consider retuning axes
    return


#===Make PNG
def create_figure_core(image_data, image_name = None, data_type=None):

    #e.g. makepng(imageOfInterest, "LASCOtest1.png")
    
    if image_name == None :
        image_name = "Test.png"
    
    figure = plt.figure(frameon=False)
    axes = plt.Axes(figure, [0., 0., 1., 1.])
    axes.set_axis_off()
    figure.add_axes(axes)
    figure.set_size_inches(15, 15)
    
    #plt.imshow(image_data, cmap='gray',axes=axes, origin="lower")
    if data_type=='stereo_dif':
        minxy=-16
        maxxy=16
        imgplot = plt.imshow(image_data, cmap='gray', vmin=-20, vmax=20, origin="lower",axes=axes)


    if data_type=='stereo':
        minxy=-16
        maxxy=16
        imgplot = plt.imshow(image_data, cmap='gray', vmin=-20, vmax=20, origin="lower",axes=axes)
    
    elif data_type=='noise':
        minxy=-16
        maxxy=16
        imgplot = plt.imshow(image_data, cmap='gray', vmin=-15, vmax=15, origin="lower",axes=axes)
    
    elif data_type=='forward':
        minxy=-32
        maxxy=32
        imgplot = plt.imshow(image_data, extent=[minxy,maxxy,minxy,maxxy], cmap='gray', origin="lower",axes=axes)

    else:
        imgplot = plt.imshow(image_data, cmap='gray', origin="lower",axes=axes)

    plt.savefig(image_name, format='png')


# create figures
def create_figure(file_list, 
                data_type=None, 
                use_mask=1, 
                use_cdelt=1,
                subtract_base_image=0,
                base_file_list=None,
                image_name=None):



    tB_hdul_data, pB_hdul_data, tB_hdul_header, pB_hdul_header = import_data(file_list, 
                data_type=data_type, 
                use_mask=use_mask, 
                use_cdelt=use_cdelt,
                subtract_base_image=subtract_base_image,
                base_file_list=base_file_list
                )

    if image_name==None:
        image_name=data_type


    pB_name="pB - "+image_name+".png"
    tB_name="tB - "+image_name+".png"
    create_figure_core(pB_hdul_data, pB_name, data_type)
    create_figure_core(tB_hdul_data, tB_name, data_type)




def make_triple_plot_data(data, aggregator, minvalue=None, maxvalue=None):
    '''
    This sums or creates another average in a direction across an input data array.
    '''
    dim=data.shape

    if minvalue==None:
        minvalue=np.min(data)
    if maxvalue==None:
        maxvalue=np.max(data)
    
    #calc y dim
    yPlot=[]
    yPlot_data_points=[]
    for iStep in np.arange(dim[0]):
        # extract row
        row=data[iStep,:]    
        outrow=row[np.where((row >= minvalue) & (row <= maxvalue))]
        if np.size(outrow)==0:
            outrow=0

        #TODO: consider making a dictionary thay can be read with the if statement.
        if aggregator=='mean':
            out_value=np.mean(outrow)

        if aggregator=='std':
            out_value=np.std(outrow)

        if aggregator=='med':
            out_value=np.median(outrow)

        if aggregator=='max':
            out_value=np.max(outrow)

        if aggregator=='min':
            out_value=np.min(outrow)
            
        if aggregator=='sum':
            out_value=np.sum(outrow)
        
        yPlot.append(out_value)
        yPlot_data_points.append(np.size(outrow))
        #print(yPlot)

    #calc x dim
    xPlot=[]
    xPlot_data_points=[]
    for jStep in np.arange(dim[1]):
        # extract column
        col=data[:,jStep]
        outcol=col[(col >= minvalue) & (col <= maxvalue)]
        if np.size(outcol)==0:
            outcol=0
        
        if aggregator=='mean':
            out_value=np.mean(outcol)

        if aggregator=='std':
            out_value=np.std(outcol)

        if aggregator=='med':
            out_value=np.median(outcol)

        if aggregator=='max':
            out_value=np.max(outcol)

        if aggregator=='min':
            out_value=np.min(outcol)
            
        if aggregator=='sum':    
            out_value=np.sum(outcol)
        
        xPlot.append(out_value)
        #print(np.size(out_value))
        xPlot_data_points.append(np.size(outcol))
    return yPlot, xPlot, yPlot_data_points, xPlot_data_points




R_SUN_KM = 695660.0

def create_triple_stereo_plot(
    tB,
    pB,
    dist_image_plane,      # impact parameter map (km), same shape as tB/pB
    dist_obs_to_source,    # observer–Sun distance (km), e.g. 1 AU
    solution='plus',       # 'plus' → foreground, 'minus' → background
    image_name=None,
    artificial_max=None,
):
    """
    Triple plot using polarization-ratio LOS distances (point-source approx).

    Parameters
    ----------
    tB, pB : 2D arrays
        Total and polarized brightness.
    dist_image_plane : 2D array
        Projected distance from Sun centre in the image plane (km).
    dist_obs_to_source : float
        Distance from observer to Sun (km), e.g. 1 AU in km.
    solution : {'plus', 'minus'}
        Which polarization-ratio solution to use:
        - 'plus'  → foreground (l_plus)
        - 'minus' → background (l_minus)
    image_name : str or None
        Output filename for the PNG.
    artificial_max : float or None
        Max LOS distance (in R_sun) for color scaling and slices.
        If None, a value is estimated from the data.
    """

    if image_name is None:
        image_name = f"TRIPLE_plot_{solution}.png"

    # ------------------------------------------------------------------
    # 1. Get LOS distances from polarization ratio
    # ------------------------------------------------------------------
    out = core.radial_position_ps(tB, pB, dist_image_plane, dist_obs_to_source)

    if len(out) == 4:
        r_plus_km, r_minus_km, l_plus_km, l_minus_km = out
    else:
        r_plus_km, r_minus_km, l_plus_km, l_minus_km, tau_plus, tau_minus, x_plus, x_minus = out

    l_plus_km  = np.asarray(l_plus_km, dtype=float)
    l_minus_km = np.asarray(l_minus_km, dtype=float)

    # Choose which LOS solution to use
    if solution == 'plus':
        l_use_km = l_plus_km
    elif solution == 'minus':
        l_use_km = l_minus_km
    else:
        raise ValueError("solution must be 'plus' or 'minus'")

    # ------------------------------------------------------------------
    # 2. Build a LOS "depth" map in solar radii
    #    rad_out = R_obs - l_use (in R_sun)
    #    → small near Sun, large near observer.
    # ------------------------------------------------------------------
    sun_obs_dist_rs = dist_obs_to_source / R_SUN_KM   # R_obs in R_sun

    l_use_rs = l_use_km / R_SUN_KM
    rad_out = sun_obs_dist_rs - l_use_rs   # in R_sun

    # Clean obvious garbage
    bad = ~np.isfinite(rad_out) | (rad_out < 0) | (rad_out > sun_obs_dist_rs * 2.0)
    rad_out[bad] = np.nan

    # Clip very large values for plotting
    if artificial_max is None:
        finite_vals = rad_out[np.isfinite(rad_out)]
        if finite_vals.size > 0:
            artificial_max = np.nanpercentile(finite_vals, 95)
        else:
            artificial_max = 10.0  # fallback
    rad_out[rad_out > artificial_max] = np.nan

    # Keep a copy before smoothing for the image
    rad_out_orig = rad_out.copy()

    # ------------------------------------------------------------------
    # 3. Smooth the map a bit
    # ------------------------------------------------------------------
    sigma_y = 1.5
    sigma_x = 1.5
    sigma = [sigma_y, sigma_x]
    rad_out_smooth = gaussian_filter(rad_out, sigma=sigma, mode='constant')

    # ------------------------------------------------------------------
    # 4. Extract slices with your make_triple_plot_data
    # ------------------------------------------------------------------
    aggregator = 'max'
    tripledata = make_triple_plot_data(
        rad_out_smooth,
        aggregator=aggregator,
        minvalue=0.0,
        maxvalue=artificial_max,
    )

    y_rad_out = np.array(tripledata[0])  # rows
    x_rad_out = np.array(tripledata[1])  # columns

    # Smooth the 1D slices (ensure window_size is valid)
    window_size = 35
    poly_order = 3
    if window_size >= len(x_rad_out):
        window_size = max(3, len(x_rad_out) // 2 * 2 + 1)
    x_rad_out = savgol_filter(x_rad_out, window_size, poly_order, mode='interp')

    if window_size >= len(y_rad_out):
        window_size = max(3, len(y_rad_out) // 2 * 2 + 1)
    y_rad_out = savgol_filter(y_rad_out, window_size, poly_order, mode='interp')

    # Mask tiny values in the slices (mostly background)
    y_out_array = y_rad_out.copy()
    x_out_array = x_rad_out.copy()
    y_out_array[y_out_array < 0.1] = np.nan
    x_out_array[x_out_array < 0.1] = np.nan

    # ------------------------------------------------------------------
    # 5. Determine POS extent from dist_image_plane (in R_sun)
    # ------------------------------------------------------------------
    rpos_rs = dist_image_plane / R_SUN_KM
    rpos_max = np.nanmax(rpos_rs)
    margin = 0.05 * rpos_max
    minxy = -(rpos_max + margin)
    maxxy = +(rpos_max + margin)

    # x-values (horizontal POS) corresponding to x_out_array
    xvalues = np.linspace(minxy, maxxy, x_out_array.size)
    # y-values (vertical POS) corresponding to y_out_array (flip so top matches top of image)
    yvalues = np.flip(y_out_array, axis=0)

    # ------------------------------------------------------------------
    # 6. Triple plot: image + horizontal + vertical slices
    # ------------------------------------------------------------------
    fig = plt.figure(constrained_layout=True, figsize=(7.0, 6))
    axd = fig.subplot_mosaic(
        [['plotx', '.'],
         ['image', 'ploty']],
        gridspec_kw={'width_ratios': [6, 1.3],
                     'height_ratios': [1.3, 6]}
    )

    # Main image: rad_out_orig in R_sun
    im = axd['image'].imshow(
        rad_out_orig,
        extent=[minxy, maxxy, minxy, maxxy],
        vmin=0.0,
        vmax=artificial_max,
        cmap='gray',   # dark near Sun (0), white near observer (~R_obs)
        origin='lower',
        aspect='auto',
    )

    axd['image'].yaxis.set_label_position("left")
    axd['image'].yaxis.tick_left()
    axd['image'].set_ylabel('POS / R$_\\odot$')
    axd['image'].set_xlabel('POS / R$_\\odot$')

    fig.colorbar(
        im,
        orientation='vertical',
        ax=axd['image'],
        label=f'LOS (R$_\\odot$), {solution} solution',
        location='left',
    )

    # Top plot: horizontal slice vs POS x
    axd['plotx'].plot(xvalues, x_out_array, linewidth=1.0, color='black')
    axd['plotx'].get_xaxis().set_visible(False)
    axd['plotx'].yaxis.set_label_position("left")
    axd['plotx'].yaxis.tick_left()
    axd['plotx'].set_ylabel('LOS (R$_\\odot$)')
    axd['plotx'].set_ylim([0.0, artificial_max])
    axd['plotx'].set_yticks([0, artificial_max / 2.0, artificial_max])

    # Right plot: vertical slice vs POS y
    axd['ploty'].plot(yvalues, np.linspace(minxy, maxxy, yvalues.size), linewidth=1.0, color='black')
    axd['ploty'].get_yaxis().set_visible(False)
    axd['ploty'].set_xlabel('LOS (R$_\\odot$)')
    axd['ploty'].set_xlim([0.0, artificial_max])
    axd['ploty'].set_xticks([0, artificial_max / 2.0, artificial_max])

    plt.suptitle(f"Polarization-ratio LOS ({solution} solution)")
    plt.savefig(image_name, dpi=300)
    plt.close(fig)




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
