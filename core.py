import warnings
import numpy as np

from vapor.support import add_one
from vapor.support import import_data

from astropy.io import fits
from astropy import constants as c
from astropy import units as u


# calculate constants
solar_radii=c.R_sun
solar_radii_in_km=solar_radii.to(u.kilometer)
solar_radii_in_km=solar_radii_in_km.value

sun_obs_dist = c.au# * u.kilometer   # km
sun_obs_dist=sun_obs_dist.to(u.kilometer)
sun_obs_dist=sun_obs_dist.value

sun_obs_dist_rs=sun_obs_dist/solar_radii_in_km

def create_distance_map(file_list, 
                data_type=None, 
                use_mask=1, 
                use_cdelt=0,
                subtract_base_image=0,
                base_file_list=None
                ):
    
    # prep the data
    tB_hdul_data, pB_hdul_data, tB_hdul_header, pB_hdul_header = import_data(file_list, 
                data_type=data_type, 
                use_mask=use_mask, 
                use_cdelt=use_cdelt,
                subtract_base_image=subtract_base_image,
                base_file_list=base_file_list,
                )
    # create distance array - for inputing pixel distance from center
    spatial_plane_distance_array = np.full_like(pB_hdul_data, 0)
    thompson_distance_array = np.full_like(pB_hdul_data, 0)


    if use_cdelt:
        x_center=( pB_hdul_header['CRPIX1'])
        y_center=( pB_hdul_header['CRPIX2'])

        if data_type=='noise_gate_data':
            x_pixel_size_km = solar_radii_in_km * pB_hdul_header['CDELT1']# * u.kilometer # rad/pixel
            y_pixel_size_km = solar_radii_in_km * pB_hdul_header['CDELT2']# * u.kilometer # rad/pixel
        else:
            x_pixel_size_km = solar_radii_in_km * pB_hdul_header['CDELT1']/(pB_hdul_header['RSUN'])# * u.kilometer # rad/pixel
            y_pixel_size_km = solar_radii_in_km * pB_hdul_header['CDELT2']/(pB_hdul_header['RSUN'])# * u.kilometer # rad/pixel
        
        xdim=pB_hdul_header['NAXIS1']
        ydim=pB_hdul_header['NAXIS2']
    
    else:
        xdim=spatial_plane_distance_array.shape[0]
        ydim=spatial_plane_distance_array.shape[1]
                          
        x_center=( xdim/2 )
        y_center=( ydim/2 )
    
        fov_solar_radii = 64
        number_pixels = xdim
        x_pixel_size_km = fov_solar_radii * solar_radii_in_km / number_pixels
        y_pixel_size_km = fov_solar_radii * solar_radii_in_km / number_pixels

    for xStep in range(xdim):
        for yStep in range(ydim):
            spatial_plane_distance_array[xStep,yStep]=( ( (xStep-x_center)*x_pixel_size_km )**2 + 
                                      ( (yStep-y_center)*y_pixel_size_km )**2 )**(1/2)

    epsilon_array=np.arctan2(spatial_plane_distance_array,sun_obs_dist)

    first_part=sun_obs_dist*(1-np.cos(epsilon_array))
    second_part=sun_obs_dist*np.sin(epsilon_array)
    thompson_distance_array=(first_part**2+second_part**2)**0.5

    # use y1_array or z1_array
    # see difference between the thompson_distance_array and the distance_array
    #print('min=',np.min(spatial_plane_distance_array),np.min(thompson_distance_array))
    #print('max=',np.max(spatial_plane_distance_array),np.max(thompson_distance_array))

    return spatial_plane_distance_array



def radial_position(file_list, 
                data_type='stereo', 
                use_mask=1, 
                use_cdelt=0,
                subtract_base_image=0,
                base_file_list=None,
                dist_obs_to_source=sun_obs_dist, 
                output_method='ps'):
    '''
    Calculates the radial position of an object in a white light image.

    If one considers the two signs of the sqrt, one gets the two angles defined as +/- each other 
    because sin(-x)=-sin(x). As there are both foreground and background solutions the scattering angle chi is defined in
    two ways below, chi_minus is greater than 90, thus 180-chi_plus

    
    Input
    -----
    pB [array, float] = Polarized brightness
    B [array, float]= Brightness
    dist_image_plane [array, float] = array of distances that should match dimensions of pB and B
    dist_obs_to_source [float] = is the distance from the observer to the Sun
    '''


     # prep the data
    tB, pB, tB_hdul_header, pB_hdul_header = import_data(file_list, 
                data_type=data_type, 
                use_mask=use_mask, 
                use_cdelt=use_cdelt,
                subtract_base_image=subtract_base_image,
                base_file_list=base_file_list
                )   

 
    dist_image_plane=create_distance_map(file_list, 
                data_type=data_type, 
                use_mask=use_mask, 
                use_cdelt=use_cdelt,
                subtract_base_image=subtract_base_image,
                base_file_list=base_file_list
                )



    if np.ndim(pB) != np.ndim(tB):
        raise Exception("pB and B arrays are of different dimensions")
    
    if np.ndim(dist_image_plane) != np.ndim(tB):
        raise Exception("the array of distances in the image plane and pB/B are of different dimensions")
  
    # calculate elongation angle
    epsilon = np.arctan2( dist_image_plane, dist_obs_to_source )


    # points of intersection with Thomson sphere
    thomson_intersection = dist_image_plane*np.cos(epsilon)
        
    # calculate polarization ratio
    pol=pB/tB
    pol[pol>1]=1
    pol[pol<0]=0
    PR=( 1-(pol) ) / ( 1+(pol) )

    # calculate scattering angles
    scattering_function =1

    
    # scattered
    if output_method=='scattered':
        chi_plus = np.arcsin(scattering_function*(1-PR)**0.5)
        chi_minus = np.pi - chi_plus

        # calculate xi angle
        xi_plus = epsilon - chi_plus + np.pi/2
        xi_minus = epsilon - chi_minus + np.pi/2.

        # distance from Earth to localized density
        dist_plus = dist_obs_to_source*( np.sin(np.pi/2. - xi_plus )/
                                        np.sin(np.pi - chi_plus )) 
        dist_minus = dist_obs_to_source*( np.sin(np.pi/2. - xi_minus)/
                                         np.sin(np.pi - chi_minus)) 

        # calculate radial position
        radial_position_plus = dist_obs_to_source*( np.sin(epsilon) /
                                                np.sin(np.pi - chi_plus) ) 
        radial_position_minus = dist_obs_to_source*( np.sin(epsilon) /
                                                 np.sin(np.pi - chi_minus) ) 
        return_value=dist_plus
    
    
    # point source
    if output_method=='ps':
        #chi_plus_ps = np.arcsin(PR**0.5)
        chi_plus_ps = np.arccos(PR**0.5)
        chi_minus_ps = np.pi - chi_plus_ps
    
        # calculate xi angle
        xi_plus_ps = epsilon - chi_plus_ps + np.pi/2
        xi_minus_ps = epsilon - chi_minus_ps + np.pi/2.

        # distance from Earth to localized density
        dist_plus_ps = dist_obs_to_source*( np.sin(np.pi/2. - xi_plus_ps )/
                                        np.sin(np.pi - chi_plus_ps )) 
        dist_minus_ps = dist_obs_to_source*( np.sin(np.pi/2. - xi_minus_ps)/
                                         np.sin(np.pi - chi_minus_ps)) 
    
        # calculate radial position
        radial_position_plus_ps = dist_obs_to_source*( np.sin(epsilon) /
                                                np.sin(np.pi - chi_plus_ps) ) 
        radial_position_minus_ps = dist_obs_to_source*( np.sin(epsilon) /
                                                 np.sin(np.pi - chi_minus_ps) ) 

        return_value=dist_plus_ps

    return return_value