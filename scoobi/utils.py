from .math_module import xp, _scipy, ensure_np_array
import scoobi.imshows as imshows

import numpy as np
import scipy
from astropy.io import fits
import skimage
import copy
import pickle
from matplotlib.patches import Circle, Rectangle

def make_grid(npix, pixelscale=1, half_shift=False):
    if half_shift:
        y,x = (xp.indices((npix, npix)) - npix//2 + 1/2)*pixelscale
    else:
        y,x = (xp.indices((npix, npix)) - npix//2)*pixelscale
    return x,y

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2].copy()
    else:
        arr_out = xp.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in
    return arr_out

def rotate_arr(arr, rotation, reshape=False, order=3):
    if arr.dtype == complex:
        arr_r = _scipy.ndimage.rotate(xp.real(arr), angle=rotation, reshape=reshape, order=order)
        arr_i = _scipy.ndimage.rotate(xp.imag(arr), angle=rotation, reshape=reshape, order=order)
        
        rotated_arr = arr_r + 1j*arr_i
    else:
        rotated_arr = _scipy.ndimage.rotate(arr, angle=rotation, reshape=reshape, order=order)
    return rotated_arr

def interp_arr(arr, pixelscale, new_pixelscale, order=3):
        Nold = arr.shape[0]
        old_xmax = pixelscale * Nold/2

        x,y = xp.ogrid[-old_xmax:old_xmax-pixelscale:Nold*1j,
                       -old_xmax:old_xmax-pixelscale:Nold*1j]

        Nnew = int(np.ceil(2*old_xmax/new_pixelscale)) - 1
        new_xmax = new_pixelscale * Nnew/2

        newx,newy = xp.mgrid[-new_xmax:new_xmax-new_pixelscale:Nnew*1j,
                             -new_xmax:new_xmax-new_pixelscale:Nnew*1j]

        x0 = x[0,0]
        y0 = y[0,0]
        dx = x[1,0] - x0
        dy = y[0,1] - y0

        ivals = (newx - x0)/dx
        jvals = (newy - y0)/dy

        coords = xp.array([ivals, jvals])

        interped_arr = _scipy.ndimage.map_coordinates(arr, coords, order=order)
        return interped_arr

def lstsq(modes, data):
    """Least-Squares fit of modes to data.

    Parameters
    ----------
    modes : iterable
        modes to fit; sequence of ndarray of shape (m, n)
    data : numpy.ndarray
        data to fit, of shape (m, n)
        place NaN values in data for points to ignore

    Returns
    -------
    numpy.ndarray
        fit coefficients

    """
    mask = xp.isfinite(data)
    data = data[mask]
    modes = xp.asarray(modes)
    modes = modes.reshape((modes.shape[0], -1))  # flatten second dim
    modes = modes[:, mask.ravel()].T  # transpose moves modes to columns, as needed for least squares fit
    c, *_ = xp.linalg.lstsq(modes, data, rcond=None)
    return c

def save_fits(fpath, data, header=None, ow=True, quiet=False):
    if header is not None:
        keys = list(header.keys())
        hdr = fits.Header()
        for i in range(len(header)):
            hdr[keys[i]] = header[keys[i]]
    else: 
        hdr = None
    
    data = ensure_np_array(data)
    
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(str(fpath), overwrite=ow) 
    if not quiet: print('Saved data to: ', str(fpath))

def load_fits(fpath, header=False):
    data = xp.array(fits.getdata(fpath))
    if header:
        hdr = fits.getheader(fpath)
        return data, hdr
    else:
        return data

# functions for saving python objects
def save_pickle(fpath, data, quiet=False):
    out = open(str(fpath), 'wb')
    pickle.dump(data, out)
    out.close()
    if not quiet: print('Saved data to: ', str(fpath))

def load_pickle(fpath):
    infile = open(str(fpath),'rb')
    pkl_data = pickle.load(infile)
    infile.close()
    return pkl_data

def create_fourier_command(x_cpa=10, y_cpa=10, Nact=34):
    # cpa = cycles per aperture
    # max cpa must be Nact/2
    if x_cpa>Nact/2 or y_cpa>Nact/2:
        raise ValueError('The cycles per aperture is too high for the specified number of actuators.')
    y,x = xp.indices((Nact, Nact)) - Nact//2
    fourier_command = xp.cos(2*np.pi*(x_cpa*x + y_cpa*y)/Nact)
    return fourier_command

def measure_center_and_angle(waffle_im, psf_pixelscale_lamD, im_thresh=1e-4, r_thresh=12,
                           verbose=True, 
                           plot=True):
    npsf = waffle_im.shape[0]
    y,x = (xp.indices((npsf, npsf)) - npsf//2)*psf_pixelscale_lamD
    r = xp.sqrt(x**2 + y**2)
    waffle_mask = (waffle_im >im_thresh) * (r>r_thresh)

    centroids = []
    for i in [0,1]:
        for j in [0,1]:
            arr = waffle_im[j*npsf//2:(j+1)*npsf//2, i*npsf//2:(i+1)*npsf//2]
            mask = waffle_mask[j*npsf//2:(j+1)*npsf//2, i*npsf//2:(i+1)*npsf//2]
            cent = np.flip(skimage.measure.centroid(ensure_np_array(mask*arr)))
            cent[0] += i*npsf//2
            cent[1] += j*npsf//2
            centroids.append(cent)
            # print(cent)
            # imshow3(mask, arr, mask*arr, lognorm2=True,
            #         patches1=[Circle(cent, 1, fill=True, color='cyan')])
    centroids.append(centroids[0])
    centroids = np.array(centroids)
    centroids[[2,3]] = centroids[[3,2]]
    if verbose: print('Centroids:\n', centroids)

    if plot: 
        patches = []
        for i in range(4):
            patches.append(Circle(centroids[i], 1, fill=False, color='black'))
        imshows.imshow3(waffle_mask, waffle_im, waffle_mask*waffle_im, lognorm2=True, vmin2=1e-5, patches1=patches)

    mean_angle = 0.0
    for i in range(4):
        angle = np.arctan2(centroids[i+1][1] - centroids[i][1], centroids[i+1][0] - centroids[i][0]) * 180/np.pi
        if angle<0:
            angle += 360
        if 0<angle<90:
            angle = 90-angle
        elif 90<angle<180:
            angle = 180-angle
        elif 180<angle<270:
            angle = 270-angle
        elif 270<angle<360:
            angle = 360-angle
        mean_angle += angle/4
    if verbose: print('Angle: ', mean_angle)

    m1 = (centroids[0][1] - centroids[2][1])/(centroids[0][0] - centroids[2][0])
    m2 = (centroids[1][1] - centroids[3][1])/(centroids[1][0] - centroids[3][0])
    # print(m1,m2)
    b1 = -m1*centroids[0][0] + centroids[0][1]
    b2 =  -m2*centroids[1][0] + centroids[1][1]
    # print(b1,b2)

    # m1*x + b1 = m2*x + b2
    # (m1-m2) * x = b2 - b1
    xc = (b2 - b1) / (m1 - m2)
    yc = m1*xc + b1
    print('Measured center in X: ', xc)
    print('Measured center in Y: ', yc)

    xshift = np.round(npsf/2 - xc)
    yshift = np.round(npsf/2 - yc)
    print('Required shift in X: ', xshift)
    print('Required shift in Y: ', yshift)

    return xshift,yshift,mean_angle

def measure_pixelscale(sin_im, cpa, 
                       dm_diam=10.2, dm_lyot_mag=9.4/9.4, lyot_diam=8.6, 
                       im_thresh=1e-4, r_thresh=20, 
                       verbose=True, plot=True,):
    npsf = sin_im.shape[0]
    y,x = (xp.indices((npsf, npsf)) - npsf//2)
    r = xp.sqrt(x**2 + y**2)
    sin_mask = (sin_im >im_thresh) * (r>r_thresh)
    imshows.imshow2(sin_mask, sin_mask*sin_im)

    centroids = []
    for i in [0,1]:
        arr = sin_im[:, i*npsf//2:(i+1)*npsf//2]
        mask = sin_mask[:, i*npsf//2:(i+1)*npsf//2]
        cent = np.flip(skimage.measure.centroid(ensure_np_array(mask*arr)))
        cent[0] += i*npsf//2
        centroids.append(cent)
        # print(cent)
        # imshow3(mask, arr, mask*arr, lognorm2=True,
        #         patches1=[Circle(cent, 1, fill=True, color='cyan')])
    centroids = np.array(centroids)
    if verbose: print('Centroids:\n', centroids)

    if plot: 
        patches = []
        for i in range(2):
            patches.append(Circle(centroids[i], 1, fill=True, color='black'))
        imshows.imshow3(sin_mask, sin_im, sin_mask*sin_im, lognorm2=True, vmin2=1e-5, patches1=patches)

    sep_pix = np.sqrt((centroids[1][0] - centroids[0][0])**2 + (centroids[1][1] - centroids[0][1])**2)
    pixelscale_lamD = (2*cpa) / sep_pix * lyot_diam/(dm_diam * dm_lyot_mag)
    if verbose: print('Pixelscale:\n', pixelscale_lamD)

    return pixelscale_lamD

