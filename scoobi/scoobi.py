from .math_module import xp, _scipy, ensure_np_array
import scoobi.utils as utils

import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
import poppy
import time
import copy
import os
from pathlib import Path

from scoobpy import utils as scoob_utils
import purepyindi
import purepyindi2
from magpyx.utils import ImageStream

import scoobi
module_path = Path(os.path.dirname(os.path.abspath(scoobi.__file__)))

def move_psf(x_pos, y_pos, client):
    client.wait_for_properties(['stagepiezo.stagepupil_x_pos', 'stagepiezo.stagepupil_y_pos'])
    scoob_utils.move_relative(client, 'stagepiezo.stagepupil_x_pos', x_pos)
    time.sleep(0.25)
    scoob_utils.move_relative(client, 'stagepiezo.stagepupil_y_pos', y_pos)
    time.sleep(0.25)

def move_block_in(client, delay=2):
    client.wait_for_properties(['stagelinear.presetName'])
    client['stagelinear.presetName.block_in'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def move_block_out(client, delay=2):
    client.wait_for_properties(['stagelinear.presetName'])
    client['stagelinear.presetName.block_out'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def set_zwo_roi(xc, yc, npix, client, delay=0.25):
    # update roi parameters
    client.wait_for_properties(['camsci.roi_region_x', 'camsci.roi_region_y', 'camsci.roi_region_h' ,'camsci.roi_region_w', 'camsci.roi_set'])
    client['camsci.roi_region_x.target'] = xc
    client['camsci.roi_region_y.target'] = yc
    client['camsci.roi_region_h.target'] = npix
    client['camsci.roi_region_w.target'] = npix
    time.sleep(delay)
    client['camsci.roi_set.request'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def set_kilo_mod_amp(amp, client, delay=0.25):
    client.wait_for_properties(['kiloModulator.amp'])
    client['kiloModulator.amp.target'] = amp
    time.sleep(delay)

def set_kilo_mod_rate(freq, client, delay=0.25):
    client.wait_for_properties(['kiloModulator.frequency'])
    client['kiloModulator.frequency.target'] = freq
    time.sleep(delay)

def start_kilo_mod(client, delay=0.25):
    client.wait_for_properties(['kiloModulator.trigger', 'kiloModulator.modulating'])
    client['kiloModulator.trigger.toggle'] = purepyindi.SwitchState.OFF
    time.sleep(delay)
    client['kiloModulator.modulating.toggle'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def stop_kilo_mod(client, delay=0.25):
    client.wait_for_properties(['kiloModulator.trigger', 'kiloModulator.modulating', 'kiloModulator.zero'])
    client['kiloModulator.modulating.toggle'] = purepyindi.SwitchState.OFF
    time.sleep(delay)
    client['kiloModulator.trigger.toggle'] = purepyindi.SwitchState.ON
    time.sleep(delay)
    client['kiloModulator.zero.request'] = purepyindi.SwitchState.ON

# define more functions for moving the fold mirror, using the tip tilt mirror, and the polarizers

class SCOOBI():

    def __init__(self, 
                 dm_channel,
                 scicam_channel=None,
                 locam_channel=None,
                 dm_ref=np.zeros((34,34)),
                 npsf=150,
                ):
        self.wavelength_c = 633e-9*u.m
        
        self.SCICAM = ImageStream(scicam_channel) if scicam_channel is not None else None
        self.LOCAM = ImageStream(locam_channel) if locam_channel is not None else None
        self.DM = scoob_utils.connect_to_dmshmim(channel=dm_channel) # channel used for writing to DM
        # self.DM_WFE = scoob_utils.connect_to_dmshmim(channel=wfe_channel) if wfe_channel is not None else None
        self.DMT = scoob_utils.connect_to_dmshmim(channel='dm00disp') # the total shared memory image
        self.dm_delay = 0.1

        # Init all DM settings
        self.Nact = 34
        self.Nacts = 951 # accounting for the bad actuator
        self.dm_shape = (self.Nact,self.Nact)
        self.act_spacing = 300e-6*u.m
        self.dm_active_diam = 10.2*u.mm
        self.dm_full_diam = 11.1*u.mm
        self.full_stroke = 1.5e-6*u.m
        self.dm_ref = dm_ref
        self.dm_gain = 1
        self.reset_dm()
        
        self.bad_acts = [(25,21)]
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing.to_value(u.mm)
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask = r<10.5/2
        self.dm_pupil_mask = r<9.1/2

        # Init camera settings
        self.psf_pixelscale = 3.76e-6*u.m/u.pix
        self.psf_pixelscale_lamD = 0.307
        self.nbits = 16
        self.Nframes = 1
        self.Nframes_locam = 1
        self.npsf = npsf
        self.nlocam = 100
        self.x_shift = 0
        self.y_shift = 0
        self.x_shift_locam = 0
        self.y_shift_locam = 0

        self.att = 1
        self.texp = 1
        self.gain = 1
        self.texp_locam = 1
        self.gain_locam = 1

        self.Imax_ref = 1
        self.texp_ref = 1
        self.gain_ref = 1
        self.texp_locam_ref = 1
        self.gain_locam_ref = 1
        self.att_ref = 1

        self.df = None
        self.subtract_dark = False

        self.df_locam = None
        self.subtract_dark_locam = False

        self.return_ni = False
        self.return_ni_locam = False
        

    def set_zwo_exp_time(self, exp_time, client, delay=0.25):
        client.wait_for_properties(['camsci.exptime'])
        client['camsci.exptime.target'] = exp_time
        time.sleep(delay)
        self.texp = exp_time
        print(f'Set the ZWO exposure time to {self.texp:.2e}s')

    def set_fib_atten(self, value, client, delay=0.1):
        client['fiberatten.atten.target'] = value
        time.sleep(delay)
        self.att = value
        print(f'Set the fiber attenuation to {value:.1f}')

    def set_zwo_emgain(self, gain, client, delay=0.1):
        client.wait_for_properties(['camsci.emgain'])
        client['camsci.emgain.target'] = gain
        time.sleep(delay)
        self.gain = gain
        print(f'Set the ZWO gain setting to {gain:.1f}')

    def set_locam_exp_time(self, exp_time, delay=0.1):
        # client['nsvcam.exptime.target'] = value
        self.texp_locam = exp_time
        print(f'Set the LOCAM exposure time to {self.texp:.2e}s')

    def set_locam_emgain(self, gain, delay=0.1):
        # client.wait_for_properties(['nsvcam.emgain'])
        # client['nsvcam.emgain.target'] = gain
        # time.sleep(delay)
        self.gain_locam = gain
        print(f'Set the LOCAM gain setting to {gain:.1f}')

    def zero_dm(self):
        self.DM.write(np.zeros(self.dm_shape))
        time.sleep(self.dm_delay)
    
    def reset_dm(self):
        self.DM.write(ensure_np_array(self.dm_ref))
        time.sleep(self.dm_delay)
    
    def set_dm(self, dm_command):
        self.DM.write(ensure_np_array(dm_command)*1e6)
        time.sleep(self.dm_delay)
    
    def add_dm(self, dm_command):
        dm_state = ensure_np_array(self.get_dm())
        self.DM.write( 1e6*(dm_state + ensure_np_array(dm_command)) )
        time.sleep(self.dm_delay)
               
    def get_dm(self):
        return xp.array(self.DM.grab_latest())/1e6
    
    def close_dm(self):
        self.DM.close()

    def normalize(self, image):
        image_ni = image/self.Imax_ref
        image_ni *= (self.texp_ref/self.texp)
        image_ni *= 10**((self.att-self.att_ref)/10)
        image_ni *= 10**(-self.gain/20 * 0.1) / 10**(-self.gain_ref/20 * 0.1)
        # gain ~ 10^(-gain_setting/20 * 0.1)
        return image_ni

    def snap(self, normalize=False, plot=False, vmin=None):
        if self.Nframes>1:
            ims = self.camsci.grab_many(self.Nframes)
            im = np.sum(ims, axis=0)/self.Nframes
        else:
            im = self.camsci.grab_latest()
        
        im = xp.array(im)
        im = _scipy.ndimage.shift(im, (self.y_shift, self.x_shift), order=0)
        im = utils.pad_or_crop(im, self.npsf)

        if self.subtract_dark and self.df is not None:
            im -= self.df
            im[im<0] = 0.0
            
        if self.return_ni:
            im = self.normalize(im)
        
        return im
    
    def normalize_locam(self, image):
        image_ni = image * (self.texp_locam_ref/self.texp_locam)
        image_ni *= 10**((self.att-self.att_ref)/10)
        # image_ni *= 10**(-self.gain_locam/20 * 0.1) / 10**(-self.gain_locam_ref/20 * 0.1)
        return image_ni

    def snap_locam(self):
        im = self.LOCAM.grab_latest()
        im = scipy.ndimage.shift(im, (self.y_shift_locam, self.x_shift_locam), order=0)
        im = utils.pad_or_crop(im, self.nlocam)

        if self.subtract_dark_locam and self.df_locam is not None:
            im -= self.df_locam
            im[im<0] = 0.0
            
        if self.return_ni_locam:
            im = self.normalize_locam(im)

        return im
    
    def stack_locam(self):
        ims = self.LOCAM.grab_many(self.Nframes_locam)
        im = np.sum(ims, axis=0)/self.Nframes_locam
        im = scipy.ndimage.shift(im, (self.y_shift_locam, self.x_shift_locam), order=0)
        im = utils.pad_or_crop(im, self.nlocam)

        if self.subtract_dark_locam and self.df_locam is not None:
            im -= self.df_locam
            im[im<0] = 0.0
            
        if self.return_ni_locam:
            im = self.normalize_locam(im)

        return im
    
# def snap_many(images, Nframes_per_exp, exp_times, gains, plot=False):
#     total_im = 0.0
#     pixel_weights = 0.0
#     for i in range(len(self.exp_times)):
#         self.exp_time = self.exp_times[i]
#         self.Nframes = self.Nframes_per_exp[i]

#         frames = self.CAM.grab_many(self.Nframes)
#         mean_frame = np.sum(frames, axis=0)/self.Nframes
#         mean_frame = _scipy.ndimage.shift(mean_frame, (self.y_shift, self.x_shift), order=0)
#         mean_frame = pad_or_crop(mean_frame, self.npsf)
            
#         pixel_sat_mask = mean_frame > self.sat_thresh

#         if self.subtract_bias is not None:
#             mean_frame -= self.subtract_bias
        
#         pixel_weights += ~pixel_sat_mask
#         normalized_im = mean_frame/self.exp_time 
#         normalized_im[pixel_sat_mask] = 0 # mask out the saturated pixels

#         if plot: 
#             imshows.imshow3(pixel_weights, mean_frame, normalized_im, 
#                             'Pixel Weight Map', 
#                             f'Frame:\nExposure Time = {self.exp_time:.2e}s', 
#                             'Masked Flux Image', 
#                             # lognorm2=True, lognorm3=True,
#                             )
            
#         total_im += normalized_im
        
#     total_im /= pixel_weights

#     return total_im

    
        
        