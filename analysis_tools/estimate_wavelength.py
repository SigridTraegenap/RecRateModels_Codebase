#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:04:12 2023

@author: sigridtragenap
code from Hayleigh, Bettina for wavelength estimation
"""

import matplotlib.pyplot as plt
import numpy as np


#from skimage.morphology import disk,erosion
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.special import jn,jv,erf
import scipy.integrate
from scipy.ndimage import map_coordinates,binary_erosion
''' Code from mustela_bh.autocorr_wavelength, just moved the functions to their own script'''

## Fit functions
def gauss(x,a,sd):
	return a*np.exp(-x**2/(2*sd**2))

def gabor(x,sd,alpha):
	return np.exp(-x**2/(2*sd**2))*np.cos(alpha*x)

def expcos(x,var,alpha):
	return np.exp(-abs(x)/(2*var))*np.cos(alpha*x)

def cauchy(x,gamma,alpha):
	return gamma**2/(x**2+gamma**2)*np.cos(alpha*x)

def damped_bessel(x,var,alpha):
	return np.exp(-abs(x)/2./var)*jv(0,alpha*x)

def linfct(x,a,b):
	return a*x+b


def average_over_interp2d(array,conversion_factor_to_mm,nangles):
	"""calculates radial average of interpolated copy of array
	nangles : number of angles on which to interpolate"""
	if array.ndim>2:
		ys,xs = [],[]
		for frame in array:
			y,x = average_over_interp2d(frame,conversion_factor_to_mm,nangles)
			ys.append(y)
			xs.append(x)
		return np.array(ys),np.array(xs)
	else:
		h,w = array.shape
		xnew = np.linspace(-(w//2),w//2,w,endpoint=True)
		ynew = 0*xnew
		coords_new = np.vstack([xnew,ynew])
		angles = np.linspace(0,2*np.pi,nangles)
		spectrum = []
		for phi in angles:
			rotm = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
			this_coord = np.dot(rotm,coords_new)
			## coordinate origin is at upper left corner for this function!
			spectrum.append( map_coordinates(array,np.vstack([this_coord[1,:],\
				this_coord[0,:]])+h//2,order=1) )
		return np.nanmean(np.array(spectrum),axis=0), xnew*conversion_factor_to_mm


def get_autocorr(pattern,rough_patch_size,norm):
	"""calculates autocorrelation of array pattern"""
	hs,ws = pattern.shape
	max_lag = rough_patch_size
	#autocorr = np.zeros((max_lag+1,max_lag+1))
	#for xlag in range(max_lag):
		#for ylag in range(max_lag):
			#autocorr[ylag,xlag] =\
			 # np.nansum(pattern[:hs-ylag,:ws-xlag]*np.conjugate(pattern[ylag:,xlag:]))/pattern[:hs-ylag,:ws-xlag].size

	## wiener-khinchin

	#frm bettina code snippet
	norm=hs*ws
	max_lag=hs//2 # min([hs//2,ws//2]) should use this probably when pattern is not square
	#first normalize pattern to 0 mean and SD 1
	pattern_norm=pattern-np.nanmean(pattern)
	pattern_norm /= np.nanstd(pattern_norm)
	pattern_norm[np.logical_not(np.isfinite(pattern_norm))]=0.0


	fft_spectrum = abs(np.fft.fft2(pattern_norm, s=(2*hs,2*ws)))**2
	autocorr = np.fft.ifft2( fft_spectrum )/norm#[:hs,:ws]
	autocorr = np.fft.fftshift(autocorr)[hs-max_lag:hs+max_lag+1,ws-max_lag:ws+max_lag+1]
	return np.real(autocorr)

def eventAutocorrFitWavelength(events,roi,ump=None,dataset=None,makeFig=True,saveFig=None,clipROI=True):
	''' computes autocorrelation for events and then attempts to fit wavelength (using bessel, gabor, etc)

		Inputs:
			events- event frames, filtered, roi masked
			roi- roi for event frames (including blood vessel mask if desired)
            ump= microns per pixel
			dataset- used to get umperpix (if ump is not provided)
			makeFig- Make figure
			saveFig- filepath to save figure to

		Returns:
			Autocorr-autocorrelations for each event (2d)
			spectrum- radial average of autocorrelations (1d)
			distance- distance in microns from center
			w_bessel-Bessel fit wavelength
			w_gabor- Gabor fit wavelength
			w_cauchy- Cauchy fit wavelength
			area_peak_locs- Location of first trough ([:,0]), and first peak ([:,1])
			area_peak_vals-Correlation values at trough and peak


	'''

	resolution=1
	#local_neighbourhood_size = int(1000/(resolution*2))		## radius of local neighbourhood around seed point in um
	rough_patch_size = int(1500/(resolution))
	#disk_neighbourhood = disk(rough_patch_size)>0
	idy,idx = np.where(roi)
	norm = np.sum(roi[min(idy):max(idy),min(idx):max(idx)])
	events[np.isnan(events)]=0
	nframes=events.shape[0]

	# take window that contains entire ROI but cuts away extra space of frame to minimize influence of ROI size
	if clipROI:
		yroi,xroi = np.where(roi)
		top_roi,bot_roi = np.nanmin(yroi),np.nanmax(yroi)
		left_roi,right_roi = np.nanmin(xroi),np.nanmax(xroi)
		center_roi_x = (right_roi + left_roi)//2
		center_roi_y = (top_roi + bot_roi)//2
		delta_x = right_roi - left_roi
		delta_y = bot_roi - top_roi
		delta = np.min([delta_x,delta_y])
		delta -= delta%2 # make sure that window length is even, easier to handle averaging I think
		events= events[:,center_roi_y-delta//2:center_roi_y+delta//2,center_roi_x-delta//2:center_roi_x+delta//2]
		roi_mask_clipped = roi[center_roi_y-delta//2:center_roi_y+delta//2,center_roi_x-delta//2:center_roi_x+delta//2]


	#calculate autocorrelation for each event
	#print('Calculating autocorrelation')
	autocorr = []
	for frame in events:
		autocorr.append(get_autocorr(frame,rough_patch_size,norm=norm) )
	autocorr = np.array(autocorr)

	spectrum,distance = average_over_interp2d(autocorr,resolution/1000,360)
	distance = distance[0,:]

# 	m = np.ones(distance.shape[0])
# 	m[distance>0.6]=0.3
# 	m[0] = 2
# 	print('Fitting wavelength')
# 	area_peak_locs,area_peak_vals = [],[]
# 	w_gabor,w_bessel,w_cauchy = [],[],[]
# 	for i in range(nframes):
# 		## Spline interpolation
# 		# print("CHECK",distance.shape, spectrum[i,:].shape,\
# 		# np.sum(np.isfinite(distance)),np.sum(np.isfinite(spectrum[i,:])))
# 		tck = interpolate.splrep(distance, spectrum[i,:], k=3)#s=50  w=1./np.abs(spectrum)*m,
# 		distance2 = np.copy(distance)
# 		distance2[1:-1] += np.random.randn(len(distance)-2)*0.001
# 		distance2 = np.sort(distance2)
# 		spectrum2 = interpolate.splev(distance2, tck)

# 		## Location of first peak and trough of corr fct
# 		derivative = interpolate.splev(distance2,tck,der=1)
# 		tck_der = interpolate.splrep(distance, derivative, w=1./np.abs(spectrum[i,:]),\
# 				 s=50, k=3)
# 		peak_locs = interpolate.sproot(tck_der)
# 		# print("peak_locs",peak_locs)
# 		if len(peak_locs)>0:
# 			peak_vals = interpolate.splev(peak_locs,tck)
# 			peak_locs = np.abs(peak_locs)
# 			peak_origin = np.nanargmin(peak_locs)

# 			if (peak_origin+1)<len(peak_locs):
# 				low_right = peak_locs[peak_origin+1]
# 				val_right = peak_vals[low_right==peak_locs][0]
# 			else:
# 				low_right = np.nan
# 				val_right = np.nan
# 			if (peak_origin-1)>0:
# 				low_left = peak_locs[peak_origin-1]
# 				val_left = peak_vals[low_left==peak_locs][0]
# 			else:
# 				low_left = np.nan
# 				val_left = np.nan
# 			first_low = np.nanmean([low_right,low_left])
# 			low_val = np.nanmean([val_right,val_left])

# 			if (peak_origin+2)<len(peak_locs):
# 				peak_right = peak_locs[peak_origin+2]
# 				val_right = peak_vals[peak_right==peak_locs][0]
# 			else:
# 				peak_right = np.nan
# 				val_right = np.nan
# 			if (peak_origin-2)>0:
# 				peak_left = peak_locs[peak_origin-2]
# 				val_left = peak_vals[peak_left==peak_locs][0]
# 			else:
# 				peak_left = np.nan
# 				val_left = np.nan
# 			first_peak = np.nanmean([peak_right,peak_left])
# 			peak_val = np.nanmean([val_right,val_left])

# 			area_peak_locs.append(np.array([first_low,first_peak]))
# 			area_peak_vals.append(np.array([low_val,peak_val]))

# 		else:
# 			peak_vals = [np.nan]
# 			peak_locs = [np.nan]

# 			area_peak_locs.append(np.array([np.nan,np.nan]))
# 			area_peak_vals.append(np.array([np.nan,np.nan]))


# 		## FIT
# 		fit_vals = np.isfinite(spectrum[i,:])#*(spectrum>0)*(distance<0.4)
# 		init_var = np.array([0.006,0.06,0.1,0.25,0.3])#np.array([0.06])#
# 		init_k = np.array([5,7,10])#np.array([10])#
# 		pcov = np.array([[1,1],[1,1]])
# 		popt = np.array([np.nan,np.nan])
# 		try:
# 			for ivar in init_var:
# 				for ik in init_k:
# 					ipopt,ipcov = curve_fit( gabor,distance[fit_vals],spectrum[i,fit_vals],
# 								p0=[ivar,ik])
# 					if np.mean(np.sqrt(np.diag(ipcov)))<np.mean(np.sqrt(np.diag(pcov))):
# 						pcov = ipcov
# 						popt = ipopt



# 		except:
# 			pass
# 		perr = np.sqrt(np.diag(pcov))
# 		w_gabor.append(np.array([2*np.pi/abs(popt[1]),\
# 					2*np.pi/abs(popt[1])**2*perr[1]]))

# 		initvar = 0.15
# 		initk = 10.
# 		pcov3 = np.array([[1,1],[1,1]])
# 		popt3 = np.array([np.nan,np.nan])
# 		try:
# 			popt3,pcov3 = curve_fit( damped_bessel,distance[fit_vals],spectrum[i,fit_vals],\
# 						p0=[initvar,initk])
# 		except:
# 			pass
# 		perr3 = np.sqrt(np.diag(pcov3))
# 		w_bessel.append(np.array([abs(popt3[1]),perr3[1]]))

# 		initgamma = 0.2
# 		initk = 7.
# 		pcov4 = np.array([[1,1],[1,1]])
# 		popt4 = np.array([np.nan,np.nan])
# 		try:
# 			popt4,pcov4 = curve_fit( cauchy,distance[fit_vals],\
# 						spectrum[i,fit_vals],p0=[initgamma,initk])
# 		except:
# 			pass
# 		perr4 = np.sqrt(np.diag(pcov4))
# 		w_cauchy.append(np.array([2*np.pi/abs(popt4[1]),\
# 					2*np.pi/abs(popt4[1])**2*perr4[1]]))

# 	area_peak_locs = np.array(area_peak_locs)
# 	area_peak_vals = np.array(area_peak_vals)
# 	w_gabor = np.array(w_gabor)
# 	w_bessel = np.array(w_bessel)
# 	w_cauchy = np.array(w_cauchy)

# 	print(peak_locs)

	return autocorr,spectrum,distance

def estimate_wavelength(act_flat):
    res_wl=eventAutocorrFitWavelength(act_flat, np.ones_like(act_flat[0]),
                                          )
    autocorr,spectrum,distance = res_wl
    pos_min = np.argmax(-spectrum[:,(spectrum.shape[-1]//2):], axis=1)
    wavelengths = 2*pos_min

    # plt.plot(spectrum[:,(spectrum.shape[-1]//2):].T, color='C1')
    # plt.title('"test peaks HM"')
    # plt.xticks(np.arange(0,20,4),np.arange(0,20,4))
    # plt.show()

    # print("test peaks HM", detect_peaks(spectrum[0]))
    # print("test peaks HM", detect_peaks(-1*spectrum[0]))
    # a=detect_peaks(spectrum[0])
    # print(np.diff(a))
    return wavelengths

def estimate_wavelength_ST(act_flat):
    spectrum=get_radial_acorr(act_flat,
                              np.ones_like(act_flat[0])
                              )
    #autocorr,spectrum,distance = res_wl
    plt.plot(spectrum.T, color='C0')
    plt.xticks(np.arange(0,20,4),np.arange(0,20,4))
    plt.show()

    # res_wl=eventAutocorrFitWavelength(act_flat, np.ones_like(act_flat[0]),
    #                                       )
    # autocorr,spectrum,distance = res_wl
    # plt.plot(spectrum[:,(spectrum.shape[-1]//2):].T, color='C1')
    # plt.show()

    #pos_min = np.argmax(-spectrum[:(spectrum.shape[-1]//2)], axis=1)



    pos_min= detect_peaks(-1*spectrum)
    # if len(pos_min)<100:
    #     plt.imshow(act_flat[0])
    #     plt.show()

    wavelengths = 2*pos_min

    return wavelengths


def get_radial_acorr(data_flat, roi):
    Npats=data_flat.shape[0]
    Nx, Ny = roi.shape
    res_profiles=[]
    for idx in range(Npats):
        pattern=data_flat[idx].copy()
        CC=cross_correlate_masked(pattern, pattern, roi, roi, mode='same', axes=(-2, -1),
                                   overlap_ratio=0.2)

        r=radial_profile(CC, [Ny//2,Nx//2])
        res_profiles.append(r)

    return np.asarray(res_profiles)

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = np.round(r)
    r = r.astype('int')

    mask=~np.isnan(data)
    tbin = np.bincount(r[mask].ravel(), data[mask].ravel())
    nr = np.bincount(r[mask].ravel())
    radialprofile = tbin / nr
    return radialprofile

from functools import partial
import scipy.fft as fftmodule
from scipy.fft import next_fast_len
def cross_correlate_masked(arr1, arr2, m1, m2, mode='full', axes=(-2, -1),
                           overlap_ratio=0.3):
    """
    Masked normalized cross-correlation between arrays.

    Parameters
    ----------
    arr1 : ndarray
        First array.
    arr2 : ndarray
        Seconds array. The dimensions of `arr2` along axes that are not
        transformed should be equal to that of `arr1`.
    m1 : ndarray
        Mask of `arr1`. The mask should evaluate to `True`
        (or 1) on valid pixels. `m1` should have the same shape as `arr1`.
    m2 : ndarray
        Mask of `arr2`. The mask should evaluate to `True`
        (or 1) on valid pixels. `m2` should have the same shape as `arr2`.
    mode : {'full', 'same'}, optional
        'full':
            This returns the convolution at each point of overlap. At
            the end-points of the convolution, the signals do not overlap
            completely, and boundary effects may be seen.
        'same':
            The output is the same size as `arr1`, centered with respect
            to the `‘full’` output. Boundary effects are less prominent.
    axes : tuple of ints, optional
        Axes along which to compute the cross-correlation.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images.

    Returns
    -------
    out : ndarray
        Masked normalized cross-correlation.

    Raises
    ------
    ValueError : if correlation `mode` is not valid, or array dimensions along
        non-transformation axes are not equal.

    References
    ----------
    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`
    """
    if mode not in {'full', 'same'}:
        raise ValueError(f"Correlation mode '{mode}' is not valid.")

    fixed_image = np.asarray(arr1)
    moving_image = np.asarray(arr2)
    float_dtype = _supported_float_type(
        (fixed_image.dtype, moving_image.dtype)
    )
    if float_dtype.kind == 'c':
        raise ValueError("complex-valued arr1, arr2 are not supported")

    fixed_image = fixed_image.astype(float_dtype)
    fixed_mask = np.array(m1, dtype=bool)
    moving_image = moving_image.astype(float_dtype)
    moving_mask = np.array(m2, dtype=bool)
    eps = np.finfo(float_dtype).eps

    # Array dimensions along non-transformation axes should be equal.
    all_axes = set(range(fixed_image.ndim))
    for axis in (all_axes - set(axes)):
        if fixed_image.shape[axis] != moving_image.shape[axis]:
            raise ValueError(
                f'Array shapes along non-transformation axes should be '
                f'equal, but dimensions along axis {axis} are not.')

    # Determine final size along transformation axes
    # Note that it might be faster to compute Fourier transform in a slightly
    # larger shape (`fast_shape`). Then, after all fourier transforms are done,
    # we slice back to`final_shape` using `final_slice`.
    final_shape = list(arr1.shape)
    for axis in axes:
        final_shape[axis] = fixed_image.shape[axis] + \
            moving_image.shape[axis] - 1
    final_shape = tuple(final_shape)
    final_slice = tuple([slice(0, int(sz)) for sz in final_shape])

    # Extent transform axes to the next fast length (i.e. multiple of 3, 5, or
    # 7)
    fast_shape = tuple([next_fast_len(final_shape[ax]) for ax in axes])

    # We use the new scipy.fft because they allow leaving the transform axes
    # unchanged which was not possible with scipy.fftpack's
    # fftn/ifftn in older versions of SciPy.
    # E.g. arr shape (2, 3, 7), transform along axes (0, 1) with shape (4, 4)
    # results in arr_fft shape (4, 4, 7)
    fft = partial(fftmodule.fftn, s=fast_shape, axes=axes)
    _ifft = partial(fftmodule.ifftn, s=fast_shape, axes=axes)

    def ifft(x):
        return _ifft(x).real

    fixed_image[np.logical_not(fixed_mask)] = 0.0
    moving_image[np.logical_not(moving_mask)] = 0.0

    # N-dimensional analog to rotation by 180deg is flip over all relevant axes.
    # See [1] for discussion.
    rotated_moving_image = _flip(moving_image, axes=axes)
    rotated_moving_mask = _flip(moving_mask, axes=axes)

    fixed_fft = fft(fixed_image)
    rotated_moving_fft = fft(rotated_moving_image)
    fixed_mask_fft = fft(fixed_mask.astype(float_dtype))
    rotated_moving_mask_fft = fft(rotated_moving_mask.astype(float_dtype))

    # Calculate overlap of masks at every point in the convolution.
    # Locations with high overlap should not be taken into account.
    number_overlap_masked_px = ifft(rotated_moving_mask_fft * fixed_mask_fft)
    number_overlap_masked_px[:] = np.round(number_overlap_masked_px)
    number_overlap_masked_px[:] = np.fmax(number_overlap_masked_px, eps)
    masked_correlated_fixed_fft = ifft(rotated_moving_mask_fft * fixed_fft)
    masked_correlated_rotated_moving_fft = ifft(
        fixed_mask_fft * rotated_moving_fft)

    numerator = ifft(rotated_moving_fft * fixed_fft)
    numerator -= masked_correlated_fixed_fft * \
        masked_correlated_rotated_moving_fft / number_overlap_masked_px

    fixed_squared_fft = fft(np.square(fixed_image))
    fixed_denom = ifft(rotated_moving_mask_fft * fixed_squared_fft)
    fixed_denom -= np.square(masked_correlated_fixed_fft) / \
        number_overlap_masked_px
    fixed_denom[:] = np.fmax(fixed_denom, 0.0)

    rotated_moving_squared_fft = fft(np.square(rotated_moving_image))
    moving_denom = ifft(fixed_mask_fft * rotated_moving_squared_fft)
    moving_denom -= np.square(masked_correlated_rotated_moving_fft) / \
        number_overlap_masked_px
    moving_denom[:] = np.fmax(moving_denom, 0.0)

    denom = np.sqrt(fixed_denom * moving_denom)

    # Slice back to expected convolution shape.
    numerator = numerator[final_slice]
    denom = denom[final_slice]
    number_overlap_masked_px = number_overlap_masked_px[final_slice]

    if mode == 'same':
        _centering = partial(_centered,
                             newshape=fixed_image.shape, axes=axes)
        denom = _centering(denom)
        numerator = _centering(numerator)
        number_overlap_masked_px = _centering(number_overlap_masked_px)

    # Pixels where `denom` is very small will introduce large
    # numbers after division. To get around this problem,
    # we zero-out problematic pixels.
    tol = 1e3 * eps * np.max(np.abs(denom), axis=axes, keepdims=True)
    nonzero_indices = denom > tol

    # explicitly set out dtype for compatibility with SciPy < 1.4, where
    # fftmodule will be numpy.fft which always uses float64 dtype.
    out = np.zeros_like(denom, dtype=float_dtype)
    out[nonzero_indices] = numerator[nonzero_indices] / denom[nonzero_indices]
    np.clip(out, a_min=-1, a_max=1, out=out)

    # Apply overlap ratio threshold
    number_px_threshold = overlap_ratio * np.max(number_overlap_masked_px,
                                                 axis=axes, keepdims=True)
    out[number_overlap_masked_px < number_px_threshold] = 0.0

    return out

def _centered(arr, newshape, axes):
    """ Return the center `newshape` portion of `arr`, leaving axes not
    in `axes` untouched. """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)

    slices = [slice(None, None)] * arr.ndim

    for ax in axes:
        startind = (currshape[ax] - newshape[ax]) // 2
        endind = startind + newshape[ax]
        slices[ax] = slice(startind, endind)

    return arr[tuple(slices)]


def _flip(arr, axes=None):
    """ Reverse array over many axes. Generalization of arr[::-1] for many
    dimensions. If `axes` is `None`, flip along all axes. """
    if axes is None:
        reverse = [slice(None, None, -1)] * arr.ndim
    else:
        reverse = [slice(None, None, None)] * arr.ndim
        for axis in axes:
            reverse[axis] = slice(None, None, -1)

    return arr[tuple(reverse)]

new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,      # np.float128 ; doesn't exist on windows
    'G': np.complex128,   # np.complex256 ; doesn't exist on windows
}

def _supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.

    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.

    Parameters
    ----------
    input_dtype : np.dtype or tuple of np.dtype
        The input dtype. If a tuple of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `np.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.

    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
    """
    if isinstance(input_dtype, tuple):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    #print(x.ndim)
    if x.ndim>1:
        res=[]
        for single_curve in x:
            dp=detect_peaks(single_curve, mph, mpd, threshold, edge,
                             kpsh, valley)
            try:
                res.append(dp[0])
            except:
                print(dp)
                plt.plot(single_curve);plt.show()
        return np.array(res)

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
     ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
             ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height

    if valley:
        if ind.size and mph is not None:
            ind = ind[x[ind] <= mph]
    else:
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])


    return ind
