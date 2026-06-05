#!/usr/bin/python
import numpy as np
import scipy.linalg as la

from .random_Gaussfields import get_random_field

def get_feedforward_conn(feedforward_params):
    N,M = feedforward_params["N,M"]
    eccentricity = feedforward_params["eccentricity"]

    binary_conn = feedforward_params["mixing_binary"]
    if binary_conn:
        radius_conn = feedforward_params["radius_FFconnectivity"]
        sigma_gauss=1
    else:
        sigma_gauss = feedforward_params["gauss_sigma"]
        radius_conn=0

    network_params = {
            'mode'                    : 'short_range',
            'noise_type'              : 'None',
            'sigmax'                  : sigma_gauss,
            'sigmax_sd'               : eccentricity*0.15/0.8,
            'ecc'                     : eccentricity,
            'ecc_sd'                  : 0.13*eccentricity,
            'orientation'             : 0,
            'orientation_sd'          : 1.0,
            'amplitude'               : 0.,  ##ensures that only positive weights
            'inh_factor'              : 1.5,
            'pbc'                     : True,
            'represent_binary'        : binary_conn,
            'radius_connectivity'     : radius_conn,
            }

    ##implement sampled gaussian check

    gauss_b,_ = gaussian_wrap(N, M,
                      network_params,
                      pbc=True,index=None,
                      full_output=True,version=0)

    gauss_2d = np.reshape(gauss_b.T, (N*M,N,M))
    #Normalize so that total sum is 1
    gauss_2d /= np.sum(gauss_2d, axis=0)[None,:,:]
    if feedforward_params["sampling"]:
        K_total = feedforward_params["sampling_K"]
        rng=np.random.RandomState(seed=None)

        empty_idx=(gauss_2d.sum(0)==0)
        #print(empty_idx.shape)
        gauss_2d=rng.binomial(K_total, gauss_2d)*1.
        gauss_2d[empty_idx.flatten(),empty_idx]=1
        w_flat=gauss_2d.reshape(N*M, N*M)
        empty_idx=w_flat.sum(0)==0
        w_flat[empty_idx,empty_idx]=1
        gauss_2d=w_flat.reshape(N*M, N,M)
        gauss_2d /= np.sum(gauss_2d, axis=0)[None,:,:]


    return gauss_2d.reshape(N*M, N*M)


def get_kmax_eigvals(w_rec,network_params):

    Ndim = len(w_rec.shape)
    if Ndim==4:
        ## get estimate of spatial scale of pattern by looking at peak in spectrum of M
        N,M = w_rec.shape[0], w_rec.shape[1]
        w_rec0 = abs(np.fft.fftshift(np.fft.fft2(w_rec[N//2,M//2,:,:])))
        kmax_flat = np.argmax(w_rec0)
        kmax = np.array([abs(kmax_flat%N-N//2),abs(kmax_flat//N-M//2)])
        w_rec = w_rec.reshape(N*M,N*M)
    else:
        kmax='None'

    ## normalize M such that real part of maximal eigenvalue is given by network_params['nonlin_fac']
    all_eigenvals =la.eigvals(w_rec)
    max_eigenval = np.nanmax(np.real(all_eigenvals))
    w_rec = network_params['nonlin_fac']*w_rec/np.real(max_eigenval)
    return w_rec, kmax, all_eigenvals*network_params['nonlin_fac']/np.real(max_eigenval)


def gauss(delta,inh_factor):
    return 1./inh_factor**2*np.exp(-delta/2./inh_factor**2)




def convolve_with_MH(input_rnd,N,M,sigma1=2,sigma2=6,return_real=True,padd=0):
    ''' use convolution with MH to get spatial scale in noisy input'''
    h,w = input_rnd.shape
    x,y = np.meshgrid(np.linspace(-N//2+1,N//2,N+2*padd),np.linspace(-M//2+1,M//2,M+2*padd))
    sig1 = sigma1#2
    sig2 = sigma2#3*sig1
    kern1 = 1./(np.sqrt(np.pi*2)*sig1)**2*np.exp((-x**2-y**2)/2./sig1**2)
    kern2 = 1./(np.sqrt(np.pi*2)*sig2)**2*np.exp((-x**2-y**2)/2./sig2**2)
    diff_gauss = kern1-kern2
    to_smooth = np.pad(input_rnd,padd,'constant')
    input_smo = np.fft.ifft2(np.fft.fft2(diff_gauss)*np.fft.fft2(to_smooth,axes=(0,1)), axes=(0,1))
    input_smo = np.fft.fftshift(input_smo)
    hn,wn = input_smo.shape
    if padd>0:
        input_smo = input_smo[hn//2-h//2:hn//2+h//2,wn//2-w//2:wn//2+w//2]
    if return_real:
        return np.real(input_smo)
    else:
        return input_smo


def noisy_mh(N,M,mode,noise_type,sigmax,sigmax_sd,ecc,ecc_sd,orientation,orientation_sd,\
a1,inh_factor,pbc=True,index=4876,full_output=True,rotate_by=None,conv_params=None):

    coord_x,coord_y= np.meshgrid(np.arange(N),np.arange(M)) #N:x dimension; M:y dim.

    new_index = index
    np.random.seed(index)


    ## periodic boundary conditions
    if pbc:
        deltax = coord_x[:,:,None,None]-coord_x[None,None,:,:]
        deltay = coord_y[:,:,None,None]-coord_y[None,None,:,:]
        absdeltax = np.abs(deltax)
        absdeltay = np.abs(deltay)
        idxx = np.argmin([absdeltax, N-absdeltax],axis=0)
        idxy = np.argmin([absdeltay, M-absdeltay],axis=0)

        deltax = deltax*(1-idxx) + np.sign(deltax)*(absdeltax-N)*idxx
        deltay = deltay*(1-idxy) + np.sign(deltay)*(absdeltay-M)*idxy
    else:
        deltax = coord_x[:,:,None,None]-coord_x[None,None,:,:]
        deltay = coord_y[:,:,None,None]-coord_y[None,None,:,:]




    if mode=='short_range':
        if 'None' in noise_type:
            '''homogeneous mhs'''
            sigmax=conv_params['s1_x']
            sigmay=conv_params['s1_x']
            delta = (deltax)**2/sigmax**2 + (deltay)**2/sigmay**2

            mh1=gauss(delta,1.)/2./np.pi
            if inh_factor is not np.inf:
                mh2=gauss(delta,inh_factor)/2./np.pi
            else:
                mh2 = np.ones_like(mh1)
            anisotropic_mh = ( mh1 - a1*mh2 )
        elif noise_type=='postsyn':
            ## generate spatial scale in x direction
            if conv_params['do_convolution_x']:
                sigmax_noise = convolve_with_MH(np.random.randn(M,N),N,M,sigma1=conv_params['s1_x'],sigma2=conv_params['s2_x'])[:,:,None,None]
            else:
                sigmax_noise = np.random.randn(M,N)[:,:,None,None]
            sigmax_noise = (sigmax + sigmax_noise/np.std(sigmax_noise)*sigmax_sd)
            sigmax_noise[sigmax_noise<0]=0.0

            ## generate eccentricity array
            if conv_params['do_convolution_ecc']:
                ecc_noise = convolve_with_MH(np.random.randn(M,N),N,M,sigma1=conv_params['s1_ecc'],sigma2=conv_params['s2_ecc'])[:,:,None,None]
            else:
                ecc_noise = np.random.randn(M,N)[:,:,None,None]
            ecc_noise = ecc + ecc_noise/np.std(ecc_noise)*ecc_sd

            #rewrite using clip
            ecc_noise[ecc_noise>0.95] = 0.95
            ecc_noise[ecc_noise<0.0] = 0.0

            ## calculate spatial scale in y-direction
            sigmay_noise = sigmax_noise*np.sqrt(1 - ecc_noise**2)

            ## generate array of orientations
            z_noise = np.random.randn(M,N) + 1j*np.random.randn(M,N)
            if conv_params['do_convolution_ori']:
                z_noise = convolve_with_MH(z_noise,N,M,sigma1=conv_params['s1_ori'],sigma2=conv_params['s2_ori'],return_real=False,padd=conv_params['padd'])
            elif conv_params['const_ori']:
                z_noise = np.zeros((M,N),dtype='complex')
            orientation_noise = np.angle(z_noise)*0.5
            orientation_noise = orientation + orientation_noise

            orientation_noise = orientation_noise + np.pi*(orientation_noise<(np.pi/2))
            orientation_noise = orientation_noise - np.pi*(orientation_noise>(np.pi/2))

            cos_noise = np.cos(orientation_noise)[:,:,None,None]
            sin_noise = np.sin(orientation_noise)[:,:,None,None]

            if not full_output:
                return ecc_noise[:,:,0,0],orientation_noise,sigmax_noise[:,:,0,0],sigmay_noise[:,:,0,0]

            delta = (deltax*cos_noise - deltay*sin_noise)**2/sigmax_noise**2 + (deltay*cos_noise + sin_noise*deltax)**2/sigmay_noise**2
            mh1 = gauss(delta,1.)/sigmay_noise/sigmax_noise/2./np.pi
            if inh_factor is not np.inf:
                mh2 = gauss(delta,inh_factor)/sigmay_noise/sigmax_noise/2./np.pi
            else:
                mh2 = np.ones_like(mh1)
            anisotropic_mh = ( mh1 - a1*mh2 )



        w_rec=np.real(anisotropic_mh)

        if conv_params['do_Binomial_sampling']:
            rng=np.random.RandomState(seed=index)
            m1_sampled=rng.binomial(conv_params['K_E'], mh1)*conv_params['w_E']
            m2_sampled=rng.binomial(conv_params['K_I'], a1*mh2)*conv_params['w_I']
            w_rec=m1_sampled+m2_sampled

        if conv_params['represent_binary']:
            radius_true = conv_params['radius_connectivity']
            delta_pure = np.sqrt((deltax)**2 + (deltay)**2)
            w_rec = np.asarray(np.copy(delta_pure<radius_true), dtype='float')

        return w_rec,new_index


def noisy_gaussian(N,M,EI_params,inh_factor, pbc=True,index=4876,
                   full_output=True,rotate_by=None,conv_params=None):

    coord_x,coord_y= np.meshgrid(np.arange(N),np.arange(M)) #N:x dimension; M:y dim.
    new_index = index
    np.random.seed(index)
    mode = EI_params['mode']
    noise_type = EI_params['noise_type']
    sigmax = EI_params['sigmax']
    sigmax_sd = EI_params['sigmax_sd']
    ecc = EI_params['ecc']
    ecc_sd = EI_params['ecc_sd']
    orientation = EI_params['orientation']



    if mode == 'short_range':
        if noise_type=='postsyn':
            ## generate spatial scale in x direction
            if conv_params['do_convolution_x']:
                sigmax_noise = convolve_with_MH(np.random.randn(M,N),N,M,sigma1=conv_params['s1_x'],sigma2=conv_params['s2_x'])[:,:,None,None]
            else:
                sigmax_noise = np.random.randn(M,N)[:,:,None,None]
            sigmax_noise = (sigmax + sigmax_noise / np.std(sigmax_noise) * sigmax_sd)
            sigmax_noise[sigmax_noise<0] = 0.0

            ## generate eccentricity array
            if conv_params['do_convolution_ecc']:
                ecc_noise = convolve_with_MH(np.random.randn(M,N),N,M,sigma1=conv_params['s1_ecc'],sigma2=conv_params['s2_ecc'])[:,:,None,None]
            else:
                ecc_noise = np.random.randn(M,N)[:,:,None,None]
            ecc_noise = ecc + ecc_noise / np.std(ecc_noise) * ecc_sd

            #rewrite using clip
            ecc_noise[ecc_noise>0.95] = 0.95
            ecc_noise[ecc_noise<0.0] = 0.0

            ## calculate spatial scale in y-direction
            sigmay_noise = sigmax_noise * np.sqrt(1 - ecc_noise ** 2)

            ## generate array of orientations
            z_noise = np.random.randn(M,N) + 1j * np.random.randn(M,N)
            if conv_params['do_convolution_ori']:
                z_noise = convolve_with_MH(z_noise,N,M,sigma1=conv_params['s1_ori'],sigma2=conv_params['s2_ori'],return_real=False,padd=conv_params['padd'])
            elif conv_params['const_ori']:
                z_noise = np.zeros((M,N),dtype='complex')
            orientation_noise = np.angle(z_noise) * 0.5
            orientation_noise = orientation + orientation_noise

            orientation_noise = orientation_noise + np.pi * (orientation_noise < (np.pi/2))
            orientation_noise = orientation_noise - np.pi * (orientation_noise > (np.pi/2))

            cos_noise = np.cos(orientation_noise)[:,:,None,None]
            sin_noise = np.sin(orientation_noise)[:,:,None,None]

            if not full_output:
                return ecc_noise[:,:,0,0] , orientation_noise , sigmax_noise[:,:,0,0] , sigmay_noise[:,:,0,0]

        ## periodic boundary conditions
        if pbc:
            deltax = coord_x[:,:,None,None] - coord_x[None,None,:,:]
            deltay = coord_y[:,:,None,None] - coord_y[None,None,:,:]
            absdeltax = np.abs(deltax)
            absdeltay = np.abs(deltay)
            idxx = np.argmin([absdeltax, N - absdeltax] , axis=0)
            idxy = np.argmin([absdeltay, M - absdeltay] , axis=0)

            deltax = deltax * (1-idxx) + np.sign(deltax) * (absdeltax-N) * idxx
            deltay = deltay * (1-idxy) + np.sign(deltay) * (absdeltay-M) * idxy
        else:
            deltax = coord_x[:,:,None,None]-coord_x[None,None,:,:]
            deltay = coord_y[:,:,None,None]-coord_y[None,None,:,:]
        delta = (deltax * cos_noise - deltay * sin_noise) ** 2 / sigmax_noise ** 2 + \
            (deltay * cos_noise + sin_noise * deltax) ** 2 / sigmay_noise ** 2

        W = gauss(delta,inh_factor) / sigmay_noise / sigmax_noise /2./np.pi


        return np.real(W),new_index


def noisy_mh_wrap_old(N,M,mode,noise_type,sigmax,sigmax_sd,ecc,ecc_sd,orientation,
                  orientation_sd,a1,inh_factor,pbc=True,index=4876,
                  full_output=True,version=0):
    rotate_by = None
    s1 = sigmax
    s2 = sigmax*inh_factor
    conv_params = {'do_convolution_ori' : False,'padd' : 0, 's1_ori' : s1, 's2_ori' : s2,
                       'do_convolution_x' : False, 's1_x' : s1, 's2_x' : s2,
                       'do_convolution_ecc' : False, 's1_ecc' : s1, 's2_ecc' : s2,
                       'do_lowpass_ori' : False,'do_lowpass_x' : False,'do_lowpass_ecc' : False,
                       'const_ori' : False,'do_Binomial_sampling': False,}

    return noisy_mh(N,M,mode,noise_type,sigmax,sigmax_sd,ecc,ecc_sd,orientation,\
    orientation_sd,a1,inh_factor,pbc=pbc,index=index,full_output=full_output,\
        rotate_by=rotate_by,conv_params=conv_params)


def noisy_EI_wrap(N,M, EI_params,  pbc=True,index=4876,full_output=True,version=0):
    """EI_params =
    { mode, noise_type, sigmax, sigmax_sd, ecc, ecc_sd, orientation, ori_sd }"""

    rotate_by = None
    inh_factor=EI_params ['inh_factor']
    pbc=EI_params['pbc']
    s1 = EI_params['sigmax']
    s2 = EI_params['sigmax'] * inh_factor
    if 'self_inhibition' in EI_params:
        alpha = EI_params['alpha']
        a_tot = EI_params['a_tot']
    # smoothly varying mh profile
    conv_params = {'do_convolution_ori' : False,
            'padd' : 0, 's1_ori' : s1, 's2_ori' : s2,  # kernel sizes of convolution
                       'do_convolution_x' : False, # change smoothly
                       's1_x' : s1, 's2_x' : s2,
                       'do_convolution_ecc' : False,
                       's1_ecc' : s1, 's2_ecc' : s2,
                       'do_lowpass_ori' : False,
                       'do_lowpass_x' : False, # always false
                       'do_lowpass_ecc' : False, #
                       'const_ori' : False}


    W_ee, inx_ee = noisy_gaussian(N,M,EI_params,1.0,pbc=pbc,index=index,full_output=full_output,rotate_by=rotate_by,conv_params=conv_params)
    if 'sum_ee' in EI_params:
        W_ee = EI_params['sum_ee'] * W_ee
    print('sum_ee',np.sum(W_ee))

    # if 'sum_eeNorm' in EI_params:
    #     normal_amplitude = EI_params['sum_eeNorm']

    W_ei, inx_ei =  noisy_gaussian(N,M,EI_params,inh_factor, pbc=pbc,index=index,full_output=full_output,rotate_by=rotate_by,conv_params=conv_params)
    # if 'sum_ei' in EI_params:
    #     W_ei = EI_params['sum_ei'] * W_ei

    W_ei = - EI_params['sum_ei'] * W_ei
    print('sum_ei',np.sum(W_ei))

    W_ie, inx_ie = noisy_gaussian(N,M,EI_params,1.0,pbc=pbc,index=index,full_output=full_output,rotate_by=rotate_by,conv_params=conv_params)
    W_ie =  EI_params['sum_ie'] * W_ie
    #W_ii = np.zeros((N*M , N*M))

    W_ii,inx_ie = noisy_gaussian(N,M,EI_params, inh_factor,pbc=pbc,index=index,full_output=full_output,rotate_by=rotate_by,conv_params=conv_params)
    if 'self_inhibition' in EI_params:
        self_inhibition = np.eye(N*M).reshape(N,M,N,M)
        W_ii = alpha * self_inhibition + (1-alpha) * W_ii

    W_ii = - EI_params['sum_ii'] * W_ii
    W_full = np.hstack( (np.vstack((W_ee.reshape(N*M, N*M), W_ie.reshape(N*M,N*M))), np.vstack((W_ei.reshape(N*M,N*M), W_ii.reshape(N*M,N*M)))))
    print('sum_ii',np.sum(W_ii))
    return W_full, index #(2N, 2M)


def sampled_mh_wrap(N,M,mode,noise_type,sigmax,sigmax_sd,number_syn,ecc_sd,orientation,orientation_sd,a1,inh_factor,pbc=True,index=4876,full_output=True,version=0):
    rotate_by = None
    s1 = sigmax
    s2 = sigmax*inh_factor
    conv_params = {'do_convolution_ori' : False,'padd' : 0, 's1_ori' : s1, 's2_ori' : s2,
                       'do_convolution_x' : False, 's1_x' : s1, 's2_x' : s2,
                       'do_convolution_ecc' : False, 's1_ecc' : s1, 's2_ecc' : s2,
                       'do_lowpass_ori' : False,'do_lowpass_x' : False,'do_lowpass_ecc' : False,
                       'const_ori' : False,
                       'do_Binomial_sampling': True, 'K_I': number_syn, 'K_E': number_syn, 'w_I':-1.*inh_factor , 'w_E': 1., }

    return noisy_mh(N,M,mode,noise_type,sigmax,sigmax_sd,number_syn,ecc_sd,orientation,\
    orientation_sd,a1,inh_factor,pbc=pbc,index=index,full_output=full_output,\
        rotate_by=rotate_by,conv_params=conv_params)


def gaussian_wrap(N,M,connectivity_params,
                  pbc=True,index=4876,full_output=True,version=0):


    mode=connectivity_params['mode']
    noise_type=connectivity_params['noise_type']
    sigmax=connectivity_params['sigmax']
    sigmax_sd=connectivity_params['sigmax_sd']
    ecc=connectivity_params['ecc']
    ecc_sd=connectivity_params['ecc_sd']
    orientation=connectivity_params['orientation']
    orientation_sd=connectivity_params['orientation_sd']
    a1=connectivity_params['amplitude']
    inh_factor=connectivity_params['inh_factor']
    pbc=connectivity_params['pbc']

    rotate_by = None
    s1 = sigmax
    s2 = sigmax*inh_factor
    conv_params = {'do_convolution_ori' : False,'padd' : 0, 's1_ori' : s1, 's2_ori' : s2,
                       'do_convolution_x' : False, 's1_x' : s1, 's2_x' : s2,
                       'do_convolution_ecc' : False, 's1_ecc' : s1, 's2_ecc' : s2,
                       'do_lowpass_ori' : False,'do_lowpass_x' : False,'do_lowpass_ecc' : False,
                       'const_ori' : False,
                       'do_Binomial_sampling': False,
                       'represent_binary': connectivity_params['represent_binary'],
                       'radius_connectivity': connectivity_params['radius_connectivity']}

    return noisy_mh(N,M,mode,noise_type,sigmax,sigmax_sd,ecc,ecc_sd,orientation,\
    orientation_sd,a1,inh_factor,pbc=pbc,index=index,full_output=full_output,\
        rotate_by=rotate_by,conv_params=conv_params)

def noisy_mh_wrap(N,M,
                  connectivity_params,
                  full_output=True,version=0,
                  index=1804,
                  *args):

    mode=connectivity_params['mode']
    noise_type=connectivity_params['noise_type']
    sigmax=connectivity_params['sigmax']
    sigmax_sd=connectivity_params['sigmax_sd']
    ecc=connectivity_params['ecc']
    ecc_sd=connectivity_params['ecc_sd']
    orientation=connectivity_params['orientation']
    orientation_sd=connectivity_params['orientation_sd']
    a1=connectivity_params['amplitude']
    inh_factor=connectivity_params['inh_factor']
    pbc=connectivity_params['pbc']

    rotate_by = None
    s1 = sigmax
    s2 = sigmax*inh_factor
    conv_params = {'do_convolution_ori' : False,'padd' : 0, 's1_ori' : s1, 's2_ori' : s2,
                       'do_convolution_x' : False, 's1_x' : s1, 's2_x' : s2,
                       'do_convolution_ecc' : False, 's1_ecc' : s1, 's2_ecc' : s2,
                       'do_lowpass_ori' : False,'do_lowpass_x' : False,'do_lowpass_ecc' : False,
                       'const_ori' : False,'do_Binomial_sampling': False,
                       'represent_binary': False,}

    return noisy_mh(N,M,mode,noise_type,sigmax,sigmax_sd,ecc,ecc_sd,orientation,\
    orientation_sd,a1,inh_factor,pbc=pbc,index=index,full_output=full_output,\
        rotate_by=rotate_by,conv_params=conv_params)

def noisysmooth_mh_wrap(N,M,
                  connectivity_params,
                  full_output=True,version=0,
                  index=1804,
                  *args):

    mode=connectivity_params['mode']
    noise_type=connectivity_params['noise_type']
    sigmax=connectivity_params['sigmax']
    sigmax_sd=connectivity_params['sigmax_sd']
    ecc=connectivity_params['ecc']
    ecc_sd=connectivity_params['ecc_sd']
    orientation=connectivity_params['orientation']
    orientation_sd=connectivity_params['orientation_sd']
    a1=connectivity_params['amplitude']
    inh_factor=connectivity_params['inh_factor']
    pbc=connectivity_params['pbc']

    rotate_by = None
    s1 = sigmax
    s2 = sigmax*inh_factor
    conv_params = {'do_convolution_ori' : True,'padd' : 0, 's1_ori' : s1, 's2_ori' : s2,
                       'do_convolution_x' : True, 's1_x' : s1, 's2_x' : s2,
                       'do_convolution_ecc' : True, 's1_ecc' : s1, 's2_ecc' : s2,
                       'do_lowpass_ori' : False,'do_lowpass_x' : False,'do_lowpass_ecc' : False,
                       'const_ori' : False,'do_Binomial_sampling': False,
                       'represent_binary': False,}

    return noisy_mh(N,M,mode,noise_type,sigmax,sigmax_sd,ecc,ecc_sd,orientation,\
    orientation_sd,a1,inh_factor,pbc=pbc,index=index,full_output=full_output,\
        rotate_by=rotate_by,conv_params=conv_params)

def get_distances_L4_L23(N_L4, N_L23, pbc=True):

    coord_x_L4, coord_y_L4 = np.meshgrid(
        np.arange(N_L4)/N_L4, np.arange(N_L4)/N_L4) #N:x dimension; M:y dim.
    coord_x_L23,coord_y_L23= np.meshgrid(
        np.arange(N_L23)/N_L23,np.arange(N_L23)/N_L23)

    deltax = coord_x_L4[:,:,None,None] - coord_x_L23[None,None,:,:]
    deltay = coord_y_L4[:,:,None,None] - coord_y_L23[None,None,:,:]
    if pbc:

        absdeltax = np.abs(deltax)
        absdeltay = np.abs(deltay)
        idxx = np.argmin([absdeltax, 1 - absdeltax] , axis=0)
        idxy = np.argmin([absdeltay, 1 - absdeltay] , axis=0)

        deltax = deltax * (1-idxx) + np.sign(deltax) * (absdeltax-1) * idxx
        deltay = deltay * (1-idxy) + np.sign(deltay) * (absdeltay-1) * idxy

    #ignore eccentricity
    delta = (deltax)**2 + (deltay)**2

    return delta

def gauss(delta,inh_factor):
    return 1./inh_factor**2*np.exp(-delta/2./inh_factor**2)

def get_Gaussian_wrapper(gauss_params):

    distances = get_distances_L4_L23(gauss_params['N_L4'],
                                     gauss_params['N_L23'],
                                     gauss_params['pbc'])

    sigma = gauss_params['sigmax']
    w_ff = gauss(distances*gauss_params['N_L23']**2/(2*sigma**2),
                 1) / sigma / sigma /2./np.pi


    if gauss_params['represent_binary']:
        radius_true = gauss_params['radius_connectivity']
        delta_pure = np.sqrt(distances)
        w_ff = np.asarray(np.copy(delta_pure*gauss_params['N_L23']<radius_true), dtype='float')

    return w_ff, distances

from tools_general import filter_funcs
import scipy.stats as stats
def multi_dym_modular_wrap(N,M,
                  connectivity_params,
                 index=None,
                 **args):
    Dym = connectivity_params["dym"]
    eigenVals = np.exp(-np.linspace(0, (M * N) - 1, (M * N)) * (0.3/(2**Dym)))
    # for dymensionality = 8 based on (sum(Eig).^2)/(sum(Eig.^2))
    patNum = np.size(eigenVals, axis=0)
    rng = np.random.default_rng(index)
    mod_patterns = rng.normal(loc=0, scale=1, size=(patNum, N, M))
    mod_sigma_high = connectivity_params["Mod_High"]
    mod_sigma_low = connectivity_params["Mod_Low"]
    mod_patterns = filter_funcs.lowhigh_filter_stack(
        mod_patterns,
        sig_high=mod_sigma_high, sig_low=mod_sigma_low)
    covar_mat = np.transpose(np.reshape(mod_patterns,[patNum, N*M]))
    rec_longrange = np.matmul(np.matmul(covar_mat, np.diag(eigenVals)), np.transpose(covar_mat))

    Dist_Spec = connectivity_params['Dist_Spec']

    XPos = np.repeat([range(0,M)],N)
    XPos = np.reshape(XPos, [M*N, 1])
    YPos= np.reshape(np.transpose(np.reshape(XPos,[M, N])),[M*N, 1])
    XDist = np.abs(XPos - np.transpose(XPos))
    YDist = np.abs(YPos - np.transpose(YPos))
    XDist[XDist>M/2] = M - XDist[XDist>M/2]
    YDist[YDist>N/2] = N - YDist[YDist>N/2]
    Dist = np.sqrt(np.square(XDist) + np.square(YDist))
    #Dist[Dist == 0] = M*2
    Dist_w = stats.norm(0, Dist_Spec).pdf(Dist)
    w_rec = np.multiply(Dist_w, rec_longrange)
    w_rec = stats.zscore(w_rec, 1)

    return  w_rec, (covar_mat, mod_patterns)


def modular_comp_wrap_old(N,M,
                  connectivity_params,
                 index=None,
                 **args):
    beta_M = connectivity_params["beta_M"]
    eigenVals = np.exp(-2*np.arange(0, (M * N))/beta_M)

    # # for dymensionality = 8 based on (sum(Eig).^2)/(sum(Eig.^2))

    rng = np.random.default_rng(index)
    mod_patterns = rng.normal(loc=0, scale=1, size=(N*M, N, M))

    mod_sigma_high = connectivity_params["Mod_High"]
    mod_sigma_low = connectivity_params["Mod_Low"]
    mod_patterns = filter_funcs.lowhigh_filter_stack(
        mod_patterns,
        sig_high=mod_sigma_high, sig_low=mod_sigma_low)

    covar_mat = np.transpose(np.reshape(mod_patterns,[N*M, N*M]))
    rec_longrange = np.matmul(np.matmul(covar_mat, np.diag(eigenVals)), np.transpose(covar_mat))

    Dist_Spec = connectivity_params['Dist_Spec']

    XPos = np.repeat([range(0,M)],N)
    XPos = np.reshape(XPos, [M*N, 1])
    YPos= np.reshape(np.transpose(np.reshape(XPos,[M, N])),[M*N, 1])
    XDist = np.abs(XPos - np.transpose(XPos))
    YDist = np.abs(YPos - np.transpose(YPos))
    XDist[XDist>M/2] = M - XDist[XDist>M/2]
    YDist[YDist>N/2] = N - YDist[YDist>N/2]


    Dist = np.sqrt(np.square(XDist) + np.square(YDist))
    #Dist[Dist == 0] = M*2  #why prevent self-connection
    Dist_w = stats.norm(0, Dist_Spec).pdf(Dist)
    w_rec = np.multiply(Dist_w, rec_longrange)
    if connectivity_params["zscore"]:
        #w_rec = stats.zscore(w_rec, 1)


        #_rec -= np.mean(w_rec, axis=0)[None,:]
        #w_rec /= np.std(w_rec, axis=0)[None,:]

        w_rec -= np.mean(w_rec, axis=1)[:,None]
        w_rec /= np.std(w_rec, axis=1)[:,None]

    return  w_rec, (covar_mat, mod_patterns)


def modular_comp_wrap_vS(N,M,
                  connectivity_params,
                 index=None,
                 **args):
    beta_M = connectivity_params["beta_M"]
    eigenVals = np.exp(-2*np.arange(0, (M * N))/beta_M)

    # # for dymensionality = 8 based on (sum(Eig).^2)/(sum(Eig.^2))

    rng = np.random.default_rng(index)
    mod_patterns = rng.normal(loc=0, scale=1, size=(N*M, N, M))

    mod_sigma_high = connectivity_params["Mod_High"]
    mod_sigma_low = connectivity_params["Mod_Low"]
    mod_patterns = filter_funcs.lowhigh_filter_stack(
        mod_patterns,
        sig_high=mod_sigma_high, sig_low=mod_sigma_low)

    covar_mat = np.transpose(np.reshape(mod_patterns,[N*M, N*M]))
    rec_longrange = np.matmul(np.matmul(covar_mat, np.diag(eigenVals)), np.transpose(covar_mat))

    Dist_Spec = connectivity_params['Dist_Spec']

    XPos = np.repeat([range(0,M)],N)
    XPos = np.reshape(XPos, [M*N, 1])
    YPos= np.reshape(np.transpose(np.reshape(XPos,[M, N])),[M*N, 1])
    XDist = np.abs(XPos - np.transpose(XPos))
    YDist = np.abs(YPos - np.transpose(YPos))
    XDist[XDist>M/2] = M - XDist[XDist>M/2]
    YDist[YDist>N/2] = N - YDist[YDist>N/2]


    Dist = np.sqrt(np.square(XDist) + np.square(YDist))
    #Dist[Dist == 0] = M*2  #why prevent self-connection
    Dist_w = stats.norm(0, Dist_Spec).pdf(Dist)
    w_rec = np.multiply(Dist_w, rec_longrange)
    if connectivity_params["zscore"]:
        #w_rec = stats.zscore(w_rec, 1)

        W_rec_inhib = w_rec.copy()
        W_rec_inhib[W_rec_inhib>0]=np.nan

        W_rec_exc = w_rec.copy()
        W_rec_exc[W_rec_exc<0]=np.nan

        sum_exc =np.nansum(W_rec_exc, axis=1)
        sum_inhib =-1*np.nansum(W_rec_inhib, axis=1)

        w_rec = w_rec * (sum_inhib/sum_exc)[:,None]
        w_rec[W_rec_inhib<0] = W_rec_inhib[~np.isnan(W_rec_inhib)]

        #print("wihtin sum", np.nansum(w_rec,axis=1))

        #w_rec -= np.mean(w_rec, axis=1)[:,None]
        w_rec /= np.std(w_rec, axis=1)[:,None]

        #print(w_rec.mean(1))

        #Augusto
        # w_rec_pos = np.clip(w_rec,0, np.inf)
        # w_rec_neg = np.clip(w_rec,-np.inf,0)

        # # plt.imshow(w_rec_pos, cmap='RdBu_r', norm=mpl.colors.CenteredNorm(), )
        # # plt.show()

        # W_rec_exc = w_rec.copy()
        # W_rec_exc[W_rec_exc<0]=np.nan

        # # plt.imshow(w_rec_neg, cmap='RdBu_r', norm=mpl.colors.CenteredNorm(), )
        # # plt.show()


        # posNeg_mult = np.divide(-np.sum(w_rec_neg,1), np.sum(w_rec_pos,1))
        # w_rec = np.divide(w_rec_neg,posNeg_mult) + w_rec_pos
        # print(np.mean(w_rec,axis=0))
        # print(np.mean(w_rec,axis=1))



        # w_rec = np.divide(w_rec, np.std(w_rec,0))
        # w_rec = np.transpose(w_rec)

    return  w_rec, (covar_mat, mod_patterns)

def modular_comp_wrap(N,M,
                  connectivity_params,
                 index=None,
                 **args):
    beta_M = connectivity_params["beta_M"]
    eigenVals = np.exp(-2*np.arange(0, (M * N))/beta_M)
    patNum = np.size(eigenVals, axis=0)
    rng = np.random.default_rng(index)
    mod_patterns = rng.normal(loc=0, scale=1, size=(patNum, N, M))
    mod_sigma_high = connectivity_params["Mod_High"]
    mod_sigma_low = connectivity_params["Mod_Low"]
    mod_patterns = filter_funcs.lowhigh_filter_stack(
        mod_patterns,
        sig_high=mod_sigma_high, sig_low=mod_sigma_low)
    covar_mat = np.transpose(np.reshape(mod_patterns,[patNum, N*M]))
    rec_longrange = np.matmul(np.matmul(covar_mat, np.diag(eigenVals)), np.transpose(covar_mat))

    Dist_Spec = connectivity_params['Dist_Spec']

    XPos = np.repeat([range(0,M)],N)
    XPos = np.reshape(XPos, [M*N, 1])
    YPos= np.reshape(np.transpose(np.reshape(XPos,[M, N])),[M*N, 1])
    XDist = np.abs(XPos - np.transpose(XPos))
    YDist = np.abs(YPos - np.transpose(YPos))
    XDist[XDist>M/2] = M - XDist[XDist>M/2]
    YDist[YDist>N/2] = N - YDist[YDist>N/2]
    Dist = np.sqrt(np.square(XDist) + np.square(YDist))
    #Dist[Dist == 0] = M*2
    Dist_w = stats.norm(0, Dist_Spec).pdf(Dist)
    w_rec = np.multiply(Dist_w, rec_longrange)

    #Homogenization
    #w_rec += w_rec.T
    # pats = w_rec[0].reshape(N,M)
    # w_rec_copy = w_rec.copy()
    # for i in np.arange(N*M):
    #     ishift,jshift = np.unravel_index(i, (N,M))
    #     im_sx=np.roll(pats.copy(),
    #                           shift=(ishift, jshift),
    #                           axis=(0,1),
    #                           )
    #     w_rec_copy[i]=im_sx.reshape(N*M)
    # w_rec=w_rec.copy()



    w_rec_pos = np.clip(w_rec,0, np.inf)
    w_rec_neg = np.clip(w_rec,-np.inf,0)
    posNeg_mult = np.divide(-np.sum(w_rec_neg,1), np.sum(w_rec_pos,1))

    #print(posNeg_mult)
    w_rec = np.divide(w_rec_neg,posNeg_mult) + w_rec_pos
    #w_rec = np.divide(w_rec, np.max(w_rec,0))

    w_rec = np.divide(w_rec, np.std(w_rec,0))
    #standardabweichung --> peak höhe, std is more important than peak
    #oder profil kopieren


    w_rec = np.transpose(w_rec)
 #   w_rec = stats.zscore(w_rec, 1)

    #print('Augusto v')

    return w_rec, (rec_longrange, mod_patterns, covar_mat)



def homogenMH_plusRF_wrap(N,M,connectivity_params,
                          full_output=True,version=0,
                          index=1804,outputshape='2d',
                          *args):


    sigmax=connectivity_params['sigmax']
    a1=connectivity_params['amplitude']
    inh_factor=connectivity_params['inh_factor']
    pbc=connectivity_params['pbc']

    sigma1_bp=connectivity_params['sigma1_bp']
    sigma2_bp=connectivity_params['sigma2_bp']
    strength_RF=connectivity_params['strength_RF']

    MH_connectivity, new_index = noisy_mh(N,M,
                    mode='short_range',
                    noise_type="None",
                    sigmax=sigmax,
                    sigmax_sd=0,
                    ecc=0,
                    ecc_sd=0,orientation=0,
                    orientation_sd=0,
                    a1=a1,
                    inh_factor=inh_factor,
                    pbc=pbc,
                    index=index,
                    full_output=full_output,
                    rotate_by=None,
                    conv_params={'s1_x': sigmax,
                                'do_Binomial_sampling': False,
                                'represent_binary': False})

    randomGaussFields = get_random_field(N,M,N_fields=N*M,
                                         sig1=sigma1_bp, sig2=sigma2_bp,
                                         seed=index)
    randomGaussFields_shaped= randomGaussFields.reshape(N,M,N,M)
    pertubationField = (1 + strength_RF*randomGaussFields_shaped)
    #print(pertubationField.mean(), pertubationField.std())
    pertubationField[pertubationField<0]=0

    #pertubation with different patterns for each unit
    #connectivity = MH_connectivity*pertubationField

    #pertubation with the same pattern (as in NatComm paper)
    if connectivity_params['perturb_samepat']:
        randomGaussFields_single = randomGaussFields_shaped[0,0]
        pertubationField_single = (1 + strength_RF*randomGaussFields_single)
        pertubationField_single[pertubationField_single<0]=0
        connectivity = MH_connectivity*pertubationField_single[None,None,:,:]


    if outputshape=='2d':
        w_rec = connectivity.reshape(N*M, N*M)



        ##Normalization
        if connectivity_params['normalization']:
            # w_rec_pos = np.clip(w_rec,0, np.inf)
            # w_rec_neg = np.clip(w_rec,-np.inf,0)
            # posNeg_mult = np.divide(-np.sum(w_rec_neg,1), np.sum(w_rec_pos,1))

            # #print(posNeg_mult)
            # w_rec = np.divide(w_rec_neg,posNeg_mult) + w_rec_pos
            # #w_rec = np.divide(w_rec, np.max(w_rec,0))

            # w_rec = np.divide(w_rec, np.std(w_rec,0))


            #Normalize so that mean 0 and std 1

            W_rec_inhib = w_rec.copy()
            W_rec_inhib[W_rec_inhib>0]=np.nan
            W_rec_exc = w_rec.copy()
            W_rec_exc[W_rec_exc<0]=np.nan

            sum_exc =np.nansum(W_rec_exc, axis=1)
            #print(sum_exc)
            sum_inhib =-1*np.nansum(W_rec_inhib, axis=1)

            #multiply with exc/inh normalization factor for each unit
            w_rec = w_rec * (sum_inhib/sum_exc)[:,None]
            #replace with old inhibitory connections
            w_rec[~np.isnan(W_rec_inhib)] = W_rec_inhib[~np.isnan(W_rec_inhib)]

        #normalize by std necessary
        #TODO: implement new parameter separating the two normalizations
        if not connectivity_params['normalization']:
            if  connectivity_params['normalization_fallback']:
                w_rec /= np.std(w_rec, axis=1)[:,None]
                w_rec /= np.sum(w_rec, axis=1)[:,None]

    elif outputshape=='4d':
        w_rec = connectivity.copy()

    if connectivity_params['return_pertubfield']:
        return w_rec, randomGaussFields_single
    else:
        return w_rec, new_index


def noisy_mh_wrap_old(N,M,mode,noise_type,sigmax,sigmax_sd,ecc,ecc_sd,orientation,
                  orientation_sd,a1,inh_factor,pbc=True,index=4876,
                  full_output=True,version=0):
    rotate_by = None
    s1 = sigmax
    s2 = sigmax*inh_factor
    conv_params = {'do_convolution_ori' : False,'padd' : 0, 's1_ori' : s1, 's2_ori' : s2,
                       'do_convolution_x' : False, 's1_x' : s1, 's2_x' : s2,
                       'do_convolution_ecc' : False, 's1_ecc' : s1, 's2_ecc' : s2,
                       'do_lowpass_ori' : False,'do_lowpass_x' : False,'do_lowpass_ecc' : False,
                       'const_ori' : False,'do_Binomial_sampling': False,}

    return noisy_mh(N,M,mode,noise_type,sigmax,sigmax_sd,ecc,ecc_sd,orientation,\
    orientation_sd,a1,inh_factor,pbc=pbc,index=index,full_output=full_output,\
        rotate_by=rotate_by,conv_params=conv_params)



def recreate_connectivity_RF(N,M,connectivity_params,
                          full_output=True,version=0,
                          index=1804,outputshape='2d',
                          randomGaussFields_single=None,
                          *args):


    sigmax=connectivity_params['sigmax']
    a1=connectivity_params['amplitude']
    inh_factor=connectivity_params['inh_factor']
    pbc=connectivity_params['pbc']

    strength_RF=connectivity_params['strength_RF']

    MH_connectivity, new_index = noisy_mh(N,M,
                    mode='short_range',
                    noise_type="None",
                    sigmax=sigmax,
                    sigmax_sd=0,
                    ecc=0,
                    ecc_sd=0,orientation=0,
                    orientation_sd=0,
                    a1=a1,
                    inh_factor=inh_factor,
                    pbc=pbc,
                    index=index,
                    full_output=full_output,
                    rotate_by=None,
                    conv_params={'s1_x': sigmax,
                                'do_Binomial_sampling': False,
                                'represent_binary': False})



    #pertubation with the same pattern (as in NatComm paper)
    if connectivity_params['perturb_samepat']:

        pertubationField_single = (1 + strength_RF*randomGaussFields_single)
        pertubationField_single[pertubationField_single<0]=0
        connectivity = MH_connectivity*pertubationField_single[None,None,:,:]


    if outputshape=='2d':
        w_rec = connectivity.reshape(N*M, N*M)

        ##Normalization
        if connectivity_params['normalization']:
            # w_rec_pos = np.clip(w_rec,0, np.inf)
            # w_rec_neg = np.clip(w_rec,-np.inf,0)
            # posNeg_mult = np.divide(-np.sum(w_rec_neg,1), np.sum(w_rec_pos,1))

            # #print(posNeg_mult)
            # w_rec = np.divide(w_rec_neg,posNeg_mult) + w_rec_pos
            # #w_rec = np.divide(w_rec, np.max(w_rec,0))

            # w_rec = np.divide(w_rec, np.std(w_rec,0))


            #Normalize so that mean 0 and std 1

            W_rec_inhib = w_rec.copy()
            W_rec_inhib[W_rec_inhib>0]=np.nan
            W_rec_exc = w_rec.copy()
            W_rec_exc[W_rec_exc<0]=np.nan

            sum_exc =np.nansum(W_rec_exc, axis=1)
            #print(sum_exc)
            sum_inhib =-1*np.nansum(W_rec_inhib, axis=1)

            #multiply with exc/inh normalization factor for each unit
            w_rec = w_rec * (sum_inhib/sum_exc)[:,None]
            #replace with old inhibitory connections
            w_rec[~np.isnan(W_rec_inhib)] = W_rec_inhib[~np.isnan(W_rec_inhib)]

        #normalize by std necessary
        #TODO: implement new parameter separating the two normalizations
        if not connectivity_params['normalization']:
            if  connectivity_params['normalization_fallback']:
                w_rec /= np.std(w_rec, axis=1)[:,None]
                w_rec /= np.sum(w_rec, axis=1)[:,None]

    return w_rec

def add_component_wrec(old_conn, add_comp,
                      connectivity_params ):

    strength_add=connectivity_params['strength_addition']
    
    w_rec = old_conn + strength_add*add_comp
    ##Normalization
    if connectivity_params['normalization']:

        W_rec_inhib = w_rec.copy()
        W_rec_inhib[W_rec_inhib>0]=np.nan
        W_rec_exc = w_rec.copy()
        W_rec_exc[W_rec_exc<0]=np.nan

        sum_exc =np.nansum(W_rec_exc, axis=1)
        #print(sum_exc)
        sum_inhib =-1*np.nansum(W_rec_inhib, axis=1)

        #multiply with exc/inh normalization factor for each unit
        w_rec = w_rec * (sum_inhib/sum_exc)[:,None]
        #replace with old inhibitory connections
        w_rec[~np.isnan(W_rec_inhib)] = W_rec_inhib[~np.isnan(W_rec_inhib)]

    #normalize by std necessary
    #TODO: implement new parameter separating the two normalizations
    if not connectivity_params['normalization']:
        if  connectivity_params['normalization_fallback']:
            w_rec /= np.std(w_rec, axis=1)[:,None]
            w_rec /= np.sum(w_rec, axis=1)[:,None]

    return w_rec

import h5py
from .global_params import global_path
def get_new_conn_RF(connectivity_params,
                    folder_index_old='0',
                    filename_key='',
                    ):
    
    filename_NP='NetworkParams_v{}'.format(filename_key)
    filename_act='activity_v{}'.format(filename_key)

    #get old pertub pattern
    f = h5py.File(global_path+"{}.hdf5".format(filename_NP), 'r')
    #take last timeframe for all spontaneous patterns
    pertubation_field = f[str(folder_index_old)]['pertubation_field'][:]
    f.close()
    
    N,M=pertubation_field.shape
    
    #get old w_rec
    w_rec_old=recreate_connectivity_RF(N,M,connectivity_params,                              
                              randomGaussFields_single=pertubation_field,
                              )
    
    #get actvitiy during waves
    file_act = h5py.File(global_path+"{}.hdf5".format(filename_act), 'r')
    #take last timeframe for all spontaneous patterns
    activity=file_act[str(folder_index_old)]["activity"][:,:,:,0]
    file_act.close()
    
    #calc correlation
    activity_cutoff=0
    act_flat = np.reshape(activity[:,:,activity_cutoff:], (-1,N*M))
    corr = np.corrcoef(act_flat.T)
    
    
    #add
    w_rec=add_component_wrec(w_rec_old, corr,
                          connectivity_params )
    return w_rec
    
    
    




if __name__=='__main__':
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    ## Network size
    N=20
    M=20

    ##OLD MH wrapper
    # ## Connectivity settings
    # ## sigmax and inh_factor define spatial scale
    # sigmax = 1.8
    # inh_factor = 2.        ## kappa in manuscript

    # a1 = 1.                ## global strength of inoisehibition

    # ## heterogeneity settings
    # ecc = 0.8
    # sigmax_sd = ecc*0.15/0.8
    # ecc_sd = ecc*0.13
    # orientation = 0.
    # ori_sd = 1.0

    # ## I used VERSION number for different network settings
    # ## and index number for different heterogeneity settings
    # VERSION = 1
    # index = 1




    # ''' plot different single mhs '''
    # gauss_b,idx = noisy_mh_wrap_old(N,M,'short_range','postsyn',sigmax,sigmax_sd,ecc,\
    # ecc_sd,orientation,ori_sd,a1,inh_factor,pbc=True,index=index,full_output=True,version=VERSION)

    # ## show complete connectivity M
    # plt.figure()#(figsize=(30,30))
    # im=plt.imshow(gauss_b.reshape(N*M,N*M),interpolation='nearest',cmap='RdBu_r')#,vmax=0.05)
    # plt.colorbar(im)

    # ## show three examples of Mexican hats and one exemplary spectrum
    # fig=plt.figure()
    # ax=fig.add_subplot(141)
    # show1 = np.fft.fftshift(gauss_b[2,3,:,:])
    # im=ax.imshow(gauss_b[2,3,:,:],interpolation='nearest',cmap='RdBu_r')
    # #plt.colorbar(im)
    # ax=fig.add_subplot(142)
    # ax.imshow(gauss_b[3,3,:,:],interpolation='nearest',cmap='RdBu_r')
    # ax=fig.add_subplot(143)
    # ax.imshow(np.fft.fftshift(gauss_b[2,5,:,:]),interpolation='nearest',cmap='RdBu_r')
    # ax=fig.add_subplot(144)
    # show3 = np.fft.fftshift(np.abs(np.fft.fft2(gauss_b[10,10,:,:])))
    # ax.imshow(show3,interpolation='nearest',cmap='RdBu_r')
    # plt.show()

    ##clean gaussian wrapper
    eccentricity = 0.0
    network_params = {
            'mode'                    : 'short_range',
            'noise_type'              : 'None',
            'sigmax'                  : 5,
            'sigmax_sd'               : eccentricity*0.15/0.8,
            'ecc'                     : eccentricity,
            'ecc_sd'                  : 0.13*eccentricity,
            'orientation'             : 0,
            'orientation_sd'          : 1.0,
            'amplitude'               : 0.,
            'inh_factor'              : 1.5,
            'pbc'                     : True,
            'represent_binary'        : True,
            'radius_connectivity'     : 5,
            }

    gauss_b,idx = gaussian_wrap(N,M,network_params,
                      pbc=True,index=4876,full_output=True,version=0)

    ## show complete connectivity M
    plt.figure()#(figsize=(30,30))
    im=plt.imshow(gauss_b.reshape(N*M,N*M),interpolation='nearest',cmap='RdBu_r')#,vmax=0.05)
    plt.colorbar(im)

    ## show three examples of Mexican hats and one exemplary spectrum
    fig=plt.figure()
    ax=fig.add_subplot(141)
    show1 = np.fft.fftshift(gauss_b[0,0,:,:])
    im=ax.imshow(gauss_b[2,3,:,:],interpolation='nearest',cmap='RdBu_r',
                 norm=mpl.colors.CenteredNorm(),)
    #plt.colorbar(im)
    ax=fig.add_subplot(142)
    ax.imshow(gauss_b[3,3,:,:],interpolation='nearest',cmap='RdBu_r',
                 norm=mpl.colors.CenteredNorm(),)
    ax=fig.add_subplot(143)
    ax.imshow(np.fft.fftshift(gauss_b[2,5,:,:]),interpolation='nearest',
              cmap='RdBu_r',
              norm=mpl.colors.CenteredNorm(),)
    ax=fig.add_subplot(144)
    show3 = np.fft.fftshift(np.abs(np.fft.fft2(gauss_b[10,10,:,:])))
    ax.imshow(show3,interpolation='nearest',cmap='RdBu_r',
                 norm=mpl.colors.CenteredNorm(),)
    plt.show()


    fig=plt.figure()
    ax=fig.add_subplot(141)
    show1 = np.fft.fftshift(gauss_b[:,:,2,3])
    im=ax.imshow(gauss_b[2,3,:,:],interpolation='nearest',cmap='RdBu_r',
                 norm=mpl.colors.CenteredNorm(),)
    #plt.colorbar(im)
    ax=fig.add_subplot(142)
    ax.imshow(gauss_b[:,:,6,3],interpolation='nearest',cmap='RdBu_r',
                 norm=mpl.colors.CenteredNorm(),)
    ax=fig.add_subplot(143)
    ax.imshow(np.fft.fftshift(gauss_b[:,:,2,5]),interpolation='nearest',
              cmap='RdBu_r',
              norm=mpl.colors.CenteredNorm(),)
    ax=fig.add_subplot(144)
    show3 = np.fft.fftshift(np.abs(np.fft.fft2(gauss_b[10,10,:,:])))
    ax.imshow(show3,interpolation='nearest',cmap='RdBu_r',
                 norm=mpl.colors.CenteredNorm(),)
    plt.show()

    print(np.min(gauss_b))
