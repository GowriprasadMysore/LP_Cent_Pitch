#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 12:58:40 2023

@author: gowriprasadmysore
"""
import numpy as np
from scipy import signal
from scipy import interpolate
import scipy as sp
from scipy.signal import hilbert
##

#%%
def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.
    Source of this function: https://github.com/jmandel/fun-with-formants/blob/master/lpc.py
    Arguments
    ---------
        r : array-like
            input array to invert (since the matrix is symmetric Toeplitz, the
            corresponding pxp matrix is defined by p items only). Generally the
            autocorrelation of the signal for linear prediction coefficients
            estimation. The first item must be a non zero real.

    Note
    ----
    This implementation is in python, hence unsuitable for any serious
    computation. Use it as educational and reference purpose only.
    

    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:

                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).  
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if order > n - 1:
        raise ValueError("Order should be <= size-1")
    elif n < 1:
        raise ValueError("Cannot operate on empty array !")

    if not np.isreal(r[0]):
        raise ValueError("First item of input must be real.")
    elif not np.isfinite(1/r[0]):
        raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order+1, r.dtype)
    # temporary array
    t = np.empty(order+1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.
    e = r[0]

    for i in range(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]

        for j in range(order):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])

        e *= 1 - k[i-1] * np.conj(k[i-1])

    return a, e, k

#%%

def autocorr_from_spectrum(pwd,winsize_stft):
    """
    Parameters
    ----------
    pwd : float array (d)
        Power Spectrum P = S**2. d: Dimension[0,fs/2]
    winsize_stft : int
        Window size in No. of Samples.

    Returns
    -------
    acorr_fft : float array
        Autocorrelation values on the signal obtained from IFFT of power spectrum
    """
    # Calculate the autocorrelation from inverse FFT of the power spectrum
    acorr_fft = np.fft.ifft(pwd).real / winsize_stft
    # acorr_fft = acorr_fft / acorr_fft[0]
    
    return acorr_fft


#%%

def cent_linear_spectrum(f,pwd,f0):
    """
    Parameters
    ----------
    f : float array
        Linear Frequency Obtained from STFT. Ranges [0, fs/2].
    pwd : float array (d)
        Power Spectrum P = S**2. d: Dimension[0,fs/2]
    f0 : int or floar value
        Tonic frequency value.

    Returns
    -------
    pwd_cent_ds : float array
        Double sided Spectrum: After linear resampling in Cent scale. [-fs/2+df:fs/2]
    f_equi_Cent : float array
        Linearly sampled Cent scale frequency axis.
    f_Cent : float array
        Regular Cent scale w.r.t 'f'. Excluding 0th instance 

    """
    
    f_Cent = 1200*np.log2(f[1:]/f0) # Linear frequency to Cent scale (Excluding the 0th value as it will be -inf)
    f_equi_Cent = np.linspace(f_Cent[0], f_Cent[-1], len(f_Cent)) # Resampling the Cent scale uniformly
    
    fn = interpolate.interp1d(f_Cent, pwd[1:]) # Interpolate the Power Spectrum w.r.t uniform cent scale 
    pwd_cent_nw = fn(f_equi_Cent)   # use interpolation function returned by `interp1d`
    
    pwd_cent_new = np.insert(pwd_cent_nw, 0, pwd[0]) # Put back the 0th instance of Power Spectrum
    
    pwd_cent_flip = pwd_cent_nw[::-1][:-1] 
    
    pwd_cent_ds = np.concatenate( (pwd_cent_new, pwd_cent_flip), axis=0) # Flip and Concat to get double sided Power Spectrum
    
    return pwd_cent_ds, f_equi_Cent, f_Cent

#%%
def LP_Coeff_from_spectrum(pwd,winsize_stft,nfft,p=40):
    """
    Parameters
    ----------
    pwd : float array (d)
        Power Spectrum P = S**2. d: Dimension[0,fs/2]
    winsize_stft : int
        Window size in No. of Samples.
    nfft : int
        FFT length N.
    p : int, optional
        No. of LP Coefficients. The default is 40.

    Returns
    -------
    h : float array
        One sided LP Spectrum.
    w : float array
        frequency axis [0,pi].
    aks : float array
        LP Coefficients.
    acorr_fft : float array
        Autocorrelation values.

    """
    
    acorr_fft = autocorr_from_spectrum(pwd,winsize_stft)
    
    r = acorr_fft[0:p,];

    aks,e, k = levinson_1d(r, p-1)

    G = r[0] + np.sum(aks[1:]*r[1:])
    w, h = signal.freqz(b = np.sqrt(G), a=aks, worN=int(nfft/2))

    return h, w, aks, acorr_fft


def LP_Coeff_from_spectrum_fast(pwd,winsize_stft,nfft,p=40):
    """
    Parameters
    ----------
    pwd : float array (d)
        Power Spectrum P = S**2. d: Dimension[0,fs/2]
    winsize_stft : int
        Window size in No. of Samples.
    nfft : int
        FFT length N.
    p : int, optional
        No. of LP Coefficients. The default is 40.

    Returns
    -------
    h : complex float array
        One sided LP Spectrum.
    w : float array
        frequency axis [0,pi].
    aks : float array
        LP Coefficients.
    acorr_fft : float array
        Autocorrelation values.

    """
    acorr_fft = autocorr_from_spectrum(pwd,winsize_stft) # Auto correlation of Power spectrum using fft
    
    r = acorr_fft[0:p,]; # Get the first p+1 values

    phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:]) # Levinson-Durbin recursion to get LP Coefficients

    aks = np.insert(phi, 0, 1); # Appeding 1 to coefficients [1, -a1, -a2, .. ]
    
    G = r[0] + np.sum(phi*r[1:]) # Compute the Gain of the LP Spectrum

    w, h = signal.freqz(b = np.sqrt(G), a=aks, worN=int(nfft/2)) # Compute LP Spectrum
    
    return h, w, aks, acorr_fft


#%%

def LP_on_STFT(stfts,f,winsize_stft,nfft_size,p,f0,fs_audio):
    """
    Parameters
    ----------
    stfts : float array [d,t]
        Power Spectrogram P = S**2. d: Dimension[0,fs/2], t: No. of Frames.
    f : float array
        Linear Frequency Obtained from STFT. Ranges [0, fs/2].
    winsize_stft : int
        Window size in No. of Samples.
    nfft_size : int
        FFT size.
    p : int 
        No. of LP Coefficients.
    f0 : float
        Tomic Frequency.
    fs_audio : float
        Sampling rate.

    Returns
    -------
    aks : float array
        LP Coefficients.
    acorr_fft : float array
        Autocorrelation values.
    LP_Cent_Spect : complex float array
        One sided LP Spectrogram [df:fs/s, t]. t: No of Frames
    pwd_cent_ds : float array
        Double sided Spectrum: After linear resampling in Cent scale. [-fs/2+df:fs/2]
    f_equi_Cent : float array
        Linearly sampled Cent scale frequency axis.
    f_Cent : float array
        Regular Cent scale w.r.t 'f'. Excluding 0th instance 

    """
    
    LP_Cent_Spect,pwd_cent_ds,aks,acorr_fft = [],[],[],[]
    for ii in range(len(stfts.T)):
        print(ii)
        energ = sum(stfts[:,ii])
        pwd_cent_ds_, f_equi_Cent, f_Cent = cent_linear_spectrum(f,stfts[:,ii],f0)
        if energ > 0.000005:
            print('Non Silent frame')
            h_, w_, aks_, acorr_fft_ = LP_Coeff_from_spectrum(pwd_cent_ds_,winsize_stft,nfft_size,p)
        else:
            print('Silent frame')
            h_ = np.zeros(int(nfft_size/2), dtype=float)
            aks_ = np.zeros(p, dtype=float)
            acorr_fft_ = np.zeros(p, dtype=float)
        LP_Cent_Spect.append(h_)
        pwd_cent_ds.append(pwd_cent_ds_)
        aks.append(aks_)
        acorr_fft.append(acorr_fft_)
        
    LP_Cent_Spect = np.array(LP_Cent_Spect)
    pwd_cent_ds = np.array(pwd_cent_ds)
    acorr_fft = np.array(acorr_fft)
    aks = np.array(aks)
    
    return aks, acorr_fft, LP_Cent_Spect, pwd_cent_ds, f_equi_Cent, f_Cent


#%%
def LP_Spectrogram_from_time_data(x,fs,f0,p,win_size,hop_size):
    """
    Parameters
    ----------
    x : float array
        time domain audio data.
    fs : float
        Sampling Frequency in Hz.
    f0 : float
        Tonic frequency in Hz.
    p : int
        No. of LP Coefficients.
    win_size : float
        Window size in seconds 0.1 for 100ms.
    hop_size : float
        Hop size in seconds.

    Returns
    -------
    Zxx : f
        DESCRIPTION.
    t : float array
        time axis, w.r.t number frames.
    f : float array
        linear frequency axis in Hz [0,fs/2].
    f_Cent : float array
        Regular Cent scale w.r.t 'f'. Excluding 0th instance
    pwd_cent_ds : float array
        Double sided Spectrum: After linear resampling in Cent scale. [-fs/2+df:fs/2]
    f_equi_Cent : float array
        Linearly sampled Cent scale frequency axis.
    LP_Cent_Spect : complex float array
        One sided LP Spectrogram [df:fs/s, t]. t: No of Frames
    aks : float array
        LP Coefficients.
    acorr_fft : float array
        Autocorrelation values.
    nfft_size : int
        FFT size.
    """
    #%% Parameter initialization
    ###############################################################################
    winsize_stft = int(win_size * fs)  # for a frame width of 0.04s
    hopsize_stft = int(hop_size * fs)  # for a hop of 0.005 s
    overlap_stft = winsize_stft - hopsize_stft
    nfft_size = 2 ** np.ceil(np.log2(2*winsize_stft - 1)).astype('int')

    #%% Compute STFT Power spectrum
    f, t, Zxx = signal.stft(x=x, fs=fs, nperseg=winsize_stft,noverlap=overlap_stft,nfft=nfft_size)
    # Get the power spectrum
    Zxx = np.abs(Zxx) ** 2
    #%% Compute LP
    aks, acorr_fft, LP_Cent_Spect, pwd_cent_ds, f_equi_Cent, f_Cent = LP_on_STFT(Zxx,f,winsize_stft,nfft_size,p,f0,fs)
    
    return Zxx, t, f, f_Cent, pwd_cent_ds, f_equi_Cent, LP_Cent_Spect, aks, acorr_fft, nfft_size


def pitch_from_LP_Spectrogram(f_equi_Cent,LP_Cent_Spect,f0):
    """
    Parameters
    ----------
    f_equi_Cent : float array
        Linearly sampled Cent scale frequency axis.
    LP_Cent_Spect : complex float array matrix
        One sided LP Spectrogram [df:fs/s, t]. t: No of Frames
    f0 : float
        tonic frequency.

    Returns
    -------
    f_pitch_cent : float array
        Pitch values in Cents for every frame.
    pitch_freq : float array
        Pitch values in Hz for every frame..

    """
    idx = (f_equi_Cent < -1240)+(f_equi_Cent > 2400)
    # idx = (f_equi_Cent > -1240)*(f_equi_Cent < 2400)
    
    f_pitch_cent = []
    for jj in range(len(LP_Cent_Spect)):
        abs_spec_cent = np.abs(LP_Cent_Spect[jj,:]) 
        abs_spec_cent[np.where(idx)] = 0.0000001
        peak = np.argmax(abs_spec_cent)
        print(peak)
        f_pitch_cent_ = f_equi_Cent[peak]
        f_pitch_cent.append(f_pitch_cent_)
        
    f_pitch_cent = np.array(f_pitch_cent)
    pitch_freq = (2 ** (f_pitch_cent / 1200)) * f0
    print("Estimated fundamental frequency (pitch):", pitch_freq, "Hz")
    
    return f_pitch_cent, pitch_freq





def get_HEFD(signal):
    
    mean_ACF_diff=np.diff(np.array(signal))

    mean_ACF_diff = hilbert(mean_ACF_diff)
    mean_ACF_diff = np.abs(mean_ACF_diff)
    mean_ACF_diff=np.concatenate((mean_ACF_diff, np.zeros((1))), axis=0)
    mean_ACF_diff=mean_ACF_diff/np.max(mean_ACF_diff)

    return mean_ACF_diff


