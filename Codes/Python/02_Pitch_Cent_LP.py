#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 00:45:25 2023

@author: gowriprasadmysore
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import utils_LP
import librosa
import pickle
import csv
from matplotlib import gridspec
from scipy import linalg, signal
from scipy import interpolate
import scipy as sp
#%%
file_path = "test_4_vocal.wav"
audio, fs_audio = librosa.load(file_path, sr=16000)
#%%
p = 18
f0 = 220
win_size = 0.2
hop_size = 0.03
Zxx, t, f, f_Cent, pwd_cent_ds, f_equi_Cent, LP_Cent_Spect, aks, acorr_fft, nfft_size = utils_LP.LP_Spectrogram_from_time_data(audio,fs_audio,f0,p,win_size,hop_size)

#%%


def hps_pitch_detection(X, f_equi_Cent, f0):
    # Compute the magnitude spectrum using FFT
    spectrum = X

    # Harmonic Product Spectrum (HPS)
    spectra = []
    for n in range(1, 4):
        s = sp.signal.resample(spectrum, len(spectrum) // n)
        spectra.append(s)

    # Truncate to most downsampled spectrum.
    l = min(len(s) for s in spectra)
    a = np.zeros((len(spectra), l), dtype=spectrum.dtype)
    for i, s in enumerate(spectra):
        a[i] += s[:l]

    # Multiply spectra per frequency bin.
    hps = np.product(np.abs(a), axis=0)

    # Find the peak in the HPS spectrum
    peak_idx = np.argmax(hps)
    fundamental_freq = f_equi_Cent[peak_idx] * 100
    pitch_freq = (2 ** (fundamental_freq / 1200)) * f0
    return pitch_freq, fundamental_freq

# Example usage
# Replace 'your_audio_signal' and 'your_sample_rate' with your audio data and sample rate

#%%

def pitch_hps(X, f_equi_Cent, f0):
    """Estimate the pitch contour in a monophonic audio signal."""

    f0s = []
    # Downsample spectrum.
    spectra = []
    for n in range(1, 4):
        s = sp.signal.resample(X, len(X) // n)
        spectra.append(s)

    # Truncate to most downsampled spectrum.
    l = min(len(s) for s in spectra)
    a = np.zeros((len(spectra), l), dtype=X.dtype)
    for i, s in enumerate(spectra):
        a[i] += s[:l]

    # Multiply spectra per frequency bin.
    hps = np.product(np.abs(a), axis=0)

    # TODO Blur spectrum to remove noise and high-frequency content.
    #kernel = sp.signal.gaussian(9, 1)
    #hps = sp.signal.fftconvolve(hps, kernel, mode='same')

    # TODO Detect peaks with a continuous wavelet transform for polyphonic signals.
    #peaks = sp.signal.find_peaks_cwt(np.abs(hps), np.arange(1, 3))

    # Pick largest peak, it's likely f0.
    peak = np.argmax(hps)
    f0 = f_equi_Cent[peak* 100]
    f0s.append(f0)


    f0s = np.array(f0s)

    # Median filter out noise.
    f0s = sp.signal.medfilt(f0s, [21])

    return f0s
#%%
# idx = (f_equi_Cent < -1240)+(f_equi_Cent > 2400)
# idx = (f_equi_Cent > -1240)*(f_equi_Cent < 2400)

pitch_freq, fundamental_freq = [],[]
for jj in range(len(LP_Cent_Spect)):
    abs_spec_cent = np.abs(LP_Cent_Spect[jj,:]) 
    # abs_spec_cent[np.where(idx)] = 0.0000001
    pitch_freq_, fundamental_freq_ = hps_pitch_detection(abs_spec_cent, f_equi_Cent, f0)
    pitch_freq.append(pitch_freq_)
    fundamental_freq.append(fundamental_freq_)
print("Estimated fundamental frequency (pitch):", pitch_freq, "Hz")


#%%
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


#%%
for jj in range(len(Zxx.T)):
    plt.figure()
    plt.plot(f_Cent,20*np.log10(np.abs(Zxx[1:,jj])))
    plt.plot(f_equi_Cent,20*np.log10(np.abs(LP_Cent_Spect[jj,:])))


#%%
plt.figure()
plt.subplot(2,1,1)
plt.pcolormesh(t,f_equi_Cent,20*np.log10(np.abs(LP_Cent_Spect.T)))
plt.plot(t,f_pitch_cent,'r')
plt.subplot(2,1,2)
# plt.pcolormesh(t, f[0:400],20*np.log10(Zxx[0:400,:]))

plt.pcolormesh(t,f_Cent,20*np.log10(Zxx[1:]))

plt.figure()
plt.pcolormesh(t,f[1:],20*np.log10(Zxx[1:]))
plt.plot(t,pitch_freq,'r')











