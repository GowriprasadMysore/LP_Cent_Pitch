#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:12:00 2023

@author: gowriprasadmysore

LP From Spectrum
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
###############################################################################
winsize_stft = int(0.2 * fs_audio)  # for a frame width of 0.04s
hopsize_stft = int(0.03 * fs_audio)  # for a hop of 0.005 s
overlap_stft = winsize_stft - hopsize_stft
nfft_size = 2 ** np.ceil(np.log2(2*winsize_stft - 1)).astype('int')
p = 30
frames = librosa.util.frame(audio, frame_length=int(winsize_stft), hop_length=int(hopsize_stft), axis=0)

#%%
f, t, Zxx = signal.stft(x=audio, fs=fs_audio, nperseg=winsize_stft,noverlap=overlap_stft,nfft=nfft_size)
f0 = 220

f_Cent = 1200*np.log2(f[1:]/f0)

freq_bins = (2 ** (f_Cent / 1200)) * f0


# Get the power spectrum
Zxx = np.abs(Zxx) ** 2

#%%
h,w,aks,acorr_fft = [],[],[],[]
for ii in range(len(Zxx.T)):
    print(ii)
    pwd_cent_ds, f_equi_Cent, f_Cent = utils_LP.cent_linear_spectrum(f,Zxx[:,ii],f0)
    h_, w_, aks_, acorr_fft_ = utils_LP.LP_Coeff_from_spectrum_fast(pwd_cent_ds,winsize_stft,nfft_size,p)
    h.append(h_)
    w.append(w_)
    aks.append(aks_)
    acorr_fft.append(acorr_fft_)
    
h = np.array(h)
w = np.array(w)
tx = np.arange(len(w))*hopsize_stft/fs_audio
#%%
p = 18
f0 = 220
win_size = 0.2
hop_size = 0.03
Zxx, t, f, f_Cent, pwd_cent_ds, f_equi_Cent, LP_Cent_Spect, aks, acorr_fft, nfft_size = utils_LP.LP_Spectrogram_from_time_data(audio,fs_audio,f0,p,win_size,hop_size)

#%%
for jj in range(len(Zxx.T)):
    plt.figure()
    plt.plot(f_Cent,20*np.log10(np.abs(Zxx[1:,jj])))
    plt.plot(f_equi_Cent,20*np.log10(np.abs(LP_Cent_Spect[jj,:])))

#%%  Saving all novelty functions ASD, STE
'''Saving all novelty functions ASD, STE'''
print('Saving all novelty functions ASD, STE')

img_name = file_path.replace('.wav','_cmpr.png')

Ann_file = os.path.join(base_dir,'Annotations', file_path[i].replace('.wav','.csv'))

with open(nov_file_name, 'wb') as f:
    pickle.dump([ASD_avg,STE,NF_M,STE_D,ASD_D,NF_RF,NF_R,NF_P,NF_All,MFCC,Post_GMM,log_ACF], f)
f.close()



# mdic1 = {"ASD_avg": ASD_avg, "STE": STE, "NF_M": NF_M, "STE_D": STE_D, "ASD_D": ASD_D, "NF_RF": NF_RF, "NF_R": NF_R, "NF_P": NF_P, "MFCC": MFCC, "Post_GMM": Post_GMM, "log_ACF": log_ACF }

# savemat(os.path.join(data_mat_dir, file_path[i].replace('.wav','_nov_all.mat')), mdic1)


#%%

plt.figure()
plt.subplot(2,1,1)
plt.pcolormesh(t,f_equi_Cent,20*np.log10(np.abs(LP_Cent_Spect.T)))

plt.subplot(2,1,2)
# plt.pcolormesh(t, f[0:400],20*np.log10(Zxx[0:400,:]))

plt.pcolormesh(t,f_Cent,20*np.log10(Zxx[1:]))
    
#%%

# f0 = 440
# f_Cent = 1200*np.log2(f[1:]/f0)
# f_equi_Cent = np.linspace(f_Cent[0], f_Cent[-1], len(f_Cent))
# #%%
# fn = interpolate.interp1d(f_Cent, Zxx[:,0][1:])
# Zxx_new = fn(f_equi_Cent)   # use interpolation function returned by `interp1d`

# Zxx_neww = np.insert(Zxx_new, 0, Zxx[0,0])

# zxx_flip = Zxx_new[::-1][:-1]

# pwd = np.concatenate( (Zxx_neww, zxx_flip), axis=0)
# #%%
# # Calculate the autocorrelation from inverse FFT of the power spectrum
# acorr_fft = np.fft.ifft(pwd).real / winsize_stft

# #%%
# M = 30

# r = acorr_fft[0:M,];

# phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])

# phi_ = np.insert(phi, 0, 1);

# w_, h_ = signal.freqz(b = 1, a=phi_, worN=int(nfft_size/2))


# plt.figure(1)
# plt.subplot(3,1,1)
# plt.plot(np.log(np.abs(h_)))
# # plt.subplot(3,1,2)
# # plt.plot(np.abs(h_akss))
# plt.subplot(3,1,2)
# plt.plot(20*np.log10(np.abs(Zxx_neww)))

# #%%
# M=40
# r = acorr_fft[0:M,];

# phi,e, k = levinson_1d(r, M-1)

# # phi_ = np.insert(phi, 0, 1);

# w, h = signal.freqz(b = 1, a=phi, worN=nfft_size/2)


# #%%
# plt.figure(1)
# plt.subplot(3,1,1)
# plt.plot(np.log(np.abs(h)))
# # plt.subplot(3,1,2)
# # plt.plot(np.abs(h_akss))
# plt.subplot(3,1,2)
# plt.plot(20*np.log10(np.abs(Zxx_neww)))

# #%%
# # f, t, Zxx = signal.stft(x, fs, nperseg=1000, return_onesided=False,scaling='psd')

# plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx)))
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

























