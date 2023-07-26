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
import librosa
import pickle
import csv
from matplotlib import gridspec
from scipy import linalg, signal

import scipy as sp


#%%
file_path = "test_1.wav"
audio, fs_audio = librosa.load(file_path, sr=16000)

#%%
###############################################################################
winsize_stft = 0.04  # for a frame width of 0.04s
hopsize_stft = 0.005  # for a hop of 0.005 s
lagsize_acf = 2     # ACF calculated upto 3 seconds for Vil & 1.5s for Madhya & Drut
winsize_acf = 4  # 5 seconds window for Vilambit & 3s for Madhya & Drut
hopsize_acf = 0.5  # successive ACF frames are shifted by 0.5 s

#%%

datax = audio[0:2048,]

# import numpy as np
# from scipy import signal
# import matplotlib.pyplot as plt

# # Generate a sample spectrum (power spectrum)
# spectrum = np.random.rand(512)

# # Compute autocorrelation from spectrum
# autocorr = np.fft.irfft(np.abs(spectrum)**2)

# # Estimate LP coefficients using Levinson-Durbin recursion
# lpc_order = 10
# lpc_coeffs = np.zeros(lpc_order+1)
# r = autocorr[:lpc_order+1]

# for i in range(1, lpc_order+1):
#     alpha = -np.dot(lpc_coeffs[1:i][::-1], r[1:i]) / r[0]
#     lpc_coeffs[i] = alpha
#     lpc_coeffs[1:i] += alpha * lpc_coeffs[1:i][::-1]

# # Compute LP spectrum using inverse filter
# lp_spectrum = np.abs(np.fft.fft(lpc_coeffs, len(spectrum)))

# # Plot the LP spectral envelope
# plt.plot(lp_spectrum)
# plt.xlabel('Frequency')
# plt.ylabel('Magnitude')
# plt.title('LP Spectral Envelope')
# plt.show()


#%%

''' np.correlate '''
# Mean
mean = np.mean(datax)

# Variance
var = np.var(datax)

# Normalized data
ndata = datax - mean

acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
acorr = acorr / var / len(ndata)

#%%

''' Fourier transform implementation '''


# Nearest size with power of 2
size = 2 ** np.ceil(np.log2(2*len(datax) - 1)).astype('int')

# Variance
var = np.var(datax)

# Normalized data
ndata = datax - np.mean(datax)

# Compute the FFT
fft_ = np.fft.fft(ndata, size)

# Get the power spectrum
pwr = np.abs(fft_) ** 2



# Calculate the autocorrelation from inverse FFT of the power spectrum
acorr_fft = np.fft.ifft(pwr).real / var / len(datax)

#%%
M = 13

r = acorr_fft[0:M,];

phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])

phi_ = np.insert(phi, 0, 1);
#%%

def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.

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

#%%
M = 40

r = acorr_fft[0:M,];

phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])

phi_ = np.insert(phi, 0, 1);

G = r[0] + np.sum(phi*r[1:])

w, h = signal.freqz(b = np.sqrt(G), a=phi_, worN=2048)
# w, h = signal.freqz(b = G, a=phi_, worN=2048)


plt.plot(20*np.log10(np.abs(fft_[0:2048])))
plt.plot(20*np.log10(np.abs(h)))

#%%
M=30
r = acorr_fft[0:M,];

phi,e, k = levinson_1d(r, M-1)

G = r[0] + np.sum(phi[1:]*r[1:])

# phi_ = np.insert(phi, 0, 1);

w, h = signal.freqz(b = np.sqrt(G), a=phi, worN=2048)
# w, h = signal.freqz(b = 1, a=phi, worN=2048)


plt.plot(20*np.log10(np.abs(fft_[0:2048])))
plt.plot(20*np.log10(np.abs(h)))
#%%
akss = librosa.lpc(ndata,M)

w, h_akss = signal.freqz(b = 1, a=akss, worN=2048)
#%%

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(20*np.log10(np.abs(h)))
plt.subplot(3,1,2)
plt.plot(20*np.log10(np.abs(h_akss)))
plt.subplot(3,1,3)
plt.plot(20*np.log10(np.abs(fft_[0:2048])))
plt.hold()
plt.plot(20*np.log10(np.abs(h)))
#%%
tt = np.arange(2048)
M = 30

aks = acorr[0:M,];
tplz = linalg.toeplitz(aks);

Acoeffs = linalg.solve(tplz, acorr[1:M+1,])

# Alp = np.insert(Acoeffs, 0, 1);

Alp = Acoeffs

# w, h = signal.freqz(b = Alp, a=1, worN=2048)

w, h = signal.freqz(b = 1, a=phi, worN=2048)

akss = librosa.lpc(ndata,13)

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(np.abs(h))
plt.subplot(2,1,2)
plt.plot(np.abs(fft_[0:2048]))
#%%

from pylab import log10, linspace, plot, xlabel, ylabel, legend, randn, pi
import scipy.signal
from spectrum import aryule, Periodogram, arma2psd
# Create a AR model
a = [1, -2.2137, 2.9403, -2.1697, 0.9606]
# create some data based on these AR parameters
y = scipy.signal.lfilter([1], a, randn(1, 1024))
# if we know only the data, we estimate the PSD using Periodogram
p = Periodogram(y[0], sampling=2)  # y is a list of list hence the y[0]
p.plot(label='Model ouput')

# now, let us try to estimate the original AR parameters
AR, P, k = aryule(y[0], 4)
PSD = arma2psd(AR, NFFT=512)
PSD = PSD[len(PSD):len(PSD)//2:-1]
plot(linspace(0, 1, len(PSD)), 10*log10(abs(PSD)*2./(2.*pi)),
    label='Estimate of y using Yule-Walker AR(4)')
xlabel(r'Normalized frequency (\times \pi rad/sample)')
ylabel('One-sided PSD (dB/rad/sample)')
legend()

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
x = np.arange(0, 10)
y = np.exp(-x/3.0)
f = interpolate.interp1d(x, y)

xnew = np.arange(0, 9, 0.1)
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()