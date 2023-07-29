#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:53:56 2023

@author: gowriprasadmysore
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import utils_LP
import librosa
import pickle
from matplotlib import gridspec
import numpy.matlib
from scipy.io import savemat
#%%
file_name = "audio_melody.txt"
file1 = open(file_name, 'r')
file_paths = file1.readlines()
file1.close()

###############################################################################
'''Parameter Initialization'''
p = 18              # LP Order
win_size = 0.2      # Window Size
hop_size = 0.03     # Hop Size

#%%%%

base_dir = '/Users/gowriprasadmysore/Documents/Work/LP_Analysis' #base path which contains audios in a separate folder called 'audios'
dump_dir = '/Users/gowriprasadmysore/Documents/Work/LP_Analysis/Audio_Data/MDB-melody-synth/LP_melody'
img_dir = '/Users/gowriprasadmysore/Documents/Work/LP_Analysis/Audio_Data/MDB-melody-synth/plots'
if not os.path.exists(dump_dir): os.makedirs(dump_dir)
if not os.path.exists(img_dir): os.makedirs(img_dir)

#%%
for file_path in open(file_name, "r"):
    file_path = file_path.rstrip('\n')
    # file_path = "Aparadhamula/Ashwath_Narayanan_-_Aparadhamula.ctonic.txt"
    fields = file_path.rstrip().split("/")
    songname=fields[-1]
    print('Audio: %s'%songname)
    
    # file_path_audio = file_path.replace('.ctonic.txt','.wav')
    # dump_file = os.path.join(data_dump_dir, fields[-3] + songname.replace('.ctonic.txt','.pickle'))
    # dump_mat_file = os.path.join(data_dump_dir, fields[-3] + songname.replace('.ctonic.txt','.mat'))
    # img_file = os.path.join(img_dir, fields[-3] + songname.replace('.ctonic.txt','.png'))

    file_path_audio = file_path
    img_file = os.path.join(img_dir, songname.replace('.wav','.png'))
    dump_dir_txt = os.path.join(dump_dir, songname.replace('.wav','.txt'))
    f0 = 136
    # with open(file_path,"r") as file:
    #     tonic = [float(line.strip()) for line in file]
    # f0 = tonic[0]
    #%% Load Audio and compute LP Spectrogram
    audio, fs_audio = librosa.load(file_path_audio, sr=16000)
    Zxx, t, f, f_Cent, pwd_cent_ds, f_equi_Cent, LP_Cent_Spect, aks, acorr_fft, nfft_size = utils_LP.LP_Spectrogram_from_time_data(audio,fs_audio,f0,p,win_size,hop_size)
    
    #%% Compute Pitch
    
    f_pitch_cent, pitch_freq = utils_LP.pitch_from_LP_Spectrogram(f_equi_Cent,LP_Cent_Spect,f0)

#%%  Saving all the data
    '''Save Spectrograms and pitch'''
    print('Save Spectrograms and pitc')
    
    # with open(dump_file, 'wb') as f:
    #     pickle.dump([Zxx, t, f, f_Cent, pwd_cent_ds, f_equi_Cent, LP_Cent_Spect, aks, acorr_fft, nfft_size, f_pitch_cent, pitch_freq], f)
    # f.close()
    
    # mdic1 = {"Zxx": Zxx, "t": t, "f": f, "f_Cent": f_Cent, "pwd_cent_ds": pwd_cent_ds, "f_equi_Cent": f_equi_Cent, "LP_Cent_Spect": LP_Cent_Spect, "aks": aks, "acorr_fft": acorr_fft, "nfft_size": nfft_size, "f_pitch_cent": f_pitch_cent, "pitch_freq": pitch_freq }
    # savemat(dump_mat_file, mdic1)

#%%

    time_pitch_vals = np.column_stack((t, pitch_freq))

    # Step 2: Save the combined array to a .txt file with float format up to 5 decimal places
    np.savetxt(dump_dir_txt, time_pitch_vals, fmt='%.5f', delimiter=' ')

# #%%
#     # create a figure
#     print('create a figure')
#     fig = plt.figure(figsize=(24,8))
#     # fig = plt.figure()
#     spec = gridspec.GridSpec(ncols=1, nrows=2, hspace=0.5, height_ratios=[1, 1])

#     ax0 = fig.add_subplot(spec[0])
#     ax0.pcolormesh(t,f[1:],20*np.log10(Zxx[1:]))
#     ax0.plot(t,pitch_freq,'r')
#     ax0.set_title('Regular STFT with Pitch track',fontsize= 20)
#     ax0.set_ylabel('Freq in Hz',fontsize= 16)

#     ax1 = fig.add_subplot(spec[1])
#     ax1.pcolormesh(t,f_equi_Cent,20*np.log10(np.abs(LP_Cent_Spect.T)))
#     ax1.plot(t,f_pitch_cent,'r')
#     ax1.set_title('Cent LP Spectrogram with Pitch track',fontsize= 20)
#     ax1.set_ylabel('Freq in Cents',fontsize= 16)
#     ax1.set_xlabel('Time [sec]',fontsize= 20)

#     plt.show()
#     #%
#     plt.savefig(img_file)
#     plt.clf()
#     plt.close('all')

#%%



# import numpy as np

# # Assuming you have two arrays of the same length
# array1 = np.array([1, 2, 3, 4, 5])
# array2 = np.array([10, 20, 30, 40, 50])

# # Step 1: Combine the arrays into a single 2D array using np.column_stack
# combined_array = np.column_stack((array1, array2))

# # Step 2: Save the combined array to a .txt file
# np.savetxt('output.txt', combined_array, fmt='%d', delimiter=' ')

