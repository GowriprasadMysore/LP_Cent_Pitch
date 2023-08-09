#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:29:50 2023

@author: gowriprasad
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import utils_LP
import librosa
import pickle
from matplotlib import gridspec
import essentia.standard as es
import pandas as pd
import numpy.matlib
from scipy.io import savemat
import crepe
from tqdm import tqdm

#%%
file_name = "audio_stem_synth.txt"
file1 = open(file_name, 'r')
file_paths = file1.readlines()
file1.close()

file_path=[]
for files in open(file_name, "r"):
    file_path.append(files.rstrip('\n'))

no_files = len(file_path)

###############################################################################
'''Parameter Initialization'''
p = 18              # LP Order
win_size = 0.2      # Window Size
hop_size = 0.03     # Hop Size



#%%%%

base_dir = '/music/gprasad/LP_Analysis' #base path which contains audios in a separate folder called 'audios'
dump_dir = '/music/gprasad/LP_Analysis/Audio_Data/MDB-stem-synth/LP_Pitch_stems'
img_dir = '/music/gprasad/LP_Analysis/Audio_Data/MDB-stem-synth/plots'
GT_dir = '/music/gprasad/LP_Analysis/Audio_Data/MDB-stem-synth/annotation_stems'
if not os.path.exists(dump_dir): os.makedirs(dump_dir)
if not os.path.exists(img_dir): os.makedirs(img_dir)


#%%
raw_pitch_LP,raw_chroma_LP,overall_accuracy_LP = [],[],[]
raw_pitch_ES,raw_chroma_ES,overall_accuracy_ES = [],[],[]
raw_pitch_CP,raw_chroma_CP,overall_accuracy_CP = [],[],[]

#%%
#%%
for i in tqdm(range(len(file_path))):
    print(i)
    
# for file_path in open(file_name, "r"):
    file_pathh = file_path[i]
    # file_path = "Aparadhamula/Ashwath_Narayanan_-_Aparadhamula.ctonic.txt"
    fields = file_pathh.rstrip().split("/")
    songname=fields[-1]
    print('Audio: %s'%songname)
    
    # file_path_audio = file_path.replace('.ctonic.txt','.wav')
    # dump_file = os.path.join(data_dump_dir, fields[-3] + songname.replace('.ctonic.txt','.pickle'))
    # dump_mat_file = os.path.join(data_dump_dir, fields[-3] + songname.replace('.ctonic.txt','.mat'))
    # img_file = os.path.join(img_dir, fields[-3] + songname.replace('.ctonic.txt','.png'))

    file_path_audio = file_pathh
    img_file = os.path.join(img_dir, songname.replace('.wav','.png'))
    
    f0 = 136
    # with open(file_path,"r") as file:
    #     tonic = [float(line.strip()) for line in file]
    # f0 = tonic[0]
    #%% Load Audio 
    audio, fs_audio = librosa.load(file_path_audio, sr=16000)
    
    winsize_stft = int(win_size * fs_audio)  # for a frame width of 0.04s
    hopsize_stft = int(hop_size * fs_audio)  # for a hop of 0.005 s
    overlap_stft = winsize_stft - hopsize_stft
    #%% Load Grount Truth File  
    GT_file = os.path.join(GT_dir, songname.replace('.wav','.csv'))
    # Load data from the .csv file
    data_GT = np.loadtxt(GT_file, delimiter=',')
    time_GT = data_GT[:, 0]
    pitch_GT = data_GT[:, 1]
    
    print('Loaded Ground Truth')
    #%% compute LP Spectrogram
    # Zxx, t, f, f_Cent, pwd_cent_ds, f_equi_Cent, LP_Cent_Spect, aks, acorr_fft, nfft_size = utils_LP.LP_Spectrogram_from_time_data(audio,fs_audio,f0,p,win_size,hop_size)
    # #%% Compute Pitch using CENT LP
    # f_pitch_cent, pitch_freq = utils_LP.pitch_from_LP_Spectrogram(f_equi_Cent,LP_Cent_Spect,f0)
    
    dump_dir_txt_LP = os.path.join(dump_dir, songname.replace('.wav','_LP.txt'))
    data_LP = np.loadtxt(dump_dir_txt_LP)
    t = data_LP[:, 0]  # First column
    pitch_freq = data_LP[:, 1]
    
    raw_pitch_LP_,raw_chroma_LP_,overall_accuracy_LP_ = utils_LP.compute_pitch_accuracy(time_GT, pitch_GT, t, pitch_freq)
    
    raw_pitch_LP.append(raw_pitch_LP_)
    raw_chroma_LP.append(raw_chroma_LP_)
    overall_accuracy_LP.append(overall_accuracy_LP_)
    
    # time_pitch_vals_LP = np.column_stack((t, pitch_freq))
    # Step 2: Save the combined array to a .txt file with float format up to 5 decimal places
    # np.savetxt(dump_dir_txt_LP, time_pitch_vals_LP, fmt='%.5f', delimiter=' ')
    print('Computed LP Accuracy')
    #%% Compute Pitch using Essentia
    pitch_extractor = es.PredominantPitchMelodia(frameSize=winsize_stft, hopSize=hopsize_stft)
    pitch_freq_essentia, pitch_confidence = pitch_extractor(audio)
    time_pitch_vals_essentia = np.column_stack((t, pitch_freq_essentia))
    
    raw_pitch_ES_,raw_chroma_ES_,overall_accuracy_ES_ = utils_LP.compute_pitch_accuracy(time_GT, pitch_GT, t, pitch_freq_essentia)
    
    raw_pitch_ES.append(raw_pitch_ES_)
    raw_chroma_ES.append(raw_chroma_ES_)
    overall_accuracy_ES.append(overall_accuracy_ES_)
    print('Computed ES Accuracy')
    dump_dir_txt_Essent = os.path.join(dump_dir, songname.replace('.wav','_Essentia.txt'))
    # Step 2: Save the combined array to a .txt file with float format up to 5 decimal places
    np.savetxt(dump_dir_txt_Essent, time_pitch_vals_essentia, fmt='%.5f', delimiter=' ')
    
    #%% Compute Pitch using CRAPE
    time, frequency, confidence, activation = crepe.predict(audio, fs_audio, viterbi=False)
    time_pitch_vals_Crape = np.column_stack((time, frequency))
    
    raw_pitch_CP_,raw_chroma_CP_,overall_accuracy_CP_ = utils_LP.compute_pitch_accuracy(time_GT, pitch_GT, time, frequency)
    
    raw_pitch_CP.append(raw_pitch_CP_)
    raw_chroma_CP.append(raw_chroma_CP_)
    overall_accuracy_CP.append(overall_accuracy_CP_)
    print('Computed CP Accuracy')
    dump_dir_txt_Crape = os.path.join(dump_dir, songname.replace('.wav','_Crape.txt'))
    # Step 2: Save the combined array to a .txt file with float format up to 5 decimal places
    np.savetxt(dump_dir_txt_Crape, time_pitch_vals_Crape, fmt='%.5f', delimiter=' ')
    
    
    #%%  Saving all the data
    '''Save Spectrograms and pitch'''
print('Save Spectrograms and pitc')
    
    
    
raw_pitch_LP = np.array(raw_pitch_LP)
raw_chroma_LP = np.array(raw_chroma_LP)
overall_accuracy_LP  = np.array(overall_accuracy_LP)
accuracies_LP = np.column_stack((raw_pitch_LP, raw_chroma_LP, overall_accuracy_LP))

np.savetxt('accuracies_LP.txt', accuracies_LP, fmt='%.5f', delimiter=' ')


raw_pitch_ES = np.array(raw_pitch_ES)
raw_chroma_ES = np.array(raw_chroma_ES)
overall_accuracy_ES = np.array(overall_accuracy_ES)
accuracies_ES = np.column_stack((raw_pitch_ES, raw_chroma_ES, overall_accuracy_ES))
np.savetxt('accuracies_ES.txt', accuracies_ES, fmt='%.5f', delimiter=' ')


raw_pitch_CP = np.array(raw_pitch_CP)
raw_chroma_CP = np.array(raw_chroma_CP)
overall_accuracy_CP = np.array(overall_accuracy_CP)
accuracies_CP = np.column_stack((raw_pitch_CP, raw_chroma_CP, overall_accuracy_CP))
np.savetxt('accuracies_CP.txt', accuracies_CP, fmt='%.5f', delimiter=' ')


#%%

accuracies_CP = np.loadtxt('accuracies_CP.txt', delimiter=' ')
raw_pitch_CP = accuracies_CP[:, 0]
raw_chroma_CP = accuracies_CP[:, 1]

accuracies_ES = np.loadtxt('accuracies_ES.txt', delimiter=' ')
raw_pitch_ES = accuracies_ES[:, 0]
raw_chroma_ES = accuracies_ES[:, 1]

accuracies_LP = np.loadtxt('accuracies_LP.txt', delimiter=' ')
raw_pitch_LP = accuracies_LP[:, 0]
raw_chroma_LP = accuracies_LP[:, 1]

#%%
    # create a figure
    # print('create a figure')
    # fig = plt.figure(figsize=(28,8))
    # # fig = plt.figure()
    # spec = gridspec.GridSpec(ncols=1, nrows=2, hspace=0.5, height_ratios=[1, 1])

    # ax0 = fig.add_subplot(spec[0])
    # ax0.pcolormesh(t,f[1:],20*np.log10(Zxx[1:]))
    # ax0.plot(t,pitch_freq,'r')
    # ax0.set_ylim(0,700)
    # ax0.set_title('Regular STFT with Pitch track',fontsize= 20)
    # ax0.set_ylabel('Freq in Hz',fontsize= 16)

    # ax1 = fig.add_subplot(spec[1])
    # ax1.pcolormesh(t,f_equi_Cent,20*np.log10(np.abs(LP_Cent_Spect.T)))
    # ax1.plot(t,f_pitch_cent,'r')
    # ax1.set_ylim(-1500,2500)
    # ax1.set_title('Cent LP Spectrogram with Pitch track',fontsize= 20)
    # ax1.set_ylabel('Freq in Cents',fontsize= 16)
    # ax1.set_xlabel('Time [sec]',fontsize= 20)

    # plt.show()
    # #%
    # plt.savefig(img_file)
    # plt.clf()
    # plt.close('all')

#%%



# import numpy as np

# # Assuming you have two arrays of the same length
# array1 = np.array([1, 2, 3, 4, 5])
# array2 = np.array([10, 20, 30, 40, 50])

# # Step 1: Combine the arrays into a single 2D array using np.column_stack
# combined_array = np.column_stack((array1, array2))

# # Step 2: Save the combined array to a .txt file
# np.savetxt('output.txt', combined_array, fmt='%d', delimiter=' ')

