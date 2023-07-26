#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:20:44 2023

@author: gowriprasadmysore
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 12:58:40 2023

@author: gowriprasadmysore
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
from scipy import interpolate

#%%
y, sr = librosa.load(librosa.ex('trumpet'))
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

#%%
S = np.abs(librosa.stft(y))
pitches, magnitudes = librosa.piptrack(S=S, sr=sr)

pitches, magnitudes = librosa.piptrack(S=S, sr=sr, threshold=1,
                                       ref=np.mean)
