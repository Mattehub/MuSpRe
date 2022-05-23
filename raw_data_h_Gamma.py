#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:23:07 2022

@author: jeremy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:48:44 2022

@author: jeremy
"""

import h5py
import mne
import numpy as np
import pandas as pd
from os.path import join as pjoin
from itertools import product

import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import colors
#import Utils_FC.py as ufc
import os
import pickle

import warnings 
warnings.simplefilter('ignore')



arr_mu = os.listdir('seeg_fif_data/music')
arr_rest = os.listdir('seeg_fif_data/speech')
arr_speech = os.listdir('seeg_fif_data/rest')

subject_set_mu=set()
subject_set_speech=set()
subject_set_rest=set()

for st in arr_mu: 
    print(st)
    subject_set_mu.add(st.partition('_')[0])
    print(st.partition('_')[0])
    
for st in arr_speech:
    subject_set_speech.add(st.partition('_')[0])
    
for st in arr_rest:
    subject_set_rest.add(st.partition('_')[0])

subject_list=list(subject_set_mu.intersection(subject_set_speech,subject_set_rest))

fmin, fmax =1, 40
for isub, subject in enumerate(subject_list):
    print('We are elaborating the data concerning subject', subject,'step', isub )
    for i, sound in enumerate(['speech', 'music', 'rest']):
        print(sound)
        path = pjoin('seeg_fif_data/', sound, )
        raw = mne.io.Raw(pjoin(path, subject +  '_' + sound + '_split_up_raw.fif'), preload=True)
        raw_filt = raw.filter(fmin, fmax, n_jobs=-1)
        raw_down= raw_filt.resample(40, npad='auto')
        data = raw_down.get_data()
        keys = (sound, )
        with h5py.File(pjoin('seeg_1_40hz_h5py/', subject+"1_40hz_seeg_preproc" + '.hdf5'), 'a') as hf:
            for k in keys[:-1]:
                hf = hf[k] if k in hf else hf.create_group(k)
            hf.create_dataset(keys[-1], data=data)
        hf.close()