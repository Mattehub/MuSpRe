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

total_channels_set=set()

for subject in subject_list:
    with h5py.File(pjoin('seeg_data_h5py/h5_electrodes/', subject + '_electrodes.hdf5'), 'r') as f:
        print(f.keys())
        print('chnames', f['chnames'].shape)
        
        chnames = f['chnames'][...].astype('U')
        total_channels_set.update(chnames)
        
print(total_channels_set)

ch_H=set()
for ch in total_channels_set:
    
    if "H" in ch:
        ch_H.add(ch)
print(ch_H)
