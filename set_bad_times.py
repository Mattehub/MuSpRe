#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:49:55 2022

@author: jeremy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 19:26:54 2022

@author: jeremy
"""

import h5py
import mne
import numpy as np
import pandas as pd
from os.path import join as pjoin
from itertools import product

import saving as sav
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import colors
#import Utils_FC.py as ufc
import os
import pickle
import Utils_FC as fc

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

#Here I create a set of the H channels
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


#length of the interval to analyse in one step 

rss_music={}
rss_speech={}
rss_rest={}

data_music={}
data_speech={}
data_rest={}

art_speech_times={}
art_music_times={}
art_rest_times={}

N=7
t=30000
for isub, subject in enumerate(subject_list):
## Load the data from the HDF fil
    print(subject, isub)
    
    #MUSIC
    with h5py.File(pjoin('seeg_hgenv_down_down_h5py/', subject + '_hgenv_down_down_seeg_preproc.hdf5'), 'r') as f:
        print(f.keys())
        print('music', f['music'].shape)

        #data_music[subject]=f['music'][...]
        data_m=f['music'][...]
    
    #SPEECH
    with h5py.File(pjoin('seeg_hgenv_down_down_h5py/', subject + '_hgenv_down_down_seeg_preproc.hdf5'), 'r') as f:
        print(f.keys())
        print('speech', f['speech'].shape)
        print('speech', f['speech'].shape)
        #data_speech[subject]= f['speech'][...]
        data_s=f['speech'][...]

    #REST
    with h5py.File(pjoin('seeg_hgenv_down_down_h5py/', subject + '_hgenv_down_down_seeg_preproc.hdf5'), 'r') as f:
        print(f.keys())
        print('rest', f['rest'].shape)
        print('rest', f['rest'].shape)
        #data_rest[subject]=f['rest'][...]
        data_r=f['rest'][...]


# redefine path
# below example of loading of music data.

    with h5py.File(pjoin('seeg_data_h5py/h5_electrodes/', subject + '_electrodes.hdf5'), 'r') as f:
        print(f.keys())
        print('chnames', f['chnames'].shape)
    
        chnames = f['chnames'][...].astype('U')

    with h5py.File(pjoin('seeg_data_h5py/h5_misc/', subject + '_misc.hdf5'), 'r') as f:
        print(f.keys())
        print('outlier_chans', f['outlier_chans']['strict_bads_names'])

        bad_chans = f['outlier_chans']['strict_bads_names'][...].astype('U')
        mu_bad_epo = f['outlier_epochs']['music']['strict_bads_epochs'][...]
        sp_bad_epo = f['outlier_epochs']['speech']['strict_bads_epochs'][...]
    
    """with h5py.File(pjoin('speech_stimulus.hdf5'), 'r') as f:
        print(f.keys())
        speech_stimulus=f['speech']['matlab']['speech_matlab_env'][...]
        plt.plot(speech_stimulus)
        print('the length of the speech stimulus is', len(speech_stimulus))
    
    with h5py.File(pjoin('music_stimulus.hdf5'), 'r') as f:
        print(f.keys())
        music_stimulus=f['music']['matlab']['music_matlab_env'][...]
        plt.plot(music_stimulus)
        print('the length of the music stimulus is', len(speech_stimulus))"""

    
    #Cleaning from bad channels
    ch_i = [i for i, ch in enumerate(chnames) if ch in bad_chans]
    clean_chnames = [ch for i, ch in enumerate(chnames) if ch not in bad_chans]
    
    clean_music = np.delete(data_m, ch_i, axis=0)
    clean_speech = np.delete(data_s, ch_i, axis=0)
    clean_rest = np.delete(data_r, ch_i, axis=0)

    #selecting only the channels we want
    ch_H_i= [i for i, ch in enumerate(clean_chnames) if ch not in ch_H]
    ch_wH_i= [i for i, ch in enumerate(clean_chnames) if ch in ch_H]
    

    clean_music_H = np.delete(clean_music, ch_H_i, axis=0)
    clean_speech_H = np.delete(clean_speech, ch_H_i, axis=0)
    clean_rest_H = np.delete(clean_rest, ch_H_i, axis=0)
    
    clean_music_without_H = np.delete(clean_music, ch_wH_i, axis=0)
    clean_speech_without_H = np.delete(clean_speech, ch_wH_i, axis=0)
    clean_rest_without_H = np.delete(clean_rest, ch_wH_i, axis=0)
    
    clean_speech2=set()
    clean_music2=set()
    clean_rest2=set()
    
    art_speech_times[subject]=set(np.argwhere(stats.zscore(clean_speech, axis=1) >N)[:,1])
    
    art_music_times[subject]=set(np.argwhere(stats.zscore(clean_music, axis=1) > N)[:,1])
    
    art_rest_times[subject]=set(np.argwhere(stats.zscore(clean_rest, axis=1) > N)[:,1])

set_speech=art_speech_times[subject_list[0]]
set_music=art_music_times[subject_list[0]]
set_rest=art_rest_times[subject_list[0]]

for subject in subject_list:
    set_speech.update(art_speech_times[subject])
    set_music.update(art_music_times[subject])
    set_rest.update(art_rest_times[subject])
    
good_times_speech=np.zeros(len(clean_music.T)+1000)
good_times_music=np.zeros(len(clean_music.T)+1000)
good_times_rest=np.zeros(len(clean_music.T)+1000)

good_times_speech[list(set_speech)]=1
good_times_music[list(set_music)]=1
good_times_rest[list(set_rest)]=1

sav.saving(set_speech, "set_speech_bad_times_100hz")
sav.saving(set_music, "set_music_bad_times_100hz")
sav.saving(set_rest, "set_rest_bad_times_100hz")

plt.plot(good_times_speech)
plt.show()
plt.close()

plt.plot(good_times_music)
plt.show()
plt.close()
plt.plot(good_times_rest)
plt.show()
plt.close()
  
    
    
    
    
    
    
    
    
        
    
    
    
