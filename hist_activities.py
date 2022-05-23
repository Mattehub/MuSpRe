#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:42:41 2022

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




mean_rest=[]
std_rest=[]

mean_music=[]
std_music=[]

mean_speech=[]
std_speech=[]

fig, axs = plt.subplots(3,6)
for isub, subject in enumerate(sub_list):
    print(isub)
    with h5py.File(pjoin('seeg_data_h_env_down_h5py/', subject + '_down_seeg_preproc.hdf5'), 'r') as f:
        print(f.keys())
        print('music', f['music'].shape)

        #data_music[subject]=f['music'][...]
        data_m=f['music'][...]
    
    #SPEECH
    with h5py.File(pjoin('seeg_data_h_env_down_h5py/', subject + '_down_seeg_preproc.hdf5'), 'r') as f:
        print(f.keys())
        print('speech', f['speech'].shape)
        print('speech', f['speech'].shape)
        #data_speech[subject]= f['speech'][...]
        data_s=f['speech'][...]

    #REST
    with h5py.File(pjoin('seeg_data_h_env_down_h5py/', subject + '_down_seeg_preproc.hdf5'), 'r') as f:
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


## Cleaning from artifacts

    ch_i = [i for i, ch in enumerate(chnames) if ch in bad_chans]
    clean_chnames = [ch for i, ch in enumerate(chnames) if ch not in bad_chans]
    
    clean_music = np.delete(data_m, ch_i, axis=0)[:,:20000]
    clean_speech = np.delete(data_s, ch_i, axis=0)[:,:20000]
    clean_rest = np.delete(data_r, ch_i, axis=0)[:,:20000]

#selecting only the channels we want, in this script H
    ch_H_i= [i for i, ch in enumerate(clean_chnames) if ch not in ch_H]
    ch_H_w_i= [i for i, ch in enumerate(clean_chnames) if ch in ch_H]
    final_channels=[ch for i, ch in enumerate(clean_chnames) if i not in ch_H_i]
    print(final_channels)
    clean_music_H = np.delete(clean_music, ch_H_i, axis=0)[:,:20000]
    clean_speech_H = np.delete(clean_speech, ch_H_i, axis=0)[:,:20000]
    clean_rest_H = np.delete(clean_rest, ch_H_i, axis=0)[:,:20000]
    
    clean_music_without_H = np.delete(clean_music, ch_H_w_i, axis=0)
    clean_speech_without_H = np.delete(clean_speech, ch_H_w_i, axis=0)
    clean_rest_without_H = np.delete(clean_rest, ch_H_w_i, axis=0)
    
    #clean_mu=clean2(clean_music_H, N=3)
    #clean_sp=clean2(clean_speech_H, N=3)
    #clean_re=clean2(clean_rest_H, N=3)
    
    zdata_speech=stats.zscore(clean_speech_H)[:,:20000]
    zdata_music=stats.zscore(clean_music_H)[:,:20000]
    zdata_rest=stats.zscore(clean_rest_H)[:,:20000]
    
    zds=np.concatenate(clean_speech_H)
    zdm=np.concatenate(clean_music_H)
    zdr=np.concatenate(clean_rest_H)
    
    xxr=plt.hist(np.abs(zdr),100)

    xxm=plt.hist(np.abs(zdm),100)

    xxs=plt.hist(np.abs(zds),100)
    plt.close()
    
    """if isub<6:
        j=0
        i=isub
    elif isub<12:
        j=1
        i=isub-6
    else:
        j=2
        i=isub-12
    
    axs[j,i].plot(xxr[1][:-1],xxr[0])
    
    axs[j,i].plot(xxm[1][:-1],xxm[0])

    axs[j,i].plot(xxs[1][:-1],xxs[0])"""
    
    plt.plot(xxr[1][:-1],xxr[0], label='rest')
    mean_rest.append(np.mean(np.abs(zdr)))
    std_rest.append(np.std(np.abs(zdr)))
    
    print('mean during rest', np.mean(np.abs(zdr)))
    print('variance during rest', np.std(np.abs(zdr)))
    
    
    plt.plot(xxm[1][:-1],xxm[0], label='music')
    
    mean_music.append(np.mean(np.abs(zdm)))
    std_music.append(np.std(np.abs(zdm)))
    
    print('mean during music', np.mean(np.abs(zdm)))
    print('variance during music', np.std(np.abs(zdm)))
    
    plt.plot(xxs[1][:-1],xxs[0], label='speech')
    
    mean_speech.append(np.mean(np.abs(zds)))
    std_speech.append(np.std(np.abs(zds)))
    
    print('mean during speech', np.mean(np.abs(zds)))
    print('variance during speech', np.std(np.abs(zds)))
    
    plt.legend()
    
    plt.yscale('log')
    
    plt.show()
    plt.close()

plt.plot(mean_rest, label='mean_rest')
plt.plot(mean_music, label='mean_music')
plt.plot(mean_speech, label='mean_speech')
plt.legend()
plt.show()
plt.close()

plt.plot(std_rest, label='std_rest')
plt.plot(std_music, label='std_music')
plt.plot(std_speech, label='std_speech')

plt.legend()

plt.show()
plt.close()

x=[np.mean(mean_rest), np.mean(mean_music), np.mean(mean_speech)]


error=[np.std(mean_rest), np.std(mean_speech), np.std(mean_music)]

plt.errorbar(range(len(x)), x, error, linestyle='None', marker='^')