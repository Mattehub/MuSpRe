#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:33:17 2022

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




def go_edge_list(tseries, edge_list):
    
    matrix_E=[]
    for edge in edge_list:
        i,j = edge
        E=np.multiply((tseries[i], tseries[j]))
        matrix_E.append(E)
        
    matrix_E=np.array(matrix_E)

    return(matrix_E)


def go_edge(tseries):
    nregions=tseries.shape[1]
    Blen=tseries.shape[0]
    nedges=int(nregions**2/2-nregions/2)
    iTriup= np.triu_indices(nregions,k=1) 
    gz=stats.zscore(tseries)
    Eseries = gz[:,iTriup[0]]*gz[:,iTriup[1]]
    return Eseries

def clean2(x1, N=3):
    #x1 is an matrix, channels*time
    #N is an input that how many times the variance is considered an artifacts.
    
    #We copy the data to not modify the original raw
    x=x1.copy()
    
    absx=np.absolute(x)
    
    #mean and standard deviation of the absolute values in the raw data
    s=np.std(absx)
    m=np.mean(absx)
    
    #where the distance betwenn the absolute value of the activity is the mean absolute values of 
    #activities is bigger than N*std, we substitute with a nan value
    x[absx-m>N*s]=np.nan
    
    #We do the mean again without consider the nan values
    m=np.nanmean(x)
    
    #Here we have the list of indeces of the nan values
    a=np.argwhere(np.isnan(x)==True)
    
    for indeces in a:
        
        k,j = indeces
        
        #in case the nan correspond to the first measurement we substitute with the mean
        if j==0:
            x[k,j]=m
        
        #in case not, we substitute with the value measured at the previous instant of time in the same channel
        else:
            x[k,j]=x[k,j-1]
    return x

def clean1(x1, N=5):
    #x1 is an matrix, channels*time
    #N is an input that how many times the variance is considered an artifacts.
    
    #We copy the data to not modify the original raw
    x=x1.copy()
    
    #mean and standard deviation of the raw
    s=np.std(x)
    m=np.mean(x)

    #where the distance betwenn the data and the mean is bigger than N*std, we substitute with a nan value
    x[np.abs(x-m)>N*s]=np.nan
    
    #We do the mean again without consider the nan value
    m=np.nanmean(x)
    
    #Here we have the list of indeces of the nan values
    a=np.argwhere(np.isnan(x)==True)
    
    
    for indeces in a:
        
        k,j = indeces
        
        #in case the nan correspond to the first measurement we substitute with the mean
        if j==0:
            x[k,j]=m
        
        #in case not, we substitute with the value measured at the previous instant of time in the same channel
        else:
            x[k,j]=x[k,j-1]
            
    return x
    


subject_list=subject_list

time_corr_speech=[]

time_corr_music=[]

time_corr_rest=[]

for isub, subject in enumerate(subject_list):
## Load the data from the HDF fil
    print(subject, isub)
    
    #MUSIC
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
    
    clean_music = np.delete(data_m, ch_i, axis=0)
    clean_speech = np.delete(data_s, ch_i, axis=0)
    clean_rest = np.delete(data_r, ch_i, axis=0)

#selecting only the channels we want, in this script H
    ch_H_i= [i for i, ch in enumerate(clean_chnames) if ch not in ch_H]
    final_channels=[ch for i, ch in enumerate(clean_chnames) if i not in ch_H_i]
    print(final_channels)
    clean_music_H = np.delete(clean_music, ch_H_i, axis=0)
    clean_speech_H = np.delete(clean_speech, ch_H_i, axis=0)
    clean_rest_H = np.delete(clean_rest, ch_H_i, axis=0)
    
    clean_music_without_H = np.delete(clean_music, ch_H_i, axis=0)
    clean_speech_without_H = np.delete(clean_speech, ch_H_i, axis=0)
    clean_rest_without_H = np.delete(clean_rest, ch_H_i, axis=0)
    
    #clean_mu=clean2(clean_music_H, N=3)
    #clean_sp=clean2(clean_speech_H, N=3)
    #clean_re=clean2(clean_rest_H, N=3)
    
    zdata_speech=stats.zscore(clean_speech)
    zdata_music=stats.zscore(clean_music)
    zdata_rest=stats.zscore(clean_rest)

    
    #SPEECH
    
    #t_tot=len(zdata_speech[1,:])
    x=zdata_speech[:,:20000]
    x=x.T
    edge=go_edge(x)
    '''for i in [1,30,100, 300,500,1000,1400,4000,5000]:
        plt.plot(edge.T[i])
        plt.show()
        plt.close()
        plt.hist(edge.T[i])
        plt.show()
        plt.close()'''
        
    
    rss=np.sqrt(np.sum(edge**2, axis=1))
    #order=np.argsort(np.abs(stats.zscore(rss)))
    
    #order=np.argsort(rss)
    
    """plt.plot(rss[:1000])
    plt.show()
    plt.close()
    
    plt.plot(rss[300:600])
    plt.show()
    plt.close()
    
    plt.plot(rss[600:900])
    plt.show()
    plt.close()
    
    plt.plot(rss[900:1200])
    plt.show()
    plt.close()"""
    
    #corr_time=np.corrcoef(edge)[np.ix_(order,order)]
    
    corr_time=np.corrcoef(edge)
    
    time_corr_speech.append(np.sum(corr_time, axis=0))
    plt.figure(figsize=(12,8))
    plt.imshow(corr_time, aspect='auto', vmin=-0.05, vmax=0.05)
    plt.colorbar()
    plt.tight_layout()
    plt.title('speech')
    plt.show()
    plt.close()
    
    #MUSIC
    
    x=zdata_music[:,:20000]
    x=x.T
    edge=go_edge(x)
    """
    for i in [1,30,56,76]:
        plt.plot(edge.T[i])
        plt.show()
        plt.close()
        plt.hist(edge.T[i])
        plt.show()
        plt.close()"""
        
    rss=np.sqrt(np.sum(edge**2, axis=1))
    #order=np.argsort(np.abs(stats.zscore(rss)))
    order=np.argsort(rss)
    
    #corr_time=np.corrcoef(edge)[np.ix_(order,order)]
    
    corr_time=np.corrcoef(edge)
    
    time_corr_music.append(np.sum(corr_time, axis=0))
    
    plt.figure(figsize=(12,8))
    plt.imshow(corr_time, aspect='auto', vmin=-0.05, vmax=0.05)
    plt.colorbar()
    plt.tight_layout()
    plt.title('music')
    plt.show()
    plt.close()
    #REST
    
    x=zdata_rest[:,:20000]
    x=x.T
    edge=go_edge(x)
    """
    for i in [1,30,100, 300,500,1000,56,243,1400,2000,3000,4000,5000]:
        plt.plot(edge.T[i])
        plt.show()
        plt.close()
        plt.hist(edge.T[i])
        plt.show()
        plt.close()"""
        
    rss=np.sqrt(np.sum(edge**2, axis=1))
    
    #order=np.argsort(np.abs(stats.zscore(rss)))
    order=np.argsort(rss)
    
    #corr_time=np.corrcoef(edge)[np.ix_(order,order)]
    
    corr_time=np.corrcoef(edge)
    
    time_corr_rest.append(np.sum(corr_time, axis=0))
    
    plt.figure(figsize=(12,8))
    plt.imshow(corr_time, aspect='auto', vmin=-0.05, vmax=0.05)
    plt.colorbar()
    plt.tight_layout()
    plt.title('rest')
    plt.show()
    plt.close()
"""    
    
time_corr_speech=np.array(time_corr_speech)
corr_time_corr_speech=np.corrcoef(time_corr_speech)
plt.figure(figsize=(12,8))
plt.imshow(corr_time_corr_speech, aspect="auto")
plt.colorbar()
plt.tight_layout()
plt.title('speech')
plt.show()
plt.close()

time_corr_music=np.array(time_corr_music)
corr_time_corr_music=np.corrcoef(time_corr_music)
plt.figure(figsize=(12,8))
plt.imshow(corr_time_corr_music, aspect="auto")
plt.colorbar()
plt.tight_layout()
plt.title('music')
plt.show()
plt.close()

time_corr_rest=np.array(time_corr_rest)
corr_time_corr_rest=np.corrcoef(time_corr_rest)
plt.figure(figsize=(12,8))
plt.imshow(corr_time_corr_rest, aspect="auto")
plt.colorbar()
plt.tight_layout()
plt.title('rest')
plt.show()
plt.close() """  
    
    
    
    
    
    
    
    
    