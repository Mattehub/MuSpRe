#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:17:51 2022

@author: jeremy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:03:46 2022

@author: jeremy
"""

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
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import colors
import os
import pickle
import utils_avalanches as av
import warnings 
import Utils_FC as fc
from lempel_ziv_complexity import lempel_ziv_complexity

warnings.simplefilter('ignore')

#path='/home/jeremy/anaconda3/matteo/'
path='C:/Users/matte/OneDrive/Documenti/matteo/'
#CREATING THE LIST OF SUBJECTS

sound_list=['rest','music','speech']
arr_mu = os.listdir(path +'seeg_fif_data/music')
arr_rest = os.listdir(path +'seeg_fif_data/speech')
arr_speech = os.listdir(path +'seeg_fif_data/rest')

subject_set_mu=set()
subject_set_speech=set()
subject_set_rest=set()

for st in arr_mu: 
    #print(st)
    subject_set_mu.add(st.partition('_')[0])
    #print(st.partition('_')[0])
    
for st in arr_speech:
    subject_set_speech.add(st.partition('_')[0])
    
for st in arr_rest:
    subject_set_rest.add(st.partition('_')[0])

subject_list=list(subject_set_mu.intersection(subject_set_speech,subject_set_rest))

#Here I create a set of the  all channels
total_channels_set=set()

for subject in subject_list:
    with h5py.File(pjoin(path +'seeg_data_h5py/h5_electrodes/', subject + '_electrodes.hdf5'), 'r') as f:
        #print(f.keys())
        #print('chnames', f['chnames'].shape)
        
        chnames = f['chnames'][...].astype('U')
        total_channels_set.update(chnames)
        
#print(total_channels_set)


#Here I create a set of the H channels
ch_H=set()
for ch in total_channels_set:
    
    if "H" in ch:
        ch_H.add(ch)
        
ch_IM=set()
for ch in total_channels_set:
    
    if "IP" in ch:
        ch_IM.add(ch)
#print(ch_H)


#PARAMETERS

subject_list=subject_list
min_sizes=np.arange(1,20,2)

size_rest=[]
size_speech=[]
size_music=[]

mean_vector_speech=[]
mean_vector_music=[]
mean_vector_rest=[]

subject_iai_speech=[]
subject_iai_music=[]
subject_iai_rest=[]

fc_dict={}

final_channels_without_H={}

final_channels_H={}

final_channels_all={}

rss_speech=[]
rss_music=[]
rss_rest=[]

complexity_list={}
    
for isub, subject in enumerate(subject_list):
    complexity_list[subject]=[]
    fc_dict[subject]={}
    
    #MUSIC
    with h5py.File(pjoin(path+'seeg_data_hgenv_down100_h5py/', subject + '_down100_seeg_preproc.hdf5'), 'r') as f:
        data_m=f['music'][...]
    
    #SPEECH
    with h5py.File(pjoin(path+'seeg_data_hgenv_down100_h5py/', subject + '_down100_seeg_preproc.hdf5'), 'r') as f:
        data_s=f['speech'][...]

    #REST
    with h5py.File(pjoin(path+'seeg_data_hgenv_down100_h5py/', subject + '_down100_seeg_preproc.hdf5'), 'r') as f:
        data_r=f['rest'][...]
        
# redefine path
# below example of loading of music data.

    with h5py.File(pjoin(path+ 'seeg_data_h5py/h5_electrodes/', subject + '_electrodes.hdf5'), 'r') as f:
        chnames = f['chnames'][...].astype('U')

    with h5py.File(pjoin(path + 'seeg_data_h5py/h5_misc/', subject + '_misc.hdf5'), 'r') as f:
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
    ch_H_w_i= [i for i, ch in enumerate(clean_chnames) if ch in ch_H]
    
    final_channels_without_H[subject]=[ch for i, ch in enumerate(clean_chnames) if i in ch_H_i]
    final_channels_H[subject]=[ch for i, ch in enumerate(clean_chnames) if i not in ch_H_i]
    final_channels_all[subject]=clean_chnames
    
    clean_music_H = np.delete(clean_music, ch_H_i, axis=0)
    clean_speech_H = np.delete(clean_speech, ch_H_i, axis=0)
    clean_rest_H = np.delete(clean_rest, ch_H_i, axis=0)
    
    clean_music_without_H = np.delete(clean_music, ch_H_w_i, axis=0)
    clean_speech_without_H = np.delete(clean_speech, ch_H_w_i, axis=0)
    clean_rest_without_H = np.delete(clean_rest, ch_H_w_i, axis=0)
    
    #clean_mu=clean2(clean_music_H, N=3)
    #clean_sp=clean2(clean_speech_H, N=3)
    #clean_re=clean2(clean_rest_H, N=3)
    
    zdata_speech_art=stats.zscore(clean_speech_H, axis=1)
    zdata_music_art=stats.zscore(clean_music_H, axis=1)
    zdata_rest_art=stats.zscore(clean_rest_H, axis=1)
    
    zdata_speech=np.where(np.abs(zdata_speech_art)>7, 0, zdata_speech_art)
    zdata_music=np.where(np.abs(zdata_music_art)>7, 0, zdata_music_art)
    zdata_rest=np.where(np.abs(zdata_rest_art)>7, 0, zdata_rest_art)
    
    speech_data_av=zdata_speech.copy()
    music_data_av=zdata_music.copy()
    rest_data_av=zdata_rest.copy()
    
    thres=np.percentile(zdata_rest, 99)
    print(thres)
    
    
    
    
    for n in np.arange(int(len(zdata_rest)/3), int(len(zdata_rest)/2)):
        
        complexity_list_speech=[]
        complexity_list_music=[]
        complexity_list_rest=[]
        
        avalanches_rest =av.go_avalanches(zdata_rest.T, thre=thres, direc=0, binsize=2)
        
        indices15=np.argwhere(np.array(avalanches_rest['siz'])==n)
        avalanches15=[]
        
        for i in indices15:
            a0=avalanches_rest['ranges'][i[0]][0]
            a1=avalanches_rest['ranges'][i[0]][1]
            proj=np.sum(avalanches_rest['Zbin'][a0:a1,:], axis=0)
            avalanches15.append(np.where(proj>0, 1, 0))
            string=''.join([str(elem) for elem in avalanches_rest['Zbin'][a0:a1,:].T.flatten()])
            complexity_list_rest.append(lempel_ziv_complexity(string))
        
        avalanches_speech=av.go_avalanches(zdata_speech.T, thre=thres, direc=0, binsize=2)

        indices15=np.argwhere(np.array(avalanches_speech['siz'])==n)
        avalanches15=[]
        size_speech.append(avalanches_speech['siz'])
    
        for i in indices15:
            a0=avalanches_speech['ranges'][i[0]][0]
            a1=avalanches_speech['ranges'][i[0]][1]
            proj=np.sum(avalanches_speech['Zbin'][a0:a1,:], axis=0)
            avalanches15.append(np.where(proj>0, 1, 0))
            string=''.join([str(elem) for elem in avalanches_speech['Zbin'][a0:a1,:].T.flatten()])
            complexity_list_speech.append(lempel_ziv_complexity(string))
        
        avalanches_music =av.go_avalanches(zdata_music.T, thre=thres, direc=0, binsize=2)
        size_music.append(avalanches_music['siz'])
        
        indices15=np.argwhere(np.array(avalanches_music['siz'])==n)
        avalanches15=[]
        for i in indices15:
            a0=avalanches_music['ranges'][i[0]][0]
            a1=avalanches_music['ranges'][i[0]][1]
            proj=np.sum(avalanches_music['Zbin'][a0:a1,:], axis=0)
            avalanches15.append(np.where(proj>0, 1, 0))
            string=''.join([str(elem) for elem in avalanches_music['Zbin'][a0:a1,:].T.flatten()])
            complexity_list_music.append(lempel_ziv_complexity(string))
        
        complexity_list[subject].append([np.nanmean(complexity_list_rest), np.nanmean(complexity_list_music), np.nanmean(complexity_list_speech)])
        """
        a=max(len(complexity_list_speech), len(complexity_list_music), len(complexity_list_rest))
        y,x,_=plt.hist(stats.zscore(complexity_list_speech[:a]), 15, color="lightblue", label='LZC speech')
        y1,x1,_=plt.hist(stats.zscore(complexity_list_music[:a]), 15, color="green", label='LZC music')
        y2,x2,_=plt.hist(stats.zscore(complexity_list_rest[:a]), 15, color="red", label='LZC rest')
        plt.legend()
        plt.show()
        plt.close()
        
        plt.plot(x[:-1],y, color="lightblue", label='LZC speech')
        plt.plot(x1[:-1],y1, color="green", label='LZC music')
        plt.plot(x2[:-1],y2, color="red", label='LZC rest')
        plt.show()
        plt.close()"""
 
for subject in subject_list:
    plt.plot(np.nanmean(np.array(complexity_list[subject]), axis=0))
plt.title('LZ complexity')
plt.xticks(np.arange(3), ['rest', 'music', 'speech'], rotation=90)
plt.show()
plt.close()       
   

plt.hist(complexity_list_rest, 30, color="red", label='LZC rest')
plt.hist(complexity_list_speech, 30, color="lightblue", label='LZC speech')
plt.hist(complexity_list_music, 30, color="green", label='LZC music')
plt.legend()
plt.show()
plt.close()


