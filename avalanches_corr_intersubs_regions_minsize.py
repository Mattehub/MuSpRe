





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

path='/home/jeremy/anaconda3/matteo/'
#path='C:/Users/matte/OneDrive/Documenti/matteo/'

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

complexity_list_speech=[]
complexity_list_music=[]
complexity_list_rest=[]
    
for isub, subject in enumerate(subject_list):
## Load the data from the HDF fil
    print(subject, isub)
    
    fc_dict[subject]={}
    
    #MUSIC
    with h5py.File(pjoin(path+'seeg_data_hgenv_down100_h5py/', subject + '_down100_seeg_preproc.hdf5'), 'r') as f:
        print(f.keys())
        print('music', f['music'].shape)

        #data_music[subject]=f['music'][...]
        data_m=f['music'][...]
    
    #SPEECH
    with h5py.File(pjoin(path+'seeg_data_hgenv_down100_h5py/', subject + '_down100_seeg_preproc.hdf5'), 'r') as f:
        print(f.keys())
        print('speech', f['speech'].shape)
        print('speech', f['speech'].shape)
        #data_speech[subject]= f['speech'][...]
        data_s=f['speech'][...]

    #REST
    with h5py.File(pjoin(path+'seeg_data_hgenv_down100_h5py/', subject + '_down100_seeg_preproc.hdf5'), 'r') as f:
        print(f.keys())
        print('rest', f['rest'].shape)
        print('rest', f['rest'].shape)
        #data_rest[subject]=f['rest'][...]
        data_r=f['rest'][...]



# redefine path
# below example of loading of music data.

    with h5py.File(pjoin(path+ 'seeg_data_h5py/h5_electrodes/', subject + '_electrodes.hdf5'), 'r') as f:
        print(f.keys())
        print('chnames', f['chnames'].shape)
    
        chnames = f['chnames'][...].astype('U')

    with h5py.File(pjoin(path + 'seeg_data_h5py/h5_misc/', subject + '_misc.hdf5'), 'r') as f:
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
    
    zdata_speech_art=stats.zscore(clean_speech_without_H, axis=1)
    zdata_music_art=stats.zscore(clean_music_without_H, axis=1)
    zdata_rest_art=stats.zscore(clean_rest_without_H, axis=1)
    
    zdata_speech=np.where(np.abs(zdata_speech_art)>7, 0, zdata_speech_art)
    zdata_music=np.where(np.abs(zdata_music_art)>7, 0, zdata_music_art)
    zdata_rest=np.where(np.abs(zdata_rest_art)>7, 0, zdata_rest_art)
    
    speech_data_av=zdata_speech.copy()
    music_data_av=zdata_music.copy()
    rest_data_av=zdata_rest.copy()
    
    thres=np.percentile(zdata_rest, 99)
    print(thres)
    
    #CREATING THE AVANLANCHES DICTIONARIES
    
    avalanches_rest =av.go_avalanches(zdata_rest.T, thre=thres, direc=0, binsize=2)
    avalanches_speech=av.go_avalanches(zdata_speech.T, thre=thres, direc=0, binsize=2)
    avalanches_music =av.go_avalanches(zdata_music.T, thre=thres, direc=0, binsize=2)
    
    
    
    min_siz=int(len(zdata_rest)/3)
    
    mins_size_speech=av.min_siz_filt(avalanches_speech, min_siz)
    
    mins_size_music=av.min_siz_filt(avalanches_music, min_siz)
    
    mins_size_rest=av.min_siz_filt(avalanches_rest, min_siz)
    
    #comuting the sum of all the activities at each instant of time
    rss_rest.append(np.sum(mins_size_rest['Zbin_reduced'].T, axis=0))
    
    rss_speech.append(np.sum(mins_size_speech['Zbin_reduced'].T, axis=0))
  
    rss_music.append(np.sum(mins_size_music['Zbin_reduced'].T, axis=0))
    
    #plotting a visualization of the avalanches
    y=np.arange(0,len(final_channels_without_H[subject]),2)
    plt.figure(figsize=(15,13))
    plt.imshow(mins_size_speech['Zbin_reduced'].T, aspect='auto', interpolation='none', vmin=-0.3, vmax=0.3)
    plt.yticks(y, final_channels_without_H[subject][::2])
    plt.title('rest, after binarization, threshold='+str(thres))
    plt.colorbar()
    plt.show()
    plt.close()
    """
    plt.figure(figsize=(15,13))
    plt.imshow(mins_size_music['Zbin_reduced'].T, aspect='auto', interpolation='none', vmin=-0.3, vmax=0.3)
    plt.title('speech, after binarization, threshold='+str(thres))
    plt.yticks(y, clean_chnames[::2])
    plt.colorbar()
    plt.show()
    plt.close()
    
    plt.figure(figsize=(15,13))
    plt.imshow(mins_size_rest['Zbin_reduced'].T, aspect='auto', interpolation='none', vmin=-0.3, vmax=0.3)
    plt.title('music, after binarization, threshold='+str(thres))
    plt.yticks(y, clean_chnames[::2])
    plt.colorbar()
    plt.show()
    plt.close()
    """
#plotting the rss
a=-3
b=3
plt.figure(figsize=(12,10))
plt.imshow(stats.zscore(rss_speech, axis=1), aspect='auto', vmin=a, vmax=b)
plt.title('speech')
plt.colorbar()
plt.show()
plt.close()

plt.figure(figsize=(12,10))
plt.imshow(stats.zscore(rss_music, axis=1), aspect='auto', vmin=a, vmax=b)
plt.title('music')
plt.colorbar()
plt.show()
plt.close()

plt.figure(figsize=(12,10))
plt.imshow(stats.zscore(rss_rest, axis=1), aspect='auto', vmin=a, vmax=b)
plt.title('rest')
plt.colorbar()
plt.show()
plt.close()
    
#plotting the correlation matrix
b=0.07
a=0
corr_matrix_speech=np.corrcoef(np.array(rss_speech))
plt.figure(figsize=(12,10))
plt.imshow(corr_matrix_speech, aspect='auto', vmin=a, vmax=b)
plt.title('speech')
plt.colorbar()
plt.show()
plt.close()

print('the mean correlation between subjects during speech is', np.mean(np.triu(corr_matrix_speech, 1)))

corr_matrix_music=np.corrcoef(np.array(rss_music))
plt.figure(figsize=(12,10))
plt.imshow(corr_matrix_music, aspect='auto', vmin=a, vmax=b)
plt.title('music')
plt.colorbar()
plt.show()
plt.close()

print('the mean correlation between subjects during music is', np.mean(np.triu(corr_matrix_music, 1)))

corr_matrix_rest=np.corrcoef(np.array(rss_rest))
plt.figure(figsize=(12,10))
plt.imshow(corr_matrix_rest, aspect='auto', vmin=a, vmax=b)
plt.title('rest')
plt.colorbar()
plt.show()
plt.close()

print('the mean correlation between subjects during rest is', np.mean(np.triu(corr_matrix_rest, 1)))

#DOING A SCRAMBLING, SHIFTING THE RSS ON TIME

list_mean_corr_speech=[]

list_mean_corr_music=[]

list_mean_corr_rest=[]

num_sim=1000
for i in range(num_sim):
    
    rss_array_music_shift=fc.shifting_matrix(np.array(rss_music))
    list_mean_corr_music.append(np.mean(np.corrcoef(rss_array_music_shift)[np.triu_indices(19, k = 1)]))
    
    rss_array_speech_shift=fc.shifting_matrix(np.array(rss_speech))
    list_mean_corr_speech.append(np.mean(np.corrcoef(rss_array_speech_shift)[np.triu_indices(19, k = 1)]))
    
    rss_array_rest_shift=fc.shifting_matrix(np.array(rss_rest))
    list_mean_corr_rest.append(np.mean(np.corrcoef(rss_array_rest_shift)[np.triu_indices(19, k = 1)]))
    
#PLOTTING THE RESULT, COMPARISON BETWEEN THE RANDOMIZATION AND THE DATA
plt.hist(list_mean_corr_music, label='random shift')
plt.axvline(x=np.mean(np.corrcoef(np.array(rss_music))[np.triu_indices(19, k = 1)]), label='our result')
plt.title('music')
plt.legend()
plt.show()
plt.close()

plt.hist(list_mean_corr_speech, label='random shift')
plt.axvline(x=np.mean(np.corrcoef(np.array(rss_speech))[np.triu_indices(19, k = 1)]), label='our relsult')
plt.title('speech')
plt.legend()
plt.show()
plt.close()

plt.hist(list_mean_corr_rest, label='random shift')
plt.axvline(x=np.mean(np.corrcoef(np.array(rss_rest))[np.triu_indices(19, k = 1)]), label='our relsult')
plt.title('rest')
plt.legend()
plt.show()
plt.close()
    
    