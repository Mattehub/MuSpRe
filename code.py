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


print(subject_list)
print(len(subject_list))

"""def go_edge_list(x, edge_list):
    for edge in edge_list:
        i,j = edge
        E=np.multiply((data[]))"""
    
    
    
    
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
    

#length of the interval to analyse in one step 
t=30000
t_tot=260000
edge_music={}
edge_speech={}
edge_rest={}
rss_music={}
rss_speech={}
rss_rest={}
data_music={}
data_speech={}
data_rest={}


for isub, subject in enumerate(subject_list):
## Load the data from the HDF fil
    print(subject, isub)
    
    #MUSIC
    with h5py.File(pjoin('seeg_data_h5py/', subject + '_seeg_preproc.hdf5'), 'r') as f:
        print(f.keys())
        print('music', f['music'].shape)

        #data_music[subject]=f['music'][...]
        data_m=f['music'][...]
    
    #SPEECH
    with h5py.File(pjoin('seeg_data_h5py/', subject + '_seeg_preproc.hdf5'), 'r') as f:
        print(f.keys())
        print('speech', f['speech'].shape)
        print('speech', f['speech'].shape)
        #data_speech[subject]= f['speech'][...]
        data_s=f['speech'][...]

    #REST
    with h5py.File(pjoin('seeg_data_h5py/', subject + '_seeg_preproc.hdf5'), 'r') as f:
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
    
    clean_mu=clean2(clean_music, N=3)
    clean_sp=clean2(clean_speech, N=3)
    clean_re=clean2(clean_rest, N=3)
    
    zdata_speech=stats.zscore(clean_sp)
    zdata_music=stats.zscore(clean_mu)
    zdata_rest=stats.zscore(clean_re)
    
    #SPEECH
    
    #t_tot=len(zdata_speech[1,:])
    num=int(t_tot/t)
    t_list=[]
    for i in range(num):
        t_list.append(t*i)
        
    if t_list[-1]!=t_tot:
        t_list.append(t_tot)
    
    
    for j in range(len(t_list)-1):
        
        x=zdata_speech[:,t_list[j]:t_list[j+1]].T
        #x=np.where(abs(x)>5, abs(x)*5/x, x)
        
        print('Done interval', j)
        
        edge_speech=go_edge(x)
        if subject in rss_speech:
            rss_speech[subject]=np.concatenate((np.sqrt(np.sum(edge_speech**2, axis=1)),rss_speech[subject]))
        else:
            rss_speech[subject]=np.sqrt(np.sum(edge_speech**2, axis=1))
            
    #MUSIC
    
    #t_tot=len(zdata_music[1,:])
    num=int(t_tot/t)
    t_list=[]
    for i in range(num):
        t_list.append(t*i)
        
    if t_list[-1]!=t_tot:
        t_list.append(t_tot)
    
    
    for j in range(len(t_list)-1):
        x=zdata_music[:,t_list[j]:t_list[j+1]].T
        #x=np.where(abs(x)>5, abs(x)*5/x, x)
    
        edge_music=go_edge(x)
        if subject in rss_music:
            rss_music[subject]=np.concatenate((np.sqrt(np.sum(edge_music**2, axis=1)),rss_music[subject]))
        else:
            rss_music[subject]=np.sqrt(np.sum(edge_music**2, axis=1))
        
    #REST
    
    #t_tot=len(zdata_rest[1,:])
    num=int(t_tot/t)
    t_list=[]
    for i in range(num):
        t_list.append(t*i)
        
    if t_list[-1]!=t_tot:
        t_list.append(t_tot)
    
    
    for j in range(len(t_list)-1):
        x=zdata_rest[:,t_list[j]:t_list[j+1]].T
        #x=np.where(abs(x)>5, abs(x)*5/x, x)
    
        edge_rest=go_edge(x)
        if subject in rss_rest:
            rss_music[subject]=np.concatenate((np.sqrt(np.sum(edge_rest**2, axis=1)),rss_rest[subject]))
        else:
            rss_rest[subject]=np.sqrt(np.sum(edge_rest**2, axis=1))
    print('')
    print('')
    
print('end data, start plotting')
#saving
"""
#SPEECH
with open('edge_speech_dict.pickle', 'wb') as f:
    pickle.dump(edge_speech, f)

with open ('rss_speech_dict', 'wb') as f:
    pickle.dump(rss_speech, f)

#MUSIC
with open('edge_music_dict.pickle', 'wb') as f:
    pickle.dump(edge_music, f)

with open ('rss_music_dict', 'wb') as f:
    pickle.dump(rss_music, f)

#REST
with open('edge_rest_dict.pickle', 'wb') as f:
    pickle.dump(edge_rest, f)

with open ('rss_rest_dict', 'wb') as f:
    pickle.dump(rss_rest, f)
"""
#preplotting and plotting

#SPEECH
rss_list_speech=[]
for subject in subject_list:
    rss_list_speech.append(rss_speech[subject].T)

rss_array_speech=np.array(rss_list_speech)
    
plt.figure(figsize=(18,8))
plt.imshow(rss_array_speech, aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.title('speech')
plt.show()
plt.close()

#MUSIC
rss_list_music=[]
for subject in subject_list:
    rss_list_music.append(rss_music[subject].T)

rss_array_music=np.array(rss_list_music)
    
plt.figure(figsize=(18,8))
plt.imshow(rss_array_music, aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.title('music')
plt.show()
plt.close()

#REST
rss_list_rest=[]
for subject in subject_list:
    rss_list_rest.append(rss_rest[subject].T)

rss_array_rest=np.array(rss_list_rest)
    
plt.figure(figsize=(18,8))
plt.imshow(rss_array_rest, aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.title('rest')
plt.show()
plt.close()

plt.figure(figsize=(18,8))
plt.imshow(np.corrcoef(rss_array_speech), aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.title('speech')
plt.show()
plt.close()

print('The mean correlation during speech listening is', np.mean(np.corrcoef(rss_array_speech)))
plt.figure(figsize=(18,8))
plt.imshow(np.corrcoef(rss_array_music), aspect='auto',)
plt.colorbar()
plt.tight_layout()
plt.title('music')
plt.show()
plt.close()
print('The mean correlation during music listening is', np.mean(np.corrcoef(rss_array_music)))

plt.figure(figsize=(18,8))
plt.imshow(np.corrcoef(rss_array_rest), aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.title('rest')
plt.show()
plt.close()
print('The mean correlation in resting state is', np.mean(np.corrcoef(rss_array_rest)))


