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

warnings.simplefilter('ignore')

#CREATING THE LIST OF SUBJECTS

sound_list=['rest','music','speech']
arr_mu = os.listdir('seeg_fif_data/music')
arr_rest = os.listdir('seeg_fif_data/speech')
arr_speech = os.listdir('seeg_fif_data/rest')

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
    with h5py.File(pjoin('seeg_data_h5py/h5_electrodes/', subject + '_electrodes.hdf5'), 'r') as f:
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

for isub, subject in enumerate(subject_list):
## Load the data from the HDF fil
    print(subject, isub)
    
    fc_dict[subject]={}
    
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
        #data_speech[subject]= f['speech'][...]
        data_s=f['speech'][...]

    #REST
    with h5py.File(pjoin('seeg_data_h_env_down_h5py/', subject + '_down_seeg_preproc.hdf5'), 'r') as f:
        print(f.keys())
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
    
    zdata_speech=stats.zscore(clean_speech, axis=1)
    zdata_music=stats.zscore(clean_music, axis=1)
    zdata_rest=stats.zscore(clean_rest, axis=1)
  
    speech_data_av=zdata_speech.copy()
    music_data_av=zdata_music.copy()
    rest_data_av=zdata_rest.copy()
    
    thres=2.8
    print(thres)
    
    avalanches_rest, _ =av.go_avalanches_general(zdata_rest.T, thre=thres, direc=0, binsize=2, event_rate=0, sampling=100, threshold=[], method='simple')
    y=np.arange(0,len(clean_chnames),2)
    plt.figure(figsize=(15,13))
    plt.imshow(avalanches_rest['Zbin'].T[:,:1000], aspect='auto', interpolation='none')
    plt.yticks(y, clean_chnames[::2])
    plt.title('rest, after binarization, threshold='+str(thres))
    plt.colorbar()
    plt.show()
    plt.close()
    
    size_rest.append(avalanches_rest['siz'])
    x,y,_=plt.hist(avalanches_rest['IAI'], 20)
    plt.show()
    plt.close()
    plt.loglog(x,y[:-1])
    plt.show()
    plt.close()
    Ebin=fc.go_edge(avalanches_rest['Zbin'])
    rss_rest.append(np.sum(Ebin.T, axis=0))
    fc_rest=np.mean(Ebin, axis=0)
    n_regions=len(avalanches_rest['Zbin'][0,:])
    FC_rest=np.zeros((n_regions,n_regions))
    FC_rest[np.triu_indices(n_regions,1)]=fc_rest
    FC_rest= FC_rest + FC_rest.T + np.identity(n_regions)
    fc_dict[subject]['rest']=FC_rest
    
    thres=2.8
    avalanches_speech=av.go_avalanches(zdata_speech.T, thre=thres, direc=0, binsize=2)
    
    size_speech.append(avalanches_speech['siz'])

    y=np.arange(0,len(clean_chnames),2)
    plt.figure(figsize=(15,13))
    plt.imshow(avalanches_speech['Zbin'].T[:,:1000], aspect='auto', interpolation='none')
    plt.title('speech, after binarization, threshold='+str(thres))
    plt.yticks(y, clean_chnames[::2])
    plt.colorbar()
    plt.show()
    plt.close()
    
    Ebin=fc.go_edge(avalanches_speech['Zbin'])
    rss_speech.append(np.sum(Ebin.T, axis=0))
    fc_speech=np.mean(Ebin, axis=0)
    n_regions=len(avalanches_speech['Zbin'][0,:])
    FC_speech=np.zeros((n_regions,n_regions))
    FC_speech[np.triu_indices(n_regions,1)]=fc_speech
    FC_speech= FC_speech + FC_speech.T + np.identity(n_regions)
    fc_dict[subject]['speech']=FC_speech
    
    avalanches_music =av.go_avalanches(zdata_music.T, thre=thres, direc=0, binsize=2)
    size_music.append(avalanches_music['siz'])
    
    y=np.arange(0,len(clean_chnames),2)
    plt.figure(figsize=(15,13))
    plt.imshow(avalanches_music['Zbin'].T[:,:1000], aspect='auto', interpolation='none')
    plt.title('music, after binarization, threshold='+str(thres))
    plt.yticks(y, clean_chnames[::2])
    plt.colorbar()
    plt.show()
    plt.close()
    
    
    Ebin=fc.go_edge(avalanches_music['Zbin'])
    rss_music.append(np.sum(Ebin.T, axis=0))
    fc_music=np.mean(Ebin, axis=0)
    n_regions=len(avalanches_music['Zbin'][0,:])
    FC_music=np.zeros((n_regions,n_regions))
    FC_music[np.triu_indices(n_regions,1)]=fc_music
    FC_music= FC_music + FC_music.T + np.identity(n_regions)
    fc_dict[subject]['music']=FC_music
    
    mean_iai_speech=[]
    mean_iai_music=[]
    mean_iai_rest=[]
    
    for min_siz in min_sizes:

        mins_avalanches_speech=av.min_siz_filt(avalanches_speech, min_siz)
        
        """plt.figure(figsize=(12,8))
        plt.imshow(mins_avalanches_speech['Zbin'].T[:,:1000], aspect='auto', interpolation='none')
        plt.yticks(y, clean_chnames[::2])
        plt.title('speech, min_size='+str(min_siz))
        plt.colorbar()
        plt.show()
        plt.close()"""
    
        mins_avalanches_music=av.min_siz_filt(avalanches_music, min_siz)
        
        """plt.figure(figsize=(12,8))
        plt.imshow(mins_avalanches_music['Zbin'].T[:,:1000], aspect='auto', interpolation='none')
        plt.yticks(y, clean_chnames[::2])
        plt.title('music, min_size='+str(min_siz))
        plt.colorbar()
        plt.show()
        plt.close()"""
    
        mins_avalanches_rest=av.min_siz_filt(avalanches_rest, min_siz)
        
        """plt.figure(figsize=(12,8))
        plt.imshow(mins_avalanches_rest['Zbin'].T[:,:1000], aspect='auto', interpolation='none')
        plt.yticks(y, clean_chnames[::2])
        plt.title('rest, min_size='+str(min_siz))
        plt.colorbar()
        plt.show()
        plt.close()"""
    
        mean_iai_speech.append(np.mean(mins_avalanches_speech['IAI']))
        mean_iai_music.append(np.mean(mins_avalanches_music['IAI']))
        mean_iai_rest.append(np.mean(mins_avalanches_rest['IAI']))
        
    
    plt.plot(min_sizes, mean_iai_speech, label='speech')
    plt.plot(min_sizes, mean_iai_music, label='music')
    plt.plot(min_sizes, mean_iai_rest, label='rest')
    plt.legend()
    plt.title('IAI')
    plt.show()
    plt.close()    
    
    subject_iai_speech.append(mean_iai_speech)
    subject_iai_music.append(mean_iai_music)
    subject_iai_rest.append(mean_iai_rest)
    
subject_iai_speech=np.array(subject_iai_speech)
subject_iai_music=np.array(subject_iai_music)
subject_iai_rest=np.array(subject_iai_rest)

subject_mean_iai_speech=np.mean(subject_iai_speech, axis=0)
subject_mean_iai_music=np.mean(subject_iai_music, axis=0)
subject_mean_iai_rest=np.mean(subject_iai_rest, axis=0)

plt.plot(min_sizes, subject_mean_iai_speech, label='speech')
plt.plot(min_sizes, subject_mean_iai_music, label='music')
plt.plot(min_sizes, subject_mean_iai_rest, label='rest')
plt.legend()
plt.title('IAI')
plt.show()
plt.close()   


plt.figure(figsize=(28,200))
for i, sub in enumerate(subject_list):
    for j, sound in enumerate(sound_list):
        x=np.arange(1, len(final_channels_all[sub])+1,3)
        y=np.arange(1, len(final_channels_all[sub])+1,3)
        plt.subplot(19, 3, 3*i + j+1)
        plt.imshow(fc_dict[sub][sound], interpolation='none', vmin=0, vmax=3*np.std(fc_dict[sub]['rest']))
        plt.xticks(x, final_channels_all[sub][::3], rotation='vertical')
        plt.yticks(y, final_channels_all[sub][::3])
        if j==0:
            plt.title(sub+'  '+sound)
        else:
            plt.title(sound)
        if j==2:
            plt.colorbar()
plt.show()
plt.close()

b=0.25
corr_matrix_speech=np.corrcoef(np.array(rss_speech))
plt.figure(figsize=(12,10))
plt.imshow(corr_matrix_speech, aspect='auto', vmin=-b, vmax=b)
plt.title('speech')
plt.colorbar()
plt.show()
plt.close()

print('the mean correlation between subjects during speech is', np.mean(np.triu(corr_matrix_speech, 1)))

corr_matrix_music=np.corrcoef(np.array(rss_music))
plt.figure(figsize=(12,10))
plt.imshow(corr_matrix_music, aspect='auto', vmin=-b, vmax=b)
plt.title('music')
plt.colorbar()
plt.show()
plt.close()

print('the mean correlation between subjects during music is', np.mean(np.triu(corr_matrix_music, 1)))

corr_matrix_rest=np.corrcoef(np.array(rss_rest))
plt.figure(figsize=(12,10))
plt.imshow(corr_matrix_rest, aspect='auto', vmin=-b, vmax=b)
plt.title('rest')
plt.colorbar()
plt.show()
plt.close()

print('the mean correlation between subjects during rest is', np.mean(np.triu(corr_matrix_rest, 1)))

size_rest_list=[element for sub in size_rest for element in sub]

y,x,_=plt.hist(size_rest_list,40)
plt.title('size distribution')
plt.show()
plt.close()

plt.loglog(x[:-1],y)
plt.show()
plt.close()

size_speech_list=[element for sub in size_speech for element in sub]

y1,x1,_=plt.hist(size_speech_list,40)
plt.title('size distribution')
plt.show()
plt.close()

plt.loglog(x1[:-1],y1)
plt.show()
plt.close()

size_music_list=[element for sub in size_music for element in sub]

y2,x2,_=plt.hist(size_music_list,40)
plt.title('size distribution')
plt.show()
plt.close()

plt.loglog(x2[:-1],y2)
plt.show()
plt.close()

plt.loglog(x[:-1],y, label='rest')
plt.loglog(x1[:-1],y1, label='speech')
plt.loglog(x2[:-1],y2, label='music')
plt.show()
plt.close()




    #plt.hist(avalanches_speech['siz'])
    #plt.show()
    #plt.close()
    
    
    
    
    





"""
    #FINDING THE EVENT - SIMPLE APPROACH 
    
    for i in range(len(speech_data_av)):
        
        
        #threshold=np.percentile(zdata_speech[i,:], 99)
        #print(threshold)
        speech_data_av[i,:][zdata_speech[i,:]<threshold]=0
        speech_data_av[i,:][zdata_speech[i,:]>=threshold]=1
   
        #threshold=np.percentile(zdata_music[i,:], 99)
        
        #print(threshold)
        music_data_av[i,:][zdata_music[i,:]<threshold]=0
        music_data_av[i,:][zdata_music[i,:]>=threshold]=1
    
        #threshold=np.percentile(zdata_rest[i,:], 99)
        #print(threshold)
        rest_data_av[i,:][zdata_rest[i,:]<threshold]=0
        rest_data_av[i,:][zdata_rest[i,:]>=threshold]=1
    
    #BINARISATION
    
    speech_data_av_bin=[]
    music_data_av_bin=[]
    rest_data_av_bin=[]
    
    for i in range(len(speech_data_av)):
        
        speech_list=[]
        music_list=[]
        rest_list=[]
        for j in range(int(len(speech_data_av[i,:])/3)-1): 
            
            if np.sum(speech_data_av[i,j*3:(j+1)*3]) >0:
                speech_list.append(1)
            else:
                speech_list.append(0)
            
            if np.sum(music_data_av[i,j*3:(j+1)*3]) >0:
                music_list.append(1)
            else:
                music_list.append(0)
            
            if np.sum(rest_data_av[i,j*3:(j+1)*3]) >0:
                rest_list.append(1)
            else:
                rest_list.append(0)
        
        speech_data_av_bin.append(speech_list)
        music_data_av_bin.append(music_list)
        rest_data_av_bin.append(rest_list)

    size_av_speech=np.sum(speech_data_av, axis=0)
    size_av_music=np.sum(music_data_av, axis=0)
    size_av_rest=np.sum(rest_data_av, axis=0)
    
    size_av_speech_bin=np.sum(speech_data_av_bin, axis=0)
    size_av_music_bin=np.sum(music_data_av_bin, axis=0)
    size_av_rest_bin=np.sum(rest_data_av_bin, axis=0)
    
    #THE AVERAGES
    print('the mean avalanche size in speech is ', np.mean(size_av_speech_bin))
    mean_vector_speech.append(np.mean(size_av_speech_bin))
    
    print('the mean avalanche size in speech is ', np.mean(size_av_music_bin))
    mean_vector_music.append(np.mean(size_av_music_bin))
    
    print('the mean avalanche size in speech is ', np.mean(size_av_rest_bin))
    mean_vector_rest.append(np.mean(size_av_rest_bin))

    #PLOTTING
    plt.hist(size_av_speech, 100)
    plt.title('size_distribution speech')
    plt.show()
    plt.close()
    
    plt.hist(size_av_music, 100)
    plt.title('size_distribution music')
    plt.show()
    plt.close()
    
    plt.hist(size_av_rest, 100)
    plt.title('size_distribution rest')
    plt.show()
    plt.close()
    
    plt.hist(size_av_speech_bin, 100)
    plt.title('size_distribution speech')
    plt.show()
    plt.close()
    
    plt.hist(size_av_music_bin, 100)
    plt.title('size_distribution music')
    plt.show()
    plt.close()
    
    plt.hist(size_av_rest_bin, 100)
    plt.title('size_distribution rest')
    plt.show()
    plt.close()
    
    
    
    plt.figure(figsize=(12,8))
    plt.imshow(speech_data_av_bin, aspect='auto')
    plt.colorbar()
    plt.title('speech')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,8))
    plt.imshow(music_data_av_bin, aspect='auto')
    plt.colorbar()
    plt.title('music')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,8))
    plt.imshow(music_data_av_bin, aspect='auto')
    plt.colorbar()
    plt.title('music')
    plt.show()
    plt.close()
    
    #FINDING THE EVENTS - AREA METHOD
    Athreshold=5
    speech_data_av2=np.zeros(np.shape(zdata_speech))
    music_data_av2=np.zeros(np.shape(zdata_music))
    rest_data_av2=np.zeros(np.shape(zdata_rest))  
    
    for i in range(len(speech_data_av2)):
        c=0
        #method to find the parameter similar to Viola one.
        '''areas=[]
        for j in range(len(speech_data_av2[i,:])):
            
            if j==0 and zdata_music[i,j]>0:
                a=0
                    
            if zdata_music[i,j]>0 and zdata_music[i,j-1]<0:
                a=j
                c=zdata_music[i,j]
                
            if zdata_music[i,j]>0 and zdata_music[i,j-1]>0:
                c+=zdata_music[i,j]
                
            if zdata_music[i,j]<0 and zdata_music[i,j-1]>0:
                areas.append(c)
                
        Athreshold=np.percentile(areas, 99)'''
                
        c=0
        for j in range(len(speech_data_av2[i,:])):
            
            #speech
            if j==0 and zdata_speech[i,j]>0:
                a=0
                    
            if zdata_speech[i,j]>0 and zdata_speech[i,j-1]<0:
                a=j
                c=zdata_speech[i,j]
                
            if zdata_speech[i,j]>0 and zdata_speech[i,j-1]>0:
                c+=zdata_speech[i,j]
                
            if zdata_speech[i,j]<0 and zdata_speech[i,j-1]>0 and c>Athreshold:
                speech_data_av2[i,np.arange(a,j)]=1
            
            #music
            if j==0 and zdata_music[i,j]>0:
                a=0
                    
            if zdata_music[i,j]>0 and zdata_music[i,j-1]<0:
                a=j
                c=zdata_music[i,j]
                
            if zdata_music[i,j]>0 and zdata_music[i,j-1]>0:
                c+=zdata_music[i,j]
                
            if zdata_music[i,j]<0 and zdata_music[i,j-1]>0 and c>Athreshold:
                music_data_av2[i,np.arange(a,j)]=1
              
            #rest
            if j==0 and zdata_rest[i,j]>0:
                a=0

            if zdata_rest[i,j]>0 and zdata_rest[i,j-1]<0:
                a=j
                c=zdata_rest[i,j]
                
            if zdata_rest[i,j]>0 and zdata_rest[i,j-1]>0:
                c+=zdata_rest[i,j]
                
            if zdata_rest[i,j]<0 and zdata_rest[i,j-1]>0 and c>Athreshold:
                rest_data_av2[i,np.arange(a,j)]=1
                
    #BINARISATION
    speech_data_av_bin2=[]
    music_data_av_bin2=[]
    rest_data_av_bin2=[]
    
    for i in range(len(speech_data_av2)):
        
        speech_list=[]
        music_list=[]
        rest_list=[]
        for j in range(int(len(speech_data_av2[i,:])/10)-1): 
            
            if np.sum(speech_data_av2[i,j*10:(j+1)*10]) >0:
                speech_list.append(1)
            else:
                speech_list.append(0)
            
            if np.sum(music_data_av2[i,j*10:(j+1)*10]) >0:
                music_list.append(1)
            else:
                music_list.append(0)
            
            if np.sum(rest_data_av2[i,j*10:(j+1)*10]) >0:
                rest_list.append(1)
            else:
                rest_list.append(0)
        
        speech_data_av_bin2.append(speech_list)
        music_data_av_bin2.append(music_list)
        rest_data_av_bin2.append(rest_list)
        
    size_av_speech_bin2=np.sum(speech_data_av_bin2, axis=0)
    size_av_music_bin2=np.sum(music_data_av_bin2, axis=0)
    size_av_rest_bin2=np.sum(rest_data_av_bin2, axis=0)
  
    iai_s=[]
    iai_m=[]
    iai_r=[]
    
    mean_iai_s=[]
    mean_iai_m=[]
    mean_iai_r=[]
    
    
    for i in min_sizes:
        
        inter_vec_s=size_av_speech_bin.copy()
        inter_vec_m=size_av_music_bin.copy()
        inter_vec_r=size_av_rest_bin.copy()
    
        inter_vec_s[size_av_speech_bin<i]=0
        inter_vec_m[size_av_music_bin<i]=0
        inter_vec_r[size_av_rest_bin<i]=0
        
        cr=0
        cm=0
        cs=0
        
        for j in range(len(inter_vec_s)):
            
            if inter_vec_s[j]==0:
                cs+=1
                
            if inter_vec_s[j]>0 and cs>0:
                iai_s.append(cs)
                cs=0
            
            if inter_vec_m[j]==0:
                cm+=1
                
            if inter_vec_m[j]>0 and cm>0:
                iai_m.append(cm)
                cm=0
                
            if inter_vec_r[j]==0:
                cr+=1
                
            if inter_vec_r[j]>0 and cr>0:
                iai_r.append(cr)
                cr=0
                
        mean_iai_s.append(np.mean(iai_s))
        mean_iai_m.append(np.mean(iai_m))
        mean_iai_r.append(np.mean(iai_r))
        
    
    plt.plot(min_sizes, mean_iai_s, label='speech')
    plt.plot(min_sizes, mean_iai_m, label='music')
    plt.plot(min_sizes, mean_iai_r, label='rest' )
    plt.xlabel('min_size_avalanches')
    plt.ylabel('mean_IAI')
    plt.legend()
    plt.title('mean IAI')
    plt.show()
    plt.close()
    
    plt.hist(iai_s, 100)
    plt.title('speech')
    plt.show()
    plt.close()
    
    plt.hist(iai_m, 100)
    plt.title('music')
    plt.show()
    plt.close()
    
    plt.hist(iai_r, 100)
    plt.title('rest')
    plt.show()
    plt.close()    
                
                
                
            
                
        
        
    
    
    
   
    #IAI
    
    
    size_av_speech_2=np.sum(speech_data_av2, axis=0)
    size_av_music_2=np.sum(music_data_av2, axis=0)
    size_av_rest_2=np.sum(rest_data_av2, axis=0)
    
    plt.hist(size_av_speech_2, 100)
    plt.title('size_distribution speech')
    plt.show()
    plt.close()
    
    plt.hist(size_av_music_2, 100)
    plt.title('size_distribution music')
    plt.show()
    plt.close()
    
    plt.hist(size_av_rest_2, 100)
    plt.title('size_distribution rest')
    plt.show()
    plt.close()
    
    
    plt.figure(figsize=(12,8))
    plt.imshow(speech_data_av2, aspect='auto')
    plt.colorbar()
    plt.title('speech')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,8))
    plt.imshow(music_data_av2, aspect='auto')
    plt.colorbar()
    plt.title('music')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,8))
    plt.imshow(music_data_av2, aspect='auto')
    plt.colorbar()
    plt.title('music')
    plt.show()
    plt.close()
    
    size_av_speech_bin2=np.sum(speech_data_av_bin2, axis=0)
    size_av_music_bin2=np.sum(music_data_av_bin2, axis=0)
    size_av_rest_bin2=np.sum(rest_data_av_bin2, axis=0)
    
    plt.hist(size_av_speech_bin2, 100)
    plt.title('size_distribution speech')
    plt.show()
    plt.close()
    
    plt.hist(size_av_music_bin2, 100)
    plt.title('size_distribution music')
    plt.show()
    plt.close()
    
    plt.hist(size_av_rest_bin2, 100)
    plt.title('size_distribution rest')
    plt.show()
    plt.close()
    
    
    
    plt.figure(figsize=(12,8))
    plt.imshow(speech_data_av_bin2, aspect='auto')
    plt.colorbar()
    plt.title('speech')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,8))
    plt.imshow(music_data_av_bin2, aspect='auto')
    plt.colorbar()
    plt.title('music')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,8))
    plt.imshow(music_data_av_bin2, aspect='auto')
    plt.colorbar()
    plt.title('music')
    plt.show()
    plt.close()
        
plt.plot([1,2,3,4,5], mean_vector_speech[:-3])  
plt.plot([1,2,3,4,5], mean_vector_music[:-3])
plt.plot([1,2,3,4,5], mean_vector_rest[:-3])  
plt.xlabel('min_size_avalanches')
plt.ylabel('mean_IAI')
plt.show()
plt.close()
    """
    
    
    
    
    
    