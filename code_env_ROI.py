#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:25:42 2022

@author: jeremy
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 22 02:08:26 2022

@author: matte
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

import h5py as h5py
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

#CREATING THE LIST OF SUBJECT TO ANALYSE. 

path='/home/jeremy/anaconda3/matteo/'
#In a first moment we create the list of all the name of the data set we have
arr_mu = os.listdir(path+'seeg_fif_data/music')
arr_rest = os.listdir(path+'seeg_fif_data/speech')
arr_speech = os.listdir(path+'seeg_fif_data/rest')

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
    with h5py.File(pjoin(path+'seeg_data_h5py/h5_electrodes/', subject + '_electrodes.hdf5'), 'r') as f:
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

#loading the bad times indices. 
set_music=sav.load_obj(path+"set_music_bad_times_100hz")
set_speech=sav.load_obj(path+"set_speech_bad_times_100hz")
set_rest=sav.load_obj(path+"set_rest_bad_times_100hz")

print("the bad point are about in music", len(set_music))
print("the bad point are about in speech", len(set_speech))
print("the bad point are about in rest", len(set_rest))

t=30000

sum_act_speech=[]
sum_act_music=[]
sum_act_rest=[]

t=20000

eROI=fc.go_edge_names(list(allr))

    
for isub, subject in enumerate(subject_list):
## Load the data from the HDF fil
    print(subject, isub)
    
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

    with h5py.File(pjoin(path+'seeg_data_h5py/h5_electrodes/', subject + '_electrodes.hdf5'), 'r') as f:
        print(f.keys())
        print('chnames', f['chnames'].shape)
    
        chnames = f['chnames'][...].astype('U')
        
        at=f['atlasses']['Brainnetome'][...]
        listR=[a[:13] for a in at]
    
    with h5py.File(pjoin(path+'seeg_data_h5py/h5_misc/', subject + '_misc.hdf5'), 'r') as f:
        print(f.keys())
        print('outlier_chans', f['outlier_chans']['strict_bads_names'])

        bad_chans = f['outlier_chans']['strict_bads_names'][...].astype('U')
        mu_bad_epo = f['outlier_epochs']['music']['strict_bads_epochs'][...]
        sp_bad_epo = f['outlier_epochs']['speech']['strict_bads_epochs'][...]
    
    #Here we load eventually the envelope of the stimula
    """with h5py.File(pjoin('speech_stimulus.hdf5'), 'r') as f:
        print(f.keys())
        speech_stimulus=f['speech']['matlab']['speech_matlab_env'][...]
        #plt.plot(speech_stimulus)
        print('the length of the speech stimulus is', len(speech_stimulus))
    
    with h5py.File(pjoin('music_stimulus.hdf5'), 'r') as f:
        print(f.keys())
        music_stimulus=f['music']['matlab']['music_matlab_env'][...]

        plt.plot(music_stimulus)
        print('the length of the music stimulus is', len(speech_stimulus))

        #plt.plot(music_stimulus)
        print('the length of the music stimulus is', len(speech_stimulus))"""



    #Cleaning from bad channels
    ch_i = [i for i, ch in enumerate(chnames) if ch in bad_chans]
    clean_chnames = [ch for i, ch in enumerate(chnames) if ch not in bad_chans]
    
    clean_music = np.delete(data_m, ch_i, axis=0)
    clean_speech = np.delete(data_s, ch_i, axis=0)
    clean_rest = np.delete(data_r, ch_i, axis=0)

    #selecting only the channels we want (H or without H or all)
    
    ch_H_i= [i for i, ch in enumerate(clean_chnames) if ch not in ch_H]
    ch_wH_i= [i for i, ch in enumerate(clean_chnames) if ch in ch_H]
    
    #the list of channels that we are studying
    #final_channels=[ch for i, ch in enumerate(clean_chnames) if i not in ch_H_i]
    #print(final_channels)
    
    #data of only H channels
    clean_music_H = np.delete(clean_music, ch_H_i, axis=0)
    clean_speech_H = np.delete(clean_speech, ch_H_i, axis=0)
    clean_rest_H = np.delete(clean_rest, ch_H_i, axis=0)
    
    #data without H
    clean_music_without_H = np.delete(clean_music, ch_wH_i, axis=0)
    clean_speech_without_H = np.delete(clean_speech, ch_wH_i, axis=0)
    clean_rest_without_H = np.delete(clean_rest, ch_wH_i, axis=0)
    
    #computation of the standard deviation
    std_speech=np.std(clean_speech_without_H)
    std_music=np.std(clean_music_without_H)
    std_rest=np.std(clean_rest_without_H)
    
    #print("The number of activity values that are above a threshold of  " +str(N)+ " standard deviation are " + str(len(np.where(clean_speech>N*std_speech+np.mean(clean_speech)))+len(np.where(clean_speech<-N*std_speech+np.mean(clean_speech)))) + "in percentage" + str(len(np.where(clean_speech>N*std_speech+np.mean(clean_speech)))+len(np.where(clean_speech<-N*std_speech+np.mean(clean_speech)))/len(clean_speech)*len(clean_speech[0,:])))
    #print("The number of activity values that are above a threshold of  " +str(N)+ " standard deviation are " + str(len(np.where(clean_music>N*std_music+np.mean(clean_music)))+len(np.where(clean_music<-N*std_music+np.mean(clean_music))))+ "in percentage" + str(len(np.where(clean_music>N*std_music+np.mean(clean_music)))+len(np.where(clean_music<-N*std_music+np.mean(clean_music)))/len(clean_music)*len(clean_music[0,:])))
    #print("The number of activity values that are above a threshold of  " +str(N)+ " standard deviation are " + str(len(np.where(clean_rest>N*std_rest+np.mean(clean_rest)))+len(np.where(clean_rest<-N*std_rest+np.mean(clean_rest))))+ "in percentage" + str(len(np.where(clean_rest>N*std_rest+np.mean(clean_rest)))+len(np.where(clean_rest<-N*std_rest+np.mean(clean_rest)))/len(clean_rest)*len(clean_rest[0,:])))
 
    zdata_speech_art=stats.zscore(clean_speech, axis=1)
    zdata_music_art=stats.zscore(clean_music, axis=1)
    zdata_rest_art=stats.zscore(clean_rest, axis=1)
    
    #CLEANING PROCESS
    
    N=7
    """
    std_speech=np.std(zdata_speech_art)
    std_music=np.std(zdata_music_art)
    std_rest=np.std(zdata_rest_art)
    
    
    
    if isub==1 or isub==10:
        
        art_list_speech=[i for sub in zdata_speech_art for i in sub]
    
        art_list_music=[i for sub in zdata_music_art for i in sub]
    
        art_list_rest=[i for sub in zdata_rest_art for i in sub]
    
        plt.hist(art_list_speech, 100)
        plt.axvline(std_speech*N, label='std*'+str(N))
        plt.axvline(-std_speech*N, label='std*'+str(N))
        plt.title('zscore activities speech distribution')
        plt.legend()
        plt.show()
        plt.close()
        plt.hist(art_list_music, 100)
        plt.axvline(std_music*N, label='std*'+str(N))
        plt.axvline(-std_music*N, label='std*'+str(N))
        plt.title('zscore activities music distribution')
        plt.legend()
        plt.show()
        plt.close()
        plt.hist(art_list_rest, 100)
        plt.axvline(std_rest*N, label='std*'+str(std_rest))
        plt.axvline(-std_rest*N, label='std*'+str(std_rest))
        plt.title('zscore activities rest distribution')
        plt.legend()
        plt.show()
        plt.close()
    
    print('In speech the number of values ' + str(N) + ' std away from the mean is ' + str(len(np.where(zdata_speech_art > N*std_speech)[0])/(len(zdata_speech_art)*len(zdata_speech_art[0,:]))))
    print('In music the number of values ' + str(N) + ' std away from the mean is ' + str(len(np.where(zdata_speech_art > N*std_speech)[0])/(len(zdata_speech_art)*len(zdata_speech_art[0,:]))))
    print('In rest the number of values ' + str(N) + 'std away from the mean is ' + str(len(np.where(zdata_speech_art > N*std_speech)[0])/(len(zdata_speech_art)*len(zdata_speech_art[0,:]))))
    
    zdata_music=fc.clean(zdata_music_art, N=6)
    zdata_speech=fc.clean(zdata_speech_art, N=6)
    zdata_rest=fc.clean(zdata_rest_art, N=6)
    
    
    sum_act_speech.append(np.sqrt(np.sum(zdata_speech**2, axis=0)))
    sum_act_music.append(np.sqrt(np.sum(zdata_music**2, axis=0)))
    sum_act_rest.append(np.sqrt(np.sum(zdata_rest**2, axis=0)))

    clean_sp=np.delete(clean_speech, list(set_speech), axis=1)
    clean_mu=np.delete(clean_music, list(set_music), axis=1)
    clean_re=np.delete(clean_rest, list(set_rest), axis=1)
    
    clean_sp=np.delete(clean_speech, list(set_speech),axis=1)
    clean_mu=np.delete(clean_music, list(set_music),axis=1)
    clean_re=np.delete(clean_rest, list(set_rest),axis=1)"""
    
    clean_sp=np.delete(clean_speech, list(set_speech),axis=1)
    clean_mu=np.delete(clean_music, list(set_music),axis=1)
    clean_re=np.delete(clean_rest, list(set_rest),axis=1)
    
    zdata_speech=stats.zscore(clean_sp, axis=1)
    zdata_music=stats.zscore(clean_mu, axis=1)
    zdata_rest=stats.zscore(clean_re, axis=1)
    
    #zdata_speech_stimulus=stats.zscore(speech_stimulus)
    #zdata_music_stimulus=stats.zscore(music_stimulus)
    
    #zdata_speech_purified=zdata_speech[:,:len(zdata_speech_stimulus)]-zdata_speech_stimulus[:len(zdata_speech)]
    #zdata_music_purified=zdata_music[:,:len(zdata_music_stimulus)]-zdata_music_stimulus[:len(zdata_music)]
   
    #SPEECH
    t_tot=len(zdata_speech[1,:])
    num=int(t_tot/t)
    t_list=[]
    
    for i in range(num+1):
        t_list.append(t*i)
        
    if t_list[-1]!=t_tot:
        t_list.append(t_tot)
        
    print(t_list)
        #SPEECH
    
    Enames=fc.go_edge_names(chnames)
    Eregions=fc.go_edge_names(listR)
    
    for j in range(len(t_list)-1):
        
        x=zdata_speech[:,t_list[j]:t_list[j+1]]
        #x=np.where(abs(x)>5, abs(x)*5/x, x)
        
        print('Done interval', j)
        
        #edge_speech=go_edge_list(x)
        
        x=x.T
        edge_speech=fc.go_edge(x)
        """
        #PLOTTING THE EDGES
        plt.figure(figsize=(12,8))
        plt.imshow(edge_speech.T[:,:10000], aspect='auto', vmin=-1, vmax=1)
        plt.xticks(np.concatenate((np.arange(0,1000,100), np.arange(1500,10000,500))),np.concatenate((np.arange(10), np.arange(15,100,5))))
        plt.xlabel("sec + " + str(t_list[j]/100) + "sec")
        plt.ylabel("edges")
        plt.yticks(np.arange(0, len(Enames), step=100), Enames[::100])
        plt.colorbar()
        plt.tight_layout()
        plt.title("speech, subject: "+ subject)
        plt.show()
        plt.close()"""
        
        for i in eROI:
            indices=[j for j, edge in enumerate(Eregions) if i in edge]
            if subject in rss_speech and i in rss_speech[subject]:
                rss_speech[subject][i]=np.concatenate((rss_speech[subject][i],np.sqrt(np.sum(edge_speech[indices]**2, axis=1))))
            elif subject in rss_speech:
                rss_speech[subject][i]=np.sqrt(np.sum(edge_speech[indices]**2, axis=1))
            else:
                rss_speech[subject]={}
                rss_speech[subject][i]=np.sqrt(np.sum(edge_speech[indices]**2, axis=1))
        
        """
        #PLOTTING THE RSS
        plt.figure(figsize=(14,6))
        plt.plot(rss_speech[subject][t_list[j]:t_list[j]+10000])
        plt.xticks(np.concatenate((np.arange(0,1000,100), np.arange(1500,10000,500))),np.concatenate((np.arange(10), np.arange(15,100,5))))
        plt.xlabel("sec + " + str(t_list[j]/100) + "sec")
        plt.title('rss_speech subject: ' + subject)
        plt.show()
        plt.close()
        #PLOTTING THE DISTRIBUTION OF THE RSS
        plt.hist(stats.zscore(rss_speech[subject]), 300)
        plt.title("zscore of the rss distribution, speech, subject: " + subject)
        plt.show()
        plt.close()  """ 
        
        if subject in rss_speech:
            rss_speech[subject]=np.concatenate((rss_speech[subject],np.sqrt(np.sum(edge_speech**2, axis=1))))
        else:
            rss_speech[subject]=np.sqrt(np.sum(edge_speech**2, axis=1))
            

    #MUSIC
    
    t_tot=len(zdata_music[1,:])
    num=int(t_tot/t)
    t_list=[]
    for i in range(num+1):
        t_list.append(t*i)
         
    if t_list[-1]!=t_tot:
        t_list.append(t_tot)
    
    print(t_list)
    for j in range(len(t_list)-1):
        x=zdata_music[:,t_list[j]:t_list[j+1]].T
        #x=np.where(abs(x)>5, abs(x)*5/x, x)
    
        edge_music=fc.go_edge(x)
        """
        #PLOTTING THE EDGES
        plt.figure(figsize=(12,8))
        plt.imshow(edge_music.T[:,:10000], aspect='auto', vmin=-1, vmax=1)
        plt.xticks(np.concatenate((np.arange(0,1000,100), np.arange(1500,10000,500))),np.concatenate((np.arange(10), np.arange(15,100,5))))
        plt.xlabel("sec + " + str(t_list[j]/100) + "sec")
        plt.ylabel("edges")
        plt.title("music, subject: "+ subject)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        plt.close()"""
        
        print('edge music shape', edge_music.shape)
        for i in eROI:
            indices=[j for j, edge in enumerate(Eregions) if i in edge]
            if subject in rss_music and i in rss_music[subject]:
                rss_music[subject][i]=np.concatenate((rss_music[subject][i],np.sqrt(np.sum(edge_music[indices]**2, axis=1))))
            elif subject in rss_music:
                rss_music[subject][i]=np.sqrt(np.sum(edge_music[indices]**2, axis=1))
            else:
                rss_music[subject]={}
                rss_music[subject][i]=np.sqrt(np.sum(edge_music[indices]**2, axis=1))
        """
        #PLOTTING THE RSS
        plt.figure(figsize=(14,6))
        plt.plot(rss_music[subject][t_list[j]:t_list[j]+10000])
        plt.xticks(np.concatenate((np.arange(0,1000,100), np.arange(1500,10000,500))),np.concatenate((np.arange(10), np.arange(15,100,5))))
        plt.xlabel("sec + " + str(t_list[j]/100) + "sec")
        plt.title('rss_speech subject: ' + subject)
        plt.show()
        plt.close()
        #PLOTTING THE DISTRIBUTION OF THE RSS
        plt.hist(stats.zscore(rss_music[subject]), 300)
        plt.title("zscore of the rss distribution, speech, subject: "+ subject)
        plt.show()
        plt.close()"""
        
    #REST
    
    t_tot=len(zdata_rest[1,:])
    num=int(t_tot/t)
    t_list=[]
    for i in range(num+1):
        t_list.append(t*i)
        
    if t_list[-1]!=t_tot:
        t_list.append(t_tot)
    
    for j in range(len(t_list)-1):
        x=zdata_rest[:,t_list[j]:t_list[j+1]].T
        #x=np.where(abs(x)>5, abs(x)*5/x, x)
    
        edge_rest=fc.go_edge(x)
        """
        #PLOTTING THE EDGES
        plt.figure(figsize=(12,8))
        plt.imshow(edge_rest.T[:,:10000], aspect='auto', vmin=-1, vmax=1)
        plt.xticks(np.concatenate((np.arange(0,1000,100), np.arange(1500,10000,500))),np.concatenate((np.arange(10), np.arange(15,100,5))))
        plt.xlabel("sec + " + str(t_list[j]/100) + "sec")
        plt.ylabel("edges")
        plt.title("rest, subject: "+ subject)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        plt.close()"""
        
        for i in eROI:
            indices=[j for j, edge in enumerate(Eregions) if i in edge]
            if subject in rss_rest and i in rss_rest[subject]:
                rss_rest[subject][i]=np.concatenate((rss_rest[subject][i],np.sqrt(np.sum(edge_rest[indices]**2, axis=1))))
            elif subject in rss_speech:
                rss_rest[subject][i]=np.sqrt(np.sum(edge_rest[indices]**2, axis=1))
            else:
                rss_rest[subject]={}
                rss_rest[subject][i]=np.sqrt(np.sum(edge_rest[indices]**2, axis=1))
        """    
        #PLOTTING THE RSS
        plt.figure(figsize=(14,6))
        plt.plot(rss_music[subject][t_list[j]:t_list[j]+10000])
        plt.xticks(np.concatenate((np.arange(0,1000,100), np.arange(1500,10000,500))),np.concatenate((np.arange(10), np.arange(15,100,5))))
        plt.xlabel("sec + " + str(t_list[j]/100) + "sec")
        plt.title('rss_speech subject: ' + subject)
        plt.show()
        plt.close()
        #PLOTTING THE DISTRIBUTION OF THE RSS
        plt.hist(stats.zscore(rss_music[subject]), 300)
        plt.title("zscore of the rss distribution, speech, subject: " + subject)
        plt.show()
        plt.close()"""
        
    print('')
    print('')
    
print('end data, start plotting')

#preplotting and plotting

#SPEECH
corr_speech=[]
rss_list_speech=[]
z_rss_list_speech=[]
for i in eROI:
    for subject in subject_list:
        rss_list_speech.append(rss_speech[subject][i].T)
        
    rss_array_speech=np.array(rss_list_speech)
    
    """
    plt.figure(figsize=(12,8))
    plt.imshow(rss_array_speech, aspect='auto')
    plt.colorbar()
    plt.tight_layout()
    plt.title('speech')
    plt.show()
    plt.close()"""
    
    corr_speech.append(np.array([a in np.corrcoef(rss_array_speech)]))
    
#MUSIC
corr_music=[]
rss_list_music=[]

for i in eROI:
    for subject in subject_list:
        rss_list_music.append(rss_music[subject][i].T)
        
    rss_array_music=np.array(rss_list_music)
    
    """
    plt.figure(figsize=(12,8))
    plt.imshow(rss_array_speech, aspect='auto')
    plt.colorbar()
    plt.tight_layout()
    plt.title('speech')
    plt.show()
    plt.close()"""
    
    corr_music.append(np.array([a in np.corrcoef(rss_array_music)]))
    
#REST
corr_rest=[]
rss_list_rest=[]
z_rss_list_rest=[]
for i in eROI:
    for subject in subject_list:
        rss_list_rest.append(rss_rest[subject][i].T)
        
    rss_array_rest=np.array(rss_list_rest)
    
    """
    plt.figure(figsize=(12,8))
    plt.imshow(rss_array_speech, aspect='auto')
    plt.colorbar()
    plt.tight_layout()
    plt.title('speech')
    plt.show()
    plt.close()"""
    
    corr_rest.append(np.array([a in np.corrcoef(rss_array_rest)]))
    

"""
sav.save_obj(np.corrcoef(rss_array_speech) , path + 'corr_matrix_allsubs_speech') 
plt.figure(figsize=(12,8))
plt.imshow(np.corrcoef(rss_array_speech), aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.title('corr matrix, speech')
plt.xlabel('subjects')
plt.ylabel('subjects')
plt.show()
plt.close()

print('The mean correlation during speech listening is', np.mean(np.corrcoef(rss_array_speech)[np.triu_indices(19, k = 1)]))
#('The result of the correlation is this number of standard daviation away from the mean of the distribution',np.mean(np.corrcoef(rss_array_speech)[np.triu_indices(19, k = 1)])/np.std(list_mean_corr_speech))
print('The number of value of correlation > 0.01is', len(np.argwhere(np.corrcoef(rss_array_speech)[np.triu_indices(19, k = 1)]>0.01)))
print('The number of values of correlation > 0.02 is', len(np.argwhere(np.corrcoef(rss_array_speech)[np.triu_indices(19, k = 1)]>0.02)))

sav.save_obj(np.corrcoef(rss_array_music) ,path + 'corr_matrix_allsubs_music')     
plt.figure(figsize=(12,8))
plt.imshow(np.corrcoef(rss_array_music), aspect='auto',)
plt.colorbar()
plt.tight_layout()
plt.title('corr matrix, music')
plt.xlabel('subjects')
plt.ylabel('subjects')
plt.show()
plt.close()
print('The mean correlation during music listening is', np.mean(np.corrcoef(rss_array_music)[np.triu_indices(19, k = 1)]))

sav.save_obj(np.corrcoef(rss_array_rest) , path +'corr_matrix_allsubs_rest') 
plt.figure(figsize=(12,8))
plt.imshow(np.corrcoef(rss_array_rest), aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.title('corr matrix, rest')
plt.xlabel('subjects')
plt.ylabel('subjects')
plt.show()
plt.close()
print('The mean correlation in resting state is', np.mean(np.corrcoef(rss_array_rest)[np.triu_indices(19, k = 1)]))
"""