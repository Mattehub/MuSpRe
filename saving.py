<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 02:12:38 2022

@author: matte
"""

import pickle

def saving(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f)
        
def loading(name):
    with open(name,"rb") as f:
        loaded_obj=pickle.load(f)
        
    return loaded_obj
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 19:24:29 2022

@author: jeremy
"""
import h5py
import pickle
"""

with h5py.File(pjoin('seeg_data_h_env_down_h5py/', subject + '_down_seeg_preproc.hdf5'), 'r') as f:
    print(f.keys())
    print('music', f['music'].shape)

    #data_music[subject]=f['music'][...]
    data_m=f['music'][...]
    data_m=np.array(data_m)
    
#SPEECH
with h5py.File(pjoin('seeg_data_h_env_down_h5py/', subject + '_down_seeg_preproc.hdf5'), 'r') as f:
    print(f.keys())
    print('speech', f['speech'].shape)
    print('speech', f['speech'].shape)
    #data_speech[subject]= f['speech'][...]
    data_s=f['speech'][...]
    data_s=np.array(data_s)    
        

#REST
with h5py.File(pjoin('seeg_data_h_env_down_h5py/', subject + '_down_seeg_preproc.hdf5'), 'r') as f:
    print(f.keys())
    print('rest', f['rest'].shape)
    print('rest', f['rest'].shape)
    #data_rest[subject]=f['rest'][...]
    data_r=f['rest'][...]
    data_r=np.array(data_r)
    
dic={'rest':data_r, 'music': data_m, 'speech': data_s}       
 
    """ 

def save_obj(obj, name ):
    with open(name+ '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('/Users/giovanni/Desktop/Neuro/Pierpaolo-MEG/ArticleResults_I_py2/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
#save_obj(dic, '1_40_fufo')  
>>>>>>> a2420c9950226a425412fadd94aaef0c87582e7b
