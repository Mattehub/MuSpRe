
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
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

