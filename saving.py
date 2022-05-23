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