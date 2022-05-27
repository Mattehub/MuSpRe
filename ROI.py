#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:06:51 2022

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

allr=set()

with h5py.File(pjoin(path+'seeg_data_h5py/h5_electrodes/', subject_list[0] + '_electrodes.hdf5'), 'r') as f:
        
    at=f['atlasses']['Brainnetome'][...]
    listR=[a[:13] for a in at]
    listROI=[i for i in listR if listR.count(i)>2]
    allr=set(listROI)
    
            
for subject in subject_list:
    
    with h5py.File(pjoin(path+'seeg_data_h5py/h5_electrodes/', subject + '_electrodes.hdf5'), 'r') as f:
        at=f['atlasses']['Brainnetome'][...]
        listR=[a[:13] for a in at]
        listROI=[i for i in listR if listR.count(i)>2]
        allr.intersection(set(listROI))
    