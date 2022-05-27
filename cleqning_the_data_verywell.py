#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:32:52 2022

@author: jeremy
"""
import matplotlib.pyplot as plt
# Plot the signal read from wav file

plt.subplot(211)

plt.title('Spectrogram of a wav file with piano music')

clean_music[36,4]=100000

plt.plot(stats.zscore(clean_music, axis=1)[36,:])

plt.xlabel('Sample')

plt.ylabel('Amplitude')

 

plt.subplot(212)

plt.specgram(stats.zscore(clean_music, axis=1)[36,:])

plt.xlabel('Time')

plt.ylabel('Frequency')

 

plt.show()

