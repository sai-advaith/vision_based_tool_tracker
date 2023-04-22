"""
This file is used to clean up the measurement data being received

The cleaning up is done using the IIR Filter
"""
# Python Libraries
import numpy as np
from scipy.signal import iirfilter, lfilter, freqz
import matplotlib.pyplot as plt
import pandas as pd
from tf.transformations import euler_from_quaternion

def iir_lowpass(cutoff, fs, order=5):
    """
    Low pass of the IIR Filter
    """
    # Nyquist frequency (nyq) is half the frequency rate
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = iirfilter(order, normal_cutoff, btype='low', analog=False)
    
    return b, a

def iir_lowpass_filter(data, cutoff, fs, order=5):
    """
    Implementing the lowpass filter
    """
    b, a = iir_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)

    return y

def data_filter(data):
    """
    Implementing IIR filter on the data
    """
    # Filter requirements.
    order = 1
    fs = 30.0       # sample rate, Hz
    cutoff = 3.667  # desired cutoff frequency of the filter, Hz

    # Filter the data
    y = iir_lowpass_filter(data, cutoff, fs, order)

    # Returning the last element of the list
    return y[-1]