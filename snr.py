import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def snr_p(z,label):
    sec = 5
    secr = sec*100
    picking = np.where(label==1)[0]
    
    if len(picking)==0 or picking[0]==0 or not picking.any():
        return None
    
    # signal: picking 與後 500 samples
    picking = picking[0]
    signal = z[picking:picking+secr]
    signal = np.percentile(signal, 95)
    
    # noise: picking 前 500 samples
    if picking<secr:
        noise = z[:picking].tolist()
        while len(noise)<secr:
            noise.extend(noise)
        noise = noise[:secr]
    else:
        noise = z[picking-secr:picking]

    noise = np.percentile(noise, 95)
    
    ratio = signal/noise
    
    return ratio
