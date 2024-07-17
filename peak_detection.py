#Author: Haruka Yamamoto
#Date: 29/4/2024
#File Description:
# This python script detects peaks from the ppg signal
# Uses HeartPy library for peak detection using a rolling mean


import heartpy as hp
from heartpy.datautils import rolling_mean
import numpy as np

class PeakDetection():

    def __int__(self):
        self.data = []
        self.wd = []

    def peaks(self, data):
        # gets data from the shared list of ppg signal values from pyqt_grapher 
        # updates the working directory which can be used to get the list of detected peak values
        self.data = data
        self.roll_mean = rolling_mean(self.data, windowsize = 0.75, sample_rate = 30)
        self.wd = hp.peakdetection.detect_peaks(np.array(self.data), self.roll_mean, ma_perc = 20, sample_rate = 30)

    def get_wd(self):
        return self.wd