#Author: Haruka Yamamoto
#Date: 10/4/2024
#File Description:
# This python script detects faces using opencv2 and will track important facial regions
# such as the forehead and cheeks. This will be used to apply PPG to extract heart rate.
# The green colour space is chosen for this as it contains the strongest 
# PPG signals.

from collections import deque
import numpy as np


class LiveFilter:
    # Parent class for live filters.
    def process(self, x):
        # do not process NaNs
        if np.isnan(x):
            return x

        return self._process(x)

    def __call__(self, x):
        return self.process(x)

    def _process(self, x):
        raise NotImplementedError("Derived class must implement -> _process")

class LiveSosFilter(LiveFilter):
    # live implementation of the scipy second order filter.
    # this makes it so each value can be fed in real time and not grabbed from a csv file
    def __init__(self, sos):
        # init the class with the sos generated by the scipy sos func
        self.sos = sos

        self.n_sections = sos.shape[0]
        self.state = np.zeros((self.n_sections, 2))

    def _process(self, x):

        for s in range(self.n_sections):  # apply filter sections in sequence
            b0, b1, b2, a0, a1, a2 = self.sos[s, :] # gettting the coefficients for filter

            # compute difference equations of transposed direct form II
            y = b0*x + self.state[s, 0]
            self.state[s, 0] = b1*x - a1*y + self.state[s, 1]
            self.state[s, 1] = b2*x - a2*y
            x = y  # set biquad output as input of next filter section.
        # return the filtered y value
        return y