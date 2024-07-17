#Author: Haruka Yamamoto
#Date: 23/3/2024
#File Description:
# This python script plots the ppg signal from the face detection program
# Also used to stored data in a list for peak detection to get heart rate

from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore

class Graph(QMainWindow):

    def __init__(self, fps):
        
        super().__init__()

        self.record_time = 20 # set the time the data is collected for        
        self.fps = fps # this is multiplied by the number above to get number of data points below
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        # The stored 600 data points is a balance between effective data and accuracy
        # The more data points means less multiplication of peaks to get bpm 
        # (600 points requires multiplication by 3 to get 30Hz for 60 secs to get bpm)
        # But more points means more variation in data is added to the bpm 

        self.x = list(range(self.fps*self.record_time))  # 600 time points equals 20 seconds of data at 30Hz sample rate
        self.y = list(np.zeros(self.fps*self.record_time)) # this list stores data for graphing and peak detection

        self.graphWidget.setBackground('w')

        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line =  self.graphWidget.plot(self.x, self.y, pen=pen)


    def update(self, data):
        # This method keeps the same number of data points and adds new 
        # data and then removes old data. First in last out stack

        self.y.append(data) # adding data to y values mapped to x values for plot

        self.x = self.x[1:]  # Remove the first x element.
        self.x.append(self.x[-1] + 1)  # Add a new value 1 higher than the last.

        self.y = self.y[1:]  # Remove the first y element

        self.data_line.setData(self.x, self.y)  # Update the data.

    def get_data(self):
        return self.y