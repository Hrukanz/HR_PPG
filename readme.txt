Author: Haruka Yamamoto
Date: 5/5/2024

This program requires the following libraries

- OpenCV
- Numpy
- Scipy
- Pyqt5
- Pyqtgraph

OpenCV need to be run in ananconda which lets you activate environments for your terminal.
Run the "conda activate your_environment" command in conda once installed and set up.
Once set up run "python face_detection.py" to start the program.
The face detecting ROI can be toggled by pressing any key while the program is running.
This can reduce computation time providing a smoother output.

The code structure is very simple.
The face detection file is the main program with the other named files doing what they are named as.
The peak detection however, shares the 600 data point list with the grapher class to remove copied code.
The size of the list is determined by the fps of the camera used and the length of time for data collection.
More information commented in the files themselves.