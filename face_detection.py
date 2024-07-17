#Author: Haruka Yamamoto
#Date: 18/3/2024
#File Description:
# This python script detects faces using opencv2 and will track important facial regions
# such as the forehead and cheeks. This will be used to apply PPG to extract heart rate.
# The green colour space is chosen for this as it contains the strongest 
# PPG signals.

from PyQt5.QtWidgets import QApplication # pip install pyqt5
import numpy as np
import cv2 # import opencv (download mini anaconda and activate a opencv2 to run this)
import filter
import scipy
import pyqt_grapher
import sys
import peak_detection as pd

def make_ROI_coords(x, y, w, h):

    # Following lists show represent the x, y, w, and h values for each ROI.
    # Each ROI is calculated from the x, y, w, and h values from the face rect.
    # They are the x, y, w, h values altered by the percentage increase of the respective values.
    # This places the ROIs proportionally on the face which move with the person.

    # note: currently not acctually ROIs, just coords!

    forehead = [(x + int(w * 0.33)),
            (y + int(h * 0.08)), 
            (int(w * 0.33)), 
            (int(h * 0.13))]

    left_cheek = [(x + int(w * 0.17)),
                (y + int(h * 0.5)), 
                (int(w * 0.2)), 
                (int(h * 0.3))]

    right_cheek = [(x + int(w * 0.67)),
                (y + int(h * 0.5)), 
                (int(w * 0.2)), 
                (int(h * 0.3))]

    # return [forehead, left_cheek, right_cheek]
    return [forehead]


def get_ppg(graph, file, live_filter, roi, peak_detect):
    # func allows for data to be processed using external class files
    # such as filtering, plotting, peak detection, and writing

    current_ppg = cv2.mean(roi)[1] # ppg signal value from the average value of green pixels in roi

    filtered_ppg = live_filter(current_ppg) # filtered value from pyqt_grapher

    graph.update(filtered_ppg) # update plot

    peak_detect.peaks(graph.get_data()) # update peak detection

    file.write(f"{filtered_ppg}\n") # write to output file

    return (3*len(peak_detect.get_wd()['peaklist'])) # return heart rate bpm


def main():
    # Load the pre-trained classifiers. These can be found in opencv2/data/haarcascades
    # but are also inside this files directory for ease of use.
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Basic set up of open cv
    cap = cv2.VideoCapture(0)  # Open the webcam device. Initially set as -1 however, only 0 works.
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0.0) # Disable automatic white balance to remove extra noise

    # Basic init of external class files or variables used in main loop
    app = QApplication(sys.argv) # used for graphing class
    graph = pyqt_grapher.Graph(int(fps)) # init graph class
    graph.show()
    peak_detect = pd.PeakDetection() # init peak detection
    find_faces = True
    file = open("output.txt", "w") # output file of ppg values for further testing

    # broad cut off frequency between 0.7Hz (42 bpm) and 3Hz (180 bpm)
    # butterworth is a bandpass filter which is flat in its passband
    sos = scipy.signal.iirfilter(4, Wn=[0.7, 3], fs=fps, btype="bandpass",
                             ftype="butter", output="sos") # generate sos using scipy with given specs
    live_filter = filter.LiveSosFilter(sos) # init live filter using the generated sos

    while True: # main loop
        ret, img = cap.read()  # Read a frame from the webcam.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale for face detection.
        if find_faces:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces: # currently can detect mutilple faces, impact of this on accuracy to be tested.

                face_coords = (x, y, w ,h) # stores face region coords for locking mode

                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),1)  # Draw a white box around face.
                
                for coords in make_ROI_coords(x, y, w, h): # currently unrequired loop. Needed if multiple ROIs used
                        # seperates green channel from roi and gets heart rate
                        x, y, w, h = coords
                        roi = img[y:y+h, x:x+w]
                        roi[:,:,0] = 0 # removes red values from roi
                        roi[:,:,2] = 0 # same for blue
                        hr = get_ppg(graph, file, live_filter, roi, peak_detect) # sends data to func for further processing
                        # prints heart rate to screen
                        cv2.putText(img, str(hr), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
                        roi_coords = (x, y, w, h) # stored coords of roi for locking mode

            find_faces = (-1 == cv2.waitKey(1)) # returns False if any key is pressed
        else:
            # find faces is False and therefore the face region and roi is locked
            # this is done to minimise noise and improve performance by reducing face tracking computation
            # code here does the same as above but with the locked coords

            xf, yf, wf, hf = face_coords # fixed face coords stored
            xr, yr, wr, hr = roi_coords # same for roi
            cv2.rectangle(img,(xf,yf),(xf+wf,yf+hf),(255,255,255),1)  # Draw a white box around face.
            
            lock_roi = img[yr:yr+hr, xr:xr+wr]
            lock_roi[:,:,0] = 0
            lock_roi[:,:,2] = 0

            hr = get_ppg(graph, file, live_filter, lock_roi, peak_detect)
            cv2.putText(img, str(hr), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            find_faces = (-1 != cv2.waitKey(1))

        cv2.imshow('img',img)
        cv2.waitKey(2) # keep window running

    cap.release()
    cv2.destroyAllWindows()
    file.close()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()