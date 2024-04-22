#Author: Haruka Yamamoto
#Date: 18/3/2024
#File Description:
# This python script detects faces using opencv2 and will track important facial regions
# such as the forehead and cheeks. This will be used to apply PPG to extract heart rate and
# blood pressure. The green colour space is chosen for this as it contains the strongest 
# PPG signals.

import numpy as np
import cv2 # import opencv (download mini anaconda and activate a opencv2 to run this)
import plotter
import heartrate
import pyqtgraph as pg
import filter
import scipy


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


def main():
    # Load the pre-trained classifiers. These can be found in opencv2/data/haarcascades
    # but are also inside this files directory for ease of use.
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Basic set up
    cap = cv2.VideoCapture(0)  # Open the webcam device. Initially set as -1 however, only 0 works.
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    p = plotter.Plotter(500,400)

    bpm = heartrate.Heartrate()
    file = open("output.txt", "w")
    
    sos = scipy.signal.iirfilter(4, Wn=[0.67, 3], fs=fps, btype="bandpass",
                             ftype="butter", output="sos")
    live_filter = filter.LiveSosFilter(sos)

    while cv2.waitKey(1) < 0:
        ret, img = cap.read()  # Read a frame from the webcam.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale for face detection.
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces: # currently can detect mutilple faces, impact of this on accuracy to be tested.

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),1)  # Draw a white box around face.
            
            for coords in make_ROI_coords(x, y, w, h):

                x, y, w, h = coords
                roi = img[y:y+h, x:x+w]
                roi[:,:,0] = 0
                roi[:,:,2] = 0
                current_ppg = cv2.mean(roi)[1]

                filtered_ppg = live_filter(current_ppg)

                p.plot(filtered_ppg)
                bpm.bpm_calc(filtered_ppg)
                file.write(f"{filtered_ppg}\n")

        cv2.imshow('img',img)

    cap.release()
    cv2.destroyAllWindows()
    file.close()


if __name__ == "__main__":
    main()
