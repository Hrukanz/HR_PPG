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
import time
import heartrate

# ----- Global / Static Variables -----
# Perhaps not the best practice but here for testing




def draw_rect(img, input, colour, thickness):
    # Rect func has a start and end point of the rect drawn given by two tuples.
    # x,y shows the origin of the rect (top left corner) and the x+w, y+h shows the end (bottom right corner).
    x, y, w, h = input
    cv2.rectangle(img, (x,y), (x+w, y+h), colour, thickness)  #draw rectangle


def get_green_colour_space(img, input):
    # Gets only the green colour space of the video feed.
    # Currently uses default BGR but could try use HSV.
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Research point: Different colour spaces
        
    x, y, w, h = input
    roi = img[y:y+h, x:x+w]

    lower_green_thresh = np.array((0,0,0))
    upper_green_thresh = np.array((0,255,0))
    mask = cv2.inRange(img, lower_green_thresh, upper_green_thresh)
    res = cv2.bitwise_and(img,img, mask= mask)


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

    cap = cv2.VideoCapture(0)  # Open the webcam device. Initially set as -1 however, only 0 works.
    p = plotter.Plotter(500,400)
    bpm = heartrate.Heartrate()
    roi_avg = 0

    while cv2.waitKey(1) < 0:
        ret, img = cap.read()  # Read a frame from the webcam.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale for face detection.

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces: # currently can detect mutilple faces, impact of this on accuracy to be tested.

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),1)  # Draw a white box around face.
            
            for coords in make_ROI_coords(x, y, w, h):
                # draw_rect(img, roi, (0,0,255), -1) #lightly filled green rectangle for roi
                # get_green_colour_space(img, roi)

                x, y, w, h = coords
                roi = img[y:y+h, x:x+w]
                roi[:,:,0] = 0
                roi[:,:,2] = 0
                p.plot(cv2.mean(roi)[1])
                bpm.bpm_calc(cv2.mean(roi)[1])


        cv2.imshow('img',img)

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()