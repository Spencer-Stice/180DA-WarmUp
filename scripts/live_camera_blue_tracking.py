'''
Sources: 
https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html

Changes:
I combined parts of the above two sources and added the section where only the largest
contour is drawn onto the image as opposed to drawing boxes around all the blue objects.
'''

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255]) 

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #Threshold the blue values
    mask = cv.inRange(frame_HSV, lower_blue, upper_blue)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    if largest_contour is not None:
        x, y, w, h = cv.boundingRect(largest_contour)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()