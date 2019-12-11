import numpy as np
import cv2
import sys
import time
import imutils

# a demo to detect an object based on a sharply contrasting color, and then track it

cv2.namedWindow("Color Detection", cv2.WINDOW_NORMAL)
# move the window
cv2.moveWindow("Color Detection", 0,100)
# resize the window
cv2.resizeWindow("Color Detection", 700,700)

cv2.setWindowProperty("Color Detection", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
# set the camera to the user's web cam

# initialize camera
camera = cv2.VideoCapture(0)
# allow camera to warm up
time.sleep(0.5)

# define upper and lower limits of your desired color
# Use range-detector to determine upper/lower limits found here:
# https://github.com/jrosebr1/imutils/blob/master/bin/range-detector
colorLower = (30, 120, 133)
colorUpper = (46, 255, 255)

# loop through real time frames of camera
while True:
    grabbed, frame = camera.read()

    # usually for reading videos only; breaks if at the end of video
    if grabbed is None:
        break

    # resize frame, blur, and convert to HSV color space
    # frame = imutils.resize(frame, width = 600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for your desired color.
    mask = cv2.inRange(hsv, colorLower, colorUpper)

    # remove dilations and erosions
    mask = cv2.erode(mask, None, iterations = 2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize center
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceeds if at least one contour was found
    if len(cnts) > 0:

        # find the largest contour, and compute minimum enclosing circle
        c = max(cnts, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceeds if radius meets a min size:
        if radius > 2:
            # draw the circle and centroid on the frame, and update list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (20, 20, 255), 3)
            # cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.drawMarker(frame, center, (0,0,255), cv2.MARKER_CROSS, int(radius), 3)

    # show the frame
    cv2.imshow("Color Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # quit if 'q' is pressed
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

