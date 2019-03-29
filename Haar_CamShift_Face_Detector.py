import numpy as np
import cv2
import sys
import time

# A demo to use Haar Cascades to first detect an object (in this case, frontal face), and then use CamShift to
# track the object.

face_cascade = cv2.CascadeClassifier('/Users/cgyurgyik/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# ensures that the correct path is used
if face_cascade.empty() is True:
    print("File path is incorrect")
    sys.exit()
else:
    print("File path correct, Haar Cascade(s) detected")



cv2.namedWindow("CV", cv2.WINDOW_NORMAL)
# move the window
cv2.moveWindow("CV", 0,100)
# resize the window
cv2.resizeWindow("CV", 700,700)

cv2.setWindowProperty("CV", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
# set the camera to the user's web cam
camera = cv2.VideoCapture(0)
# let the camera warm up
time.sleep(2.0)


# method to implement cam shift
def camshift_track(frame, bb):
    face_detect = False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x0, y0, w, h = bb
    x1 = x0 + w - 1
    y1 = y0 + h - 1
    hsv_roi = hsv[y0:y1, x0:x1]
    # mask the dark areas to improve results. The first array is the low limit, second the high limit
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    # calculate the hue histogram of unmasked region
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # calculate histogram of back projection
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    # termination criteria: either finish 20 iterations, or move < 0.3 pixel
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.3)
    # apply CamShift to get new location of object
    new_ellipse, track_window = cv2.CamShift(dst, bb, term_crit)

    # circle drawn to display image being tracked
    a,b,c,d = track_window

    # a place holder to determine when the face leaves the frame entirely
    # TODO: need to find better criteria to determine when face leaves the screen
    # Since using CamShift and Hue Based, hands also detected unfortunately
    if a < 10 or b < 10 or c < 10 or d < 10:
        face_detect = False
        print("Face Has Left the Frame")
    else:
        cv2.circle(frame, (int(a + c / 2), int(b + d / 2)), int(d / 2), 255, 2)
        face_detect = True

    return track_window, face_detect


# the bounding box for cam shift, when no face is discovered
global bb
face_detected = False

# a loop for each frame of the camera's real time feed
while True:
    global x, y, w, h
    grabbed, frame = camera.read()
    status = "Face(s) Detected: False"

    # ensures another frame is available
    if grabbed is True:
        # convert the current frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Haar Cascade
        if face_detected is False:
            # detect the object in the cascade
            for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.3, 5):
                # draw a rectangle around the object
                s = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                status = "Face(s) Detected: True"
                bb = (x, y, w, h)
                face_detected = True
                print("Face Detected")

        # CamShift
        if face_detected is True and bb is not None:
            bb, face_detected = camshift_track(frame, bb)
            status = "Face(s) Detected: True"



        # add text to the current frame
        cv2.putText(frame, status, (20, 30), cv2.FONT_ITALIC, 1.0,
                    (0, 0, 255), 2)

        # show the current frame
        cv2.imshow("CV", frame)
        key = cv2.waitKey(1) & 0xFF

        # if 'q' is pressed, close the window
        if key == ord('q'):
            break
    else:
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()