import cv2
import numpy as np
import imutils


cv2.namedWindow("CV")
camera = cv2.VideoCapture(0)

while True:
    grabbed, frame = camera.read()
    status = "No Target"

    # convert to gray scale, blur, and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # detect circles
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 500)
    # double check these parameters might need minRadius=

    # find contour lines
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over contour lines
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        # creates an array with the number of vertices
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # ensure that contours form a rectangular
        if len(approx) >= 4 and len(approx) <= 6:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspectRatio = w / float(h)

            # compute the solidity
            area = cv2.contourArea(c)
            hullArea = cv2.contourArea(cv2.convexHull(c))
            solidity = area / float(hullArea)

            # ensure that solidity, w, h, and aspect ratio fall within bounds
            keepDims = w > 25 and h > 25
            keepSolidity = solidity > 0.9
            keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

            # ensure that the contour passes all our tests
            if keepDims and keepSolidity and keepAspectRatio:
                # draw an outline around the target and update the status
                # text
                cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
                status = "Rectangular Target(s) Acquired"
    # ensure that the contours form a circle
    # if circles is not None:
    #     # convert the x,y coordinates and radius to integers
    #     circles = np.round(circles[0, :]).astype('int')
    #
    #     # loop over the coordinates/radius
    #     for (x,y,r) in circles:
    #         # draw the circle in output image
    #         cv2.circle(frame, (x,y), r, (0, 255, 0), 4)
    #         cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    #         status = "Circle Target(s) Acquired"



    # draw the status text on the frame
    cv2.putText(frame, status, (20,30), cv2.FONT_ITALIC, 0.5,
                (0,0,255), 2)

    # show the frame and record if a key is pressed
    cv2.imshow("CV", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()