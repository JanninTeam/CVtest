from collections import deque
import imutils # to make basic image processing functions such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with OpenCV
from imutils.video import VideoStream # for video streaming
import numpy as np # for numerical processing
import argparse # to parse command line arguments
import cv2 # OpenCV library
import time # to keep track of time

# Main function to run the program
def main():
    print("Starting program...")

    # construct the argument parse and parse the arguments
    arguments = argparse.ArgumentParser()
    arguments.add_argument("-v", "--video", help="path to the (optional) video file")
    arguments.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    args = vars(arguments.parse_args())
    
    # define the lower and upper boundaries of the "orange" ball in the HSV color space, then initialize the list of tracked points
    COLOURLOWER = (0, 140, 140)
    COLOURUPPER = (255, 255, 255)
    
    print(args["buffer"])
    # initialize the list of tracked points, the frame counter, and the coordinate deltas
    trackedPoints = deque(maxlen=args["buffer"])

    # if a video path was not supplied, grab the reference to the webcam
    if not args.get("video", False):
        videoStream = VideoStream(src=0).start()
    # otherwise, grab a reference to the video file
    else:
        videoStream = cv2.VideoCapture(args["video"])

    # allow the camera or video file to warm up
    time.sleep(2.0)
    
    while True:
        frame = videoStream.read()
        frame = frame[1] if args.get("video", False) else frame

        if frame is None:
            break

        frame = imutils.resize(frame, width=600)
        blurredFrame = cv2.GaussianBlur(frame, (11, 11), 0)
        hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsvFrame, COLOURLOWER, COLOURUPPER)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        centre = None

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            M = cv2.moments(c)
            centre = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, centre, 5, (0, 0, 255), -1)

        trackedPoints.appendleft(centre)

        for i in range(1, len(trackedPoints)):
            if trackedPoints[i - 1] is None or trackedPoints[i] is None:
                continue

            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, trackedPoints[i - 1], trackedPoints[i], (0, 0, 255), thickness)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    if not args.get("video", False):
        videoStream.stop()

    else:
        videoStream.release()

    cv2.destroyAllWindows()

        
if __name__ == "__main__":
    main()
