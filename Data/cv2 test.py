import cv2
import time
webcam = cv2.VideoCapture(0)
previous = time.time()
while True:
    current = time.time()
    ret, frame = webcam.read()
    cv2.imshow('test',frame)
    cv2.waitKey(1)
    if((current-previous) > 4):
        cv2.namedWindow('test')
        ret1,frame1 = webcam.read()
        cv2.imshow('image',frame1)
        cv2.waitKey(1000)
        previous = current




