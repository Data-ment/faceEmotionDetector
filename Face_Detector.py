import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
import csv
import datetime
import time
#write test for making files
#write test for reading files
# test model for accuracy
#add a timer method
#add an inactive method

#cv2 keeps taking a pic after a few ms find a way to solve it.

imgSize = 128
model = keras.models.load_model('Completed-Model/model.keras')
waitTimeSeconds = 10 #time between each collection
inactiveTimerMinutes = 10
def main():
    previousTime = time.time()
    file , writer = setup_writer(datetime.datetime.now())
    cascadeDir = "haarcascade_frontalface_default.xml"
    detectorModel = cv2.CascadeClassifier(cascadeDir)
    webcam = cv2.VideoCapture(0)
    print('loading complete')
    timeKey = 1
    k = 1
    isInactive = False
    while True:
        #keep running until q is pressed to end program
        currentTime = time.time()
        difference = currentTime - previousTime
        cv2.namedWindow('face')
        ret, frame = webcam.read()
        frameGrayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceCoordinates = detectorModel.detectMultiScale(frameGrayscale)

        k = cv2.waitKey(1)
        if k == 81 or k == 113:
            break

        if difference < waitTimeSeconds: #threshold between pics
            continue

        if difference >= inactiveTimerMinutes *60 and isInactive == True:
            #user is not here after certain amount of time. this shuts down program
            print("user is inactive. program shutting down now")
            break

        if len(faceCoordinates ) ==0: #no face was detected
             #this wont stop the program if an incorrect face is detected
            print('cant detect face')
            isInactive = True
            continue
        isInactive = False #user is here
        cropped = crop_image(faceCoordinates, frame)


        image = cv2.resize(cropped, (imgSize, imgSize))
        image = np.array(image).reshape(-1, imgSize, imgSize, 3)
        prediction = model.predict(image)
        prediction = prediction.argmax(axis=-1)
        cv2.imshow('face',cropped)



        previousTime = currentTime #privous time needs to be at end to allow it to run over
        #each iteration should be around 10 seconds. due to conditions like time it takes to load and cases where would give it extra delay :(.

        print( datetime.datetime.now().strftime('%m/%d, %H:%M:%S'), ' ' , prediction)

        row = [currentTime,prediction] #0: neutral 1: happy
        writer.writerow(row)




        # cv2.destroyAllWindows()
    # file.close()
    webcam.release()
    cv2.destroyAllWindows()


def crop_image(faceCoordinates, frame):
    for (x, y, w, h) in faceCoordinates:
        # crops image to only show face
        cropped = frame[y:y + h, x:x + w]
        # shows image
    return cropped


def get_prediction(faceCoordinates, frame):
    #using emotion model to detect the emotion of the face
    for (x, y, w, h) in faceCoordinates:
        #crops image to only show face
        cropped = frame[y:y + h, x:x + w]
    #shows image
    cv2.imshow('test', cropped)

    image = cv2.resize(cropped, (imgSize, imgSize))
    image = np.array(image).reshape(-1, imgSize, imgSize, 3)
    prediction = model.predict(image)
    prediction = prediction.argmax(axis=-1)
    return prediction


def get_coordinates(detectorModel, webcam):
    #using cascade to get the coordinates for the face region
    cv2.namedWindow('test')
    ret, frame = webcam.read()
    frameGrayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCoordinates = detectorModel.detectMultiScale(frameGrayscale)
    return faceCoordinates, frame


def setup_writer(previousTime):
    fileName = 'Data/' + str(previousTime.strftime("%Y%m%d-%H%M%S")) + '.csv' #name of file
    file = open(fileName, 'w')
    writer = csv.writer(file, lineterminator='\n')
    header = ['DateTime','Class']
    writer.writerow(header)
    return file,writer


if __name__ == "__main__":
    main()