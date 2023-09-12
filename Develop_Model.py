# tf
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import numpy as np
import gc
# reading
import cv2
import os

# vis
import pandas as pd
import matplotlib.pyplot as plt
imgSize = 128
def main():
    Setup_GPU()
    trainDir = 'train/'
    testDir = 'test/'
    classes = ['0','1'] #nutral, happy, angry
    trainingData = create_data(classes,trainDir)
    print(trainingData[0][1])
    index = 0
    newTrainingData = []
    limit = 5000
    counter = 0
    for index in range(len(trainingData)):
        check = trainingData[index][1]
        if check== 1:
            #data inbalance at class 1, set limit to prevent additional values from being added
            if counter <= limit:
                newTrainingData.append(trainingData[index])
                counter = counter +1

        else:
            #add the rest of the training data
            newTrainingData.append(trainingData[index])
    trainingData = newTrainingData

    random.shuffle(trainingData)
    x,y = feature_label_split(trainingData)
    x = x[:limit]
    y = y[:limit]
    x = x/255.0
    x,y,xVal,yVal = shorten_data(x,y,1.1)
    #normalise data

    print('normComplete')

    testData = create_data(classes,testDir)
    x_test, y_test = feature_label_split(testData)
    x_test = x_test[:(int)(limit/2)]
    y_test = y_test[:(int)(limit/2)]
    x_test = x_test/255.0
    model = setup_Model()
    model.summary()
    model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
    print('model compiled')
    model.fit(x,y,epochs= 30 ,validation_data =(xVal,yVal))
    print('model Fitted')
    testLoss, testAcc = model.evaluate(x_test,y_test,verbose = 2)
    print(testAcc)
    model.save('Completed-Model/model.keras')


def shorten_data(x,y,cut):
    x = np.array(x)[:int(len(x)/cut)]
    y = np.array(y)[:int(len(y)/cut)]
    xVal = np.array(x)[int(len(x)/cut):]
    yVal = np.array(y)[int(len(y)/cut):]
    return x,y,xVal,yVal
def setup_Model():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(imgSize,imgSize,3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(3,3),activation= 'relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10))
    return model
def feature_label_split(data):
    x = []
    y = []
    for features ,label in data:
        x.append(features)
        y.append(label)
    x = np.array(x).reshape(-1,imgSize,imgSize,3) #244 by 224 by 3 colours
    y = np.array(y)
    return x,y
def create_data(classes, dir):
    trainingData = []
    for category in classes:
        path = os.path.join(dir,category)
        classNum = classes.index(category)
        for img in os.listdir(path):
            try:
                imgArray = cv2.imread(os.path.join(path,img))
                newArray = cv2.resize(imgArray,(imgSize,imgSize))
                trainingData.append([newArray,classNum])
            except Exception as e:
                print(e)
                pass
    return trainingData


def Setup_GPU():
    #allows gpu accelaration
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu,True)
        except RuntimeError as e:
            print(e)


if __name__ == "__main__":
    main()