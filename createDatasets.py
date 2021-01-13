import numpy as np
import os
import cv2
import random
import json
import pickle

imageList = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
genderToInt = {'m': 0, 'f': 1}

TRAIN_SAMPLE_SIZE = 7  # 1 - 10
RESIZE_IMG_SIZE = 150

leftEars = set(open('awe/lefts.txt').read().replace("\n", " ").split(" "))
rightEars = set(open('awe/rights.txt').read().replace("\n", " ").split(" "))

def readImage(subdir, imagePath):
    try:
        annotations = json.load(open(subdir + '/annotations.json', encoding='utf-8'))
        img_array = cv2.imread(subdir + '/' + imagePath + '.png')
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        resized_img_array = cv2.resize(img_array, (RESIZE_IMG_SIZE, RESIZE_IMG_SIZE))
        personId = int(subdir[4:])-1
        imageNumber = (personId - 1) * 10 + int(imagePath)
        side = 0 if str(imageNumber) in leftEars else 1
        if 1 <= int(annotations['ethnicity']) <= 6:
            return [resized_img_array, personId, int(annotations['ethnicity']) - 1, genderToInt[annotations['gender']], side]
    except Exception as e:
        return None

def createIdTrainingAndTestData():

    subdirs = [x[0] for x in os.walk('awe/')][1:]
    trainingData, testData = [], []

    for subdir in subdirs:
        selectionTrain = random.sample(imageList, TRAIN_SAMPLE_SIZE)
        selectionData = list(set(imageList) - set(selectionTrain))

        for imagePath in selectionTrain:
            imgData = readImage(subdir, imagePath)
            if imgData:
                trainingData.append(readImage(subdir, imagePath))

        for imagePath in selectionData:
            imgData = readImage(subdir, imagePath)
            if imgData:
                testData.append(readImage(subdir, imagePath))

    random.shuffle(trainingData)
    random.shuffle(testData)

    return trainingData, testData

trainingData, testData = createIdTrainingAndTestData()

if not os.path.exists('datasets'):
    os.makedirs('datasets')

X_train = []
y_train = []

X_test = []
y_test = []

for trainingSample in trainingData:
    X_train.append(trainingSample[0])
    y_train.append([trainingSample[1], trainingSample[2], trainingSample[3], trainingSample[4]])

for trainingSample in testData:
    X_test.append(trainingSample[0])
    y_test.append([trainingSample[1], trainingSample[2], trainingSample[3], trainingSample[4]])

X_train = np.array(X_train).reshape(-1, RESIZE_IMG_SIZE, RESIZE_IMG_SIZE, 3)
y_train = np.array(y_train)

X_test = np.array(X_test).reshape(-1, RESIZE_IMG_SIZE, RESIZE_IMG_SIZE, 3)
y_train = np.array(y_train)

pickle_out = open("datasets/X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()
pickle_out = open("datasets/y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()


pickle_out = open("datasets/X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()
pickle_out = open("datasets/y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()