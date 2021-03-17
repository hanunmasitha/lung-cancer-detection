from TA.hyperparameter import *
from TA.preprocessing import *
from TA.performance_meansure import *

import shutil
import sys
import os
import time

import numpy as np  # linear algebra
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import random

datasetFolderName = 'Dataset'
train_path=datasetFolderName+'/Training/'
validation_path=datasetFolderName+'/Validation/'
test_path=datasetFolderName+'/Testing/'
sourceFiles = []
classLabels = ['Kanker', 'Normal', 'Tumor']
picture_size = 32

start_time = time.time()
def transferBetweenFolders(source, dest, splitRate):
    global sourceFiles
    sourceFiles = os.listdir(source)
    if (len(sourceFiles) != 0):
        transferFileNumbers = int(len(sourceFiles) * splitRate)
        transferIndex = random.sample(range(0, len(sourceFiles)), transferFileNumbers)
        for eachIndex in transferIndex:
            shutil.move(source + str(sourceFiles[eachIndex]), dest + str(sourceFiles[eachIndex]))
    else:
        print("No file moved. Source empty!")


def transferAllClassBetweenFolders(source, dest, splitRate):
    for label in classLabels:
        transferBetweenFolders(datasetFolderName + '/' + source + '/' + label + '/',
                               datasetFolderName + '/' + dest + '/' + label + '/',
                               splitRate)


# First, check if test folder is empty or not, if not transfer all existing files to train
transferAllClassBetweenFolders('Testing', 'Training', 1.0)
# Now, split some part of train data into the test folders.
transferAllClassBetweenFolders('Training', 'Testing', 0.20)

X = []
Y = []


def prepareNameWithLabels(folderName):
    sourceFiles = os.listdir(datasetFolderName + '/Training/' + folderName)
    for val in sourceFiles:
        X.append(val)
        if (folderName == classLabels[0]):
            Y.append(0)
        elif (folderName == classLabels[1]):
            Y.append(1)
        else:
            Y.append(2)


# Organize file names and class labels in X and Y variables
prepareNameWithLabels(classLabels[0])
prepareNameWithLabels(classLabels[1])
prepareNameWithLabels(classLabels[2])

X = np.asarray(X)
Y = np.asarray(Y)

train_data, validation_data = preprocessing.getLungCancer(train_path,
                                                          validation_path,
                                                          picture_size)

best_param = hyperparameter.GridSearch(train_data,
                                       arsitektur.create_model_resnet)

cnn = arsitektur.create_model_resnet(best_param['activation_function'],
               #best_param['kernel_initializer'],
               best_param['optimizer'],
               #best_param['dropout_rate']
                )

print(cnn.summary())
# ===============Stratified K-Fold======================
skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(X, Y)
foldNum=0
for train_index, val_index in skf.split(X, Y):
    #First cut all images from validation to train (if any exists)
    transferAllClassBetweenFolders('Validation', 'Training', 1.0)
    foldNum+=1
    print("Results for fold",foldNum)

    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]
    # Move validation images of this fold from train folder to the validation folder
    for eachIndex in range(len(X_val)):
        classLabel=''
        if(Y_val[eachIndex]==0):
            classLabel=classLabels[0]
        elif(Y_val[eachIndex]==1):
            classLabel=classLabels[1]
        else:
            classLabel=classLabels[2]
        #Then, copy the validation images to the validation folder
        shutil.move(datasetFolderName+'/Training/'+classLabel+'/'+X_val[eachIndex],
                    datasetFolderName+'/Validation/'+classLabel+'/'+X_val[eachIndex])

    training_set, test_set = preprocessing.getLungCancer(train_path,
                                                         validation_path,
                                                         picture_size,
                                                         best_param['batch_size'])

    # Training the CNN on the Training set and evaluating it on the Test set
    fit = cnn.fit(x=training_set,
                  epochs=best_param['epochs'],
                  validation_data=test_set,
                  verbose=1)

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(['acc', 'loss']):
    ax[i].plot(fit.history[met])
    ax[i].plot(fit.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])
plt.show()

print("==============TEST RESULTS============")
test_datagen = preprocessing.imageDatagen()

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(picture_size, picture_size),
        batch_size=best_param['batch_size'],
        class_mode=None,
        shuffle=False)
predictions = cnn.predict(test_generator, verbose=1)
yPredictions = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

testAcc,testPrec, testFScore = performance_meansure.my_metrics(true_classes, yPredictions)

cnn_time = time.time() - start_time
print("Time : {}".format(cnn_time))