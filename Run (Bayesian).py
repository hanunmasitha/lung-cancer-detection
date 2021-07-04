import os
import shutil

import numpy as np
import tensorflow as tf
from IPython import display
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from hyperopt import Trials, tpe
import random
import time

from TA.ModelBayesian import ModelBayesianLeNet
from TA.Arsitektur import LeNet
from TA import performance_meansure,preprocessing

datasetFolderName = 'Dataset'
sourceFiles = []
classLabels = ['Kanker', 'Normal', 'Tumor']

train_path=datasetFolderName+'/Training/'
validation_path=datasetFolderName+'/Validation/'
test_path=datasetFolderName+'/Testing/'

picture_size = 64

start_time = time.time()

x_train, y_train, x_test, y_test = ModelBayesianLeNet.getDataBaysean()
def getBestParam(best_param=None):
    from hyperas import optim
    activation = ['relu', 'elu', 'tanh', 'sigmoid']
    kernel_initializer = ['random_normal','random_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    optimizer = ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta']
    epochs = [10,5]
    batch_size = [16, 32, 64]
    dropout_rate = [0.2, 0.4]
    if __name__ == '__main__':
        best_run, best_model = optim.minimize(model=ModelBayesianLeNet.getModelBaysean,
                                              data=ModelBayesianLeNet.getDataBaysean,
                                              algo=tpe.suggest,
                                              max_evals=70,
                                              trials=Trials())

        print("Evalutation of best performing model:")
        print(best_model.evaluate(x_test, y_test))
        best_run['activation'] = activation[best_run['activation']]
        best_run['kernel_initializer'] = kernel_initializer[best_run['kernel_initializer']]
        best_run['optimizer'] = optimizer[best_run['optimizer']]
        best_run['epochs'] = epochs[best_run['epochs']]
        best_run['batch_size'] = batch_size[best_run['batch_size']]
        best_run['dropout_rate'] = dropout_rate[best_run['dropout_rate']]

        print("Best performing model chosen hyper-parameters:")
        print(best_run)
        print(best_run['optimizer'])

        return best_run

#================================================================================================================#

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

best_param = getBestParam()
cnn = LeNet.getModel(best_param['activation'],
                    best_param['kernel_initializer'],
                    best_param['optimizer'],
                    best_param['dropout_rate'],
                    len(classLabels),
                    [picture_size,picture_size,3])

display.display(display.Image('small_lenet.png'))

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

model_path = "Save Model\LeNet Model"
cnn.save(model_path)
print("==============TEST RESULTS============")
test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(picture_size, picture_size),
        batch_size=best_param['batch_size'],
        class_mode=None,
        shuffle=False)
predictions = cnn.predict(test_generator, verbose=1)
yPredictions = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
classes = []
for i in true_classes:
    if(i == 0):
        classes.append([1,0,0])
    elif(i == 1):
        classes.append([0,1,0])
    else:
        classes.append([0,0,1])

testAcc,testPrec, testFScore = performance_meansure.my_metrics(true_classes, yPredictions, classes, predictions)
cnn_time = time.time() - start_time
print("Time : {}".format(cnn_time))


