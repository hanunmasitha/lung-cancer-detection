import tensorflow as tf
from tensorflow.python.keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

datasetFolderName = 'Dataset'
sourceFiles = []
classLabels = ['Kanker', 'Normal', 'Tumor']

train_path=datasetFolderName+'/Training/'
validation_path=datasetFolderName+'/Validation/'
test_path=datasetFolderName+'/Testing/'

TARGET_SIZE = 64
def getDataBaysean():
    batch_size = 16

    datasetFolderName = 'Dataset'
    train_path = datasetFolderName + '/Training/'
    validation_path = datasetFolderName + '/Validation/'
    TARGET_SIZE = 64

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(TARGET_SIZE, TARGET_SIZE),
                                                     batch_size=batch_size,
                                                     class_mode='categorical')

    # Preprocessing the Test set
    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)
    test_set = test_datagen.flow_from_directory(validation_path,
                                                target_size=(TARGET_SIZE, TARGET_SIZE),
                                                batch_size=batch_size,
                                                class_mode='categorical')

    x_train, y_train = training_set.next()
    x_test, y_test = test_set.next()
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = getDataBaysean()

from hyperas.distributions import choice
from hyperopt import STATUS_OK
def getModelBaysean():
    classLabels = ['Kanker', 'Normal', 'Tumor']
    activation = {{choice(['relu', 'elu', 'tanh', 'sigmoid'])}}
    kernel_initializer = {{choice(['random_normal','random_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])}}
    optimizer = {{choice(['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta'])}}
    epochs = {{choice([10,5])}} # You can also try 20, 30, 40, etc...
    batch_size = {{choice([16, 32, 64])}}  # You can also try 2, 4, 8, 16, 32, 64, 128 etc...
    dropout_rate = {{choice([0.2, 0.4])}}

    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=5,
                               activation=activation,
                               input_shape=[TARGET_SIZE, TARGET_SIZE, 3]),

        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=5,
                               activation=activation),
        tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               activation=activation),
        tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(dropout_rate),
    ])

    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())

    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(units=512,
                                  kernel_initializer=kernel_initializer,
                                  activation=activation))
    cnn.add(tf.keras.layers.Dropout(dropout_rate))

    # Step 5 - Output Layer
    cnn.add(tf.keras.layers.Dense(units=len(classLabels),
                                  activation='softmax'))

    # Part 3 - Training the CNN
    # Compiling the CNN
    cnn.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=["acc"])

    cnn.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))

    score, acc = cnn.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': cnn}