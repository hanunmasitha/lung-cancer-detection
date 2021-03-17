from keras.preprocessing.image import ImageDataGenerator

def imageDatagen():
    datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    return datagen
def getLungCancer(train_path,
                  validation_path,
                  picture_size = 32,
                  batch_size = 15 ):

    datagen = imageDatagen()
    training_set = datagen.flow_from_directory(train_path,
                                                     target_size=(picture_size, picture_size),
                                                     batch_size=batch_size,
                                                     class_mode='categorical')

    # Preprocessing the Test set
    test_set = datagen.flow_from_directory(validation_path,
                                                target_size=(picture_size, picture_size),
                                                batch_size=batch_size,
                                                class_mode='categorical')

    return training_set, test_set