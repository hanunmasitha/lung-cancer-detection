import tensorflow as tf
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow import keras


def seperableConv_2d(filters, kernel_size, pool_size, activation_function):
    conv = tf.keras.models.Sequential([
        tf.keras.layers.SeparableConv2D(filters=filters,
                                        kernel_size=kernel_size,
                                        activation=activation_function,
                                        padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=pool_size),
    ])

    return conv

def fullconnected(units, kernel_initializer, activation_function):
    fullcon = tf.keras.Sequential([
        tf.keras.layers.Dense(units=units,
                              kernel_initializer=kernel_initializer,
                              activation=activation_function
                              ),
        tf.keras.layers.BatchNormalization()
    ])

    return fullcon

def alexnet(activation_function = 'relu',
                    kernel_initializer = 'uniform',
                    optimizer = 'adam',
                    dropout_rate = 0,
                    #learning_rate = 0.01,
                    #momentum_rate = 0.9,
                    class_number = 3,
                    picture_size = [64,64,3]):

    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                               activation=activation_function,
                               input_shape=picture_size),

        seperableConv_2d(96, 3, 2, activation_function),
        seperableConv_2d(192, 3, 2, activation_function),
        tf.keras.layers.Dropout(dropout_rate),
        seperableConv_2d(192, 3, 2, activation_function),
        tf.keras.layers.Dropout(dropout_rate),
    ])

    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())

    # Step 4 - Full Connection
    cnn.add(fullconnected(512, kernel_initializer, activation_function))
    #cnn.add(fullconnected(64, kernel_initializer, activation_function))

    # Step 5 - Output Layer
    cnn.add(tf.keras.layers.Dense(units=class_number, activation='softmax'))

    # Part 3 - Training the CNN
    # Compiling the CNN
    if(optimizer == "Adam"):
        #opt = keras.optimizers.Adam(lr=learning_rate, beta_1 = momentum_rate)
        opt = keras.optimizers.Adam()
    elif(optimizer == "RMSprop"):
        #opt = keras.optimizers.RMSprop(lr=learning_rate, momentum = momentum_rate)
        opt = keras.optimizers.RMSprop()

    cnn.compile(optimizer=opt,
                loss=categorical_crossentropy,
                metrics=['acc'])

    return cnn

from keras.layers import *
from keras.models import Model

# ResNet building block of two layers
def building_block(X, filter_size, filters, stride=1, activation_function = 'relu'):

    # Save the input value for shortcut
    X_shortcut = X

    # Reshape shortcut for later adding if dimensions change
    if stride > 1:

        X_shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    # First layer of the block
    X = Conv2D(filters, kernel_size = filter_size, strides=stride, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation(activation_function)(X)

    # Second layer of the block
    X = Conv2D(filters, kernel_size = filter_size, strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = add([X, X_shortcut])  # Add shortcut value to main path
    X = Activation(activation_function)(X)

    return X


# Full model
def resnet(activation_function = 'relu',
            kernel_initializer = 'uniform',
            optimizer = 'adam',
            dropout_rate = 0,
            class_number = 3,
            picture_size = [32,32,3]):


    # Define the input
    X_input = Input(picture_size)

    # Stage 1
    X = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation(activation_function)(X)
    X = Dropout(dropout_rate)(X)

    # Stage 2
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = Dropout(dropout_rate)(X)

    # Stage 3
    X = building_block(X, filter_size=3, filters=32, stride=2)  # dimensions change (stride=2)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = Dropout(dropout_rate)(X)

    # Stage 4
    X = building_block(X, filter_size=3, filters=64, stride=2)  # dimensions change (stride=2)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = Dropout(dropout_rate)(X)

    # Average pooling and output layer
    X = GlobalAveragePooling2D()(X)
    X = Dense(class_number, activation='softmax', kernel_initializer=kernel_initializer,)(X)

    # Create model
    model = Model(inputs=X_input, outputs=X)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['acc'])
    return model