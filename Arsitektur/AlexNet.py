import tensorflow as tf
from tensorflow.python.keras.losses import categorical_crossentropy

def getModel(activation_function = 'relu',
            kernel_initializer = 'uniform',
            optimizer = 'adam',
            dropout_rate = 0,
            class_number = 3,
            picture_size = [64,64,3]):
    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=11,
                               strides=4,
                               padding='same',
                               activation=activation_function,
                               input_shape=picture_size),
        tf.keras.layers.Lambda(tf.nn.local_response_normalization),
        tf.keras.layers.MaxPool2D(2, strides=2),

        tf.keras.layers.Conv2D(filters=256, kernel_size=5,
                               strides=4,
                               padding='same',
                               activation=activation_function),
        tf.keras.layers.Lambda(tf.nn.local_response_normalization),
        tf.keras.layers.MaxPool2D(2, strides=2),

        tf.keras.layers.Conv2D(filters=384, kernel_size=3,
                               strides=4,
                               padding='same',
                               activation=activation_function),

        tf.keras.layers.Conv2D(filters=384, kernel_size=3,
                               strides=4,
                               padding='same',
                               activation=activation_function),

        tf.keras.layers.Conv2D(filters=384, kernel_size=3,
                               strides=4,
                               padding='same',
                               activation=activation_function),
    ])

    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())

    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(4096,
                                  kernel_initializer=kernel_initializer,
                                  activation=activation_function))
    cnn.add(tf.keras.layers.Dropout(dropout_rate))
    cnn.add(tf.keras.layers.Dense(4096,
                                  kernel_initializer=kernel_initializer,
                                  activation=activation_function))
    cnn.add(tf.keras.layers.Dropout(dropout_rate))
    # cnn.add(fullconnected(64, kernel_initializer, activation_function))

    # Step 5 - Output Layer
    cnn.add(tf.keras.layers.Dense(units=class_number, activation='softmax'))

    # Part 3 - Training the CNN
    # Compiling the CNN
    cnn.compile(optimizer=optimizer,
                loss=categorical_crossentropy,
                metrics=['acc'])
    
    return cnn