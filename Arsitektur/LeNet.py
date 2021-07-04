import tensorflow as tf
from tensorflow.python.keras.losses import categorical_crossentropy

def getModel(activation_function = 'relu',
            kernel_initializer = 'uniform',
            optimizer = 'adam',
            dropout_rate = 0,
            class_number = 3,
            picture_size = [64,64,3]):
    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=5,
                               activation=activation_function,
                               input_shape=picture_size),

        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=5,
                               activation=activation_function),
        tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               activation=activation_function),
        tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(dropout_rate),
    ])

    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())

    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(units=512,
                                  kernel_initializer=kernel_initializer,
                                  activation=activation_function))
    cnn.add(tf.keras.layers.Dropout(dropout_rate))

    # Step 5 - Output Layer
    cnn.add(tf.keras.layers.Dense(units=class_number,
                                  activation='softmax'))

    # Part 3 - Training the CNN
    # Compiling the CNN
    cnn.compile(optimizer=optimizer,
                loss=categorical_crossentropy,
                metrics=["acc"])
    print(cnn.summary())
    return cnn