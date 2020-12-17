import tensorflow as tf
from tensorflow.keras import layers


if __name__ == "__main__":
    '''功能性API搭建ResNET'''

    # ------------------- model ------------------------
    inputs = tf.keras.Input(shape=(32, 32, 3), name='img')
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding="same")(block_1_output)
    x = layers.Conv2D(64, 3, activation='relu', padding="same")(x)
    block_2_output = layers.add([x, block_1_output])        # ResNET

    x = layers.Conv2D(64, 3, activation='relu', padding="same")(block_2_output)
    x = layers.Conv2D(64, 3, activation='relu', padding="same")(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)

    model = tf.keras.Model(inputs, outputs, name="toy_resnet")
    model.summary()

    # ------------------------ train ------------------
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-3),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['acc'],
    )
    model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=2, validation_split=0.2)

