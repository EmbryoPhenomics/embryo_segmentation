from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def FCN(input_shape, n_classes):
    inputs = keras.Input(shape=input_shape)

    # Rescale by default
    x = layers.Rescaling(1.0 / 255)(inputs)

    # VGG16 Encoder
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    residual1 = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(512, (3,3), padding='same')(residual1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    residual2 = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(512, (3,3), padding='same')(residual2)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(4096, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(4096, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(n_classes, 3, padding='same')(x)

    # # FCN8 Decoder
    x = layers.Conv2DTranspose(n_classes, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)    
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D((2,2))(x)

    p = layers.Conv2D(n_classes, (3,3), padding='same')(residual2)
    p = layers.Activation('relu')(p)

    x = layers.add([x, p])

    x = layers.Conv2DTranspose(n_classes, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)    
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D((2,2))(x)

    p = layers.Conv2D(n_classes, (3,3), padding='same')(residual1)
    p = layers.Activation('relu')(p)

    x = layers.add([x, p])

    # Classification
    x = layers.Conv2DTranspose(n_classes, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)    
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D((8,8))(x)
    outputs = layers.Conv2D(n_classes, 3, activation="sigmoid", padding="same", dtype=tf.float32)(x)

    return keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    model = FCN(input_shape=(256, 256, 3), n_classes=3)
    model.summary()