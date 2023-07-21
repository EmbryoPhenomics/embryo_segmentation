from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def UNet(input_shape, n_classes):
    inputs = keras.Input(shape=input_shape)

    # Rescale by default
    inputs = layers.Rescaling(1.0 / 255)(inputs)

    # Encoder
    conv1 = layers.Conv2D(16, 3, padding = 'same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Conv2D(16, 3, padding = 'same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(32, 3, padding = 'same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Conv2D(32, 3, padding = 'same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(64, 3, padding = 'same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Conv2D(64, 3, padding = 'same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(128, 3, padding = 'same')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Conv2D(128, 3, padding = 'same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(256, 3, padding = 'same')(pool4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)
    conv5 = layers.Conv2D(256, 3, padding = 'same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)
    pool5 = layers.MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = layers.Conv2D(512, 3, padding = 'same')(pool5)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)
    conv6 = layers.Conv2D(512, 3, padding = 'same')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)

    # Decoder
    up5 = layers.Conv2D(256, 3, padding = 'same')(layers.UpSampling2D(size = (2,2))(conv6))
    up5 = layers.BatchNormalization()(up5)
    up5 = layers.Activation('relu')(up5)
    merge5 = layers.concatenate([conv5,up5], axis = 3)

    conv6_ = layers.Conv2D(256, 3, padding = 'same')(merge5)
    conv6_ = layers.BatchNormalization()(conv6_)
    conv6_ = layers.Activation('relu')(conv6_)
    conv6_ = layers.Conv2D(256, 3, padding = 'same')(conv6_)
    conv6_ = layers.BatchNormalization()(conv6_)
    conv6_ = layers.Activation('relu')(conv6_)

    up6 = layers.Conv2D(128, 3, padding = 'same')(layers.UpSampling2D(size = (2,2))(conv6_))
    up6 = layers.BatchNormalization()(up6)
    up6 = layers.Activation('relu')(up6)
    merge6 = layers.concatenate([conv4,up6], axis = 3)

    conv6 = layers.Conv2D(128, 3, padding = 'same')(merge6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)
    conv6 = layers.Conv2D(128, 3, padding = 'same')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)

    up7 = layers.Conv2D(64, 3, padding = 'same')(layers.UpSampling2D(size = (2,2))(conv6))
    up7 = layers.BatchNormalization()(up7)
    up7 = layers.Activation('relu')(up7)
    merge7 = layers.concatenate([conv3,up7], axis = 3)

    conv7 = layers.Conv2D(64, 3, padding = 'same')(merge7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation('relu')(conv7)
    conv7 = layers.Conv2D(64, 3, padding = 'same')(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation('relu')(conv7)

    up8 = layers.Conv2D(32, 3, padding = 'same')(layers.UpSampling2D(size = (2,2))(conv7))
    up8 = layers.BatchNormalization()(up8)
    up8 = layers.Activation('relu')(up8)
    merge8 = layers.concatenate([conv2,up8], axis = 3)

    conv8 = layers.Conv2D(32, 3, padding = 'same')(merge8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Activation('relu')(conv8)
    conv8 = layers.Conv2D(32, 3, padding = 'same')(conv8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Activation('relu')(conv8)

    up9 = layers.Conv2D(16, 3, padding = 'same')(layers.UpSampling2D(size = (2,2))(conv8))
    up9 = layers.BatchNormalization()(up9)
    up9 = layers.Activation('relu')(up9)
    merge9 = layers.concatenate([conv1,up9], axis = 3)

    conv9 = layers.Conv2D(16, 3, padding = 'same')(merge9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Activation('relu')(conv9)
    conv9 = layers.Conv2D(16, 3, padding = 'same')(conv9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Activation('relu')(conv9)

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(n_classes, 3, activation="sigmoid", padding="same", dtype=tf.float32)(conv9)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

if __name__ == '__main__':
    model = UNet(input_shape=(256, 256, 1), n_classes=1)
    model.summary()
