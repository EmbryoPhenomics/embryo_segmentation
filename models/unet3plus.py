import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.activations as activations

# Defining the encoder's down-sampling blocks.
def encoder_block(inputs, n_filters, kernel_size, strides):
    encoder = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activations.gelu)(encoder)
    encoder = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', use_bias=False)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activations.gelu)(encoder)
    return encoder


# Defining the decoder's up-sampling blocks.
def upscale_blocks(inputs):
    n_upscales = len(inputs)
    upscale_layers = []

    for i, inp in enumerate(inputs):
        p = n_upscales - i
        u = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2**p, padding='same')(inp)

        for i in range(2):
            u = layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(u)
            u = layers.BatchNormalization()(u)
            u = layers.Activation(activations.gelu)(u)

        upscale_layers.append(u)
    return upscale_layers


# Defining the decoder's whole blocks.
def decoder_block(layers_to_upscale, inputs):
    upscaled_layers = upscale_blocks(layers_to_upscale)

    decoder_blocks = []

    for i, inp in enumerate(inputs):
        d = layers.Conv2D(filters=64, kernel_size=3, strides=2**i, padding='same', use_bias=False)(inp)
        d = layers.BatchNormalization()(d)
        d = layers.Activation(activations.gelu)(d)
        d = layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(d)
        d = layers.BatchNormalization()(d)
        d = layers.Activation(activations.gelu)(d)

        decoder_blocks.append(d)

    decoder = layers.concatenate(upscaled_layers + decoder_blocks)
    decoder = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=False)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activations.gelu)(decoder)

    return decoder


def UNet3plus(input_shape, n_classes=1):
    inputs = layers.Input(input_shape)

    # Rescale by default
    inputs = layers.Rescaling(1.0 / 255)(inputs)

    e1 = encoder_block(inputs, n_filters=32, kernel_size=3, strides=1)
    e2 = encoder_block(e1, n_filters=64, kernel_size=3, strides=2)
    e3 = encoder_block(e2, n_filters=128, kernel_size=3, strides=2)
    e4 = encoder_block(e3, n_filters=256, kernel_size=3, strides=2)
    e5 = encoder_block(e4, n_filters=512, kernel_size=3, strides=2)

    d4 = decoder_block(layers_to_upscale=[e5], inputs=[e4, e3, e2, e1])
    d3 = decoder_block(layers_to_upscale=[e5, d4], inputs=[e3, e2, e1])
    d2 = decoder_block(layers_to_upscale=[e5, d4, d3], inputs=[e2, e1])
    d1 = decoder_block(layers_to_upscale=[e5, d4, d3, d2], inputs=[e1])

    output = layers.Conv2D(n_classes, 1, activation='sigmoid', padding='same', dtype=tf.float32)(d1)

    model = models.Model(inputs, output)
    return model

if __name__ == '__main__':
    model = UNet3plus(input_shape=(256, 256, 1), n_classes=1)
    model.summary()