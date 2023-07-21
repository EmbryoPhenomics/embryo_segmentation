from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Source: https://keras.io/examples/vision/deeplabv3_plus/#building-the-deeplabv3-model

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def dilated_spatial_pyramid_pooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeepLabV3(input_shape, n_classes):
    model_input = keras.Input(shape=input_shape)

    # Rescale by default
    inputs = layers.Rescaling(1.0 / 255)(model_input)

    resnet50 = keras.applications.ResNet50(
        weights=None, include_top=False, input_tensor=inputs
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = dilated_spatial_pyramid_pooling(x)

    input_a = layers.UpSampling2D(
        size=(input_shape[0] // 4 // x.shape[1], input_shape[1] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(input_shape[0] // x.shape[1], input_shape[1] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(n_classes, kernel_size=(1, 1), padding="same", dtype=tf.float32, activation='sigmoid' if n_classes == 1 else 'softmax')(x)
    return keras.Model(inputs=model_input, outputs=model_output)


if __name__ == '__main__':
    model = DeepLabV3(input_shape=(256, 256, 3), n_classes=3)
    model.summary()
