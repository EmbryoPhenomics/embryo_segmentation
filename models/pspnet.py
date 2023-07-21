from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def vgg(inputs):
    conv1 = layers.Conv2D(32, 3, padding = 'same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Conv2D(32, 3, padding = 'same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, padding = 'same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Conv2D(64, 3, padding = 'same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, 3, padding = 'same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Conv2D(128, 3, padding = 'same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, 3, padding = 'same')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Conv2D(256, 3, padding = 'same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(512, 3, padding = 'same')(pool4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)
    conv5 = layers.Conv2D(512, 3, padding = 'same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)
    
    return conv5, [conv1, conv2, conv3, conv4]

def unet_decoder(x, n_classes, resnet, skip_connections):
    conv1, conv2, conv3, conv4 = skip_connections

    up6 = layers.Conv2D(256, 2, padding = 'same')(layers.UpSampling2D(size = (2,2))(x))
    up6 = layers.BatchNormalization()(up6)
    up6 = layers.Activation('relu')(up6)
    merge6 = layers.concatenate([conv4,up6], axis = 3)

    conv6 = layers.Conv2D(256, 3, padding = 'same')(merge6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)
    conv6 = layers.Conv2D(256, 3, padding = 'same')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation('relu')(conv6)

    up7 = layers.Conv2D(128, 2, padding = 'same')(layers.UpSampling2D(size = (2,2))(conv6))
    up7 = layers.BatchNormalization()(up7)
    up7 = layers.Activation('relu')(up7)
    merge7 = layers.concatenate([conv3,up7], axis = 3)

    conv7 = layers.Conv2D(128, 3, padding = 'same')(merge7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation('relu')(conv7)
    conv7 = layers.Conv2D(128, 3, padding = 'same')(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation('relu')(conv7)

    up8 = layers.Conv2D(64, 2, padding = 'same')(layers.UpSampling2D(size = (2,2))(conv7))
    up8 = layers.BatchNormalization()(up8)
    up8 = layers.Activation('relu')(up8)
    merge8 = layers.concatenate([conv2,up8], axis = 3)

    conv8 = layers.Conv2D(64, 3, padding = 'same')(merge8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Activation('relu')(conv8)
    conv8 = layers.Conv2D(64, 3, padding = 'same')(conv8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Activation('relu')(conv8)

    up9 = layers.Conv2D(32, 2, padding = 'same')(layers.UpSampling2D(size = (2,2))(conv8))
    up9 = layers.BatchNormalization()(up9)
    up9 = layers.Activation('relu')(up9)
    merge9 = layers.concatenate([conv1,up9], axis = 3)

    if resnet:
        conv9 = layers.Conv2D(32, 2, padding = 'same')(layers.UpSampling2D(size = (2,2))(merge9))
    else:
        conv9 = layers.Conv2D(32, 3, padding = 'same')(merge9)

    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Activation('relu')(conv9)
    conv9 = layers.Conv2D(32, 3, padding = 'same')(conv9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Activation('relu')(conv9)

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(n_classes, 3, activation="sigmoid", padding="same", dtype=tf.float32)(conv9)
    return outputs

def PSPNet(input_shape, n_classes, encoder='vgg', use_unet_decoder=False):

    inputs = keras.Input(shape=input_shape)

    # Rescale by default
    inputs = layers.Rescaling(1.0 / 255)(inputs)

    # Encoder specification --------------------------------------------
    if encoder == 'vgg':
        feature_maps, skip_connections = vgg(inputs)
    elif encoder == 'ResNet50':
        resnet50 = keras.applications.ResNet50(
            weights=None, include_top=False, input_tensor=inputs
        )
        feature_maps = resnet50.get_layer('conv5_block3_out').output

        skip_connection_names = [
            'conv1_relu',
            'conv2_block3_out', 
            'conv3_block4_out', 
            'conv4_block6_out'
        ]
        skip_connections = []
        for name in skip_connection_names:
            skip_connections.append(resnet50.get_layer(name).output)

    elif encoder == 'ResNet101':
        resnet101 = keras.applications.ResNet101(
            weights=None, include_top=False, input_tensor=inputs
        )
        feature_maps = resnet101.get_layer('conv5_block3_out').output

        skip_connection_names = [
            'conv1_relu',
            'conv2_block3_out', 
            'conv3_block4_out', 
            'conv4_block23_out'
        ]
        skip_connections = []
        for name in skip_connection_names:
            skip_connections.append(resnet101.get_layer(name).output)
    else:
        print('Encoder not supported.')

    # Pyramid pooling -------------------------------------------------------
    spatial_size = feature_maps.shape

    # # Ensure feature maps are 1/8 in w,h as inputs to match original paper
    # upsample_size = [(input_shape[0])/8 // spatial_size[1], (input_shape[1])/8 // spatial_size[2]]
    # feature_maps = layers.UpSampling2D(upsample_size)(feature_maps)

    # For channels last input format
    transform_size = lambda level: [spatial_size[1] // level, spatial_size[2] // level]

    blocks = []
    for level in [1,2,4,8]: # levels 1,2,3,6 require custom pool factos depending on resolution, hence why these instead
        block = layers.AveragePooling2D(transform_size(level))(feature_maps)
        block = layers.Conv2D(512, 1, padding='same')(block)
        block = layers.BatchNormalization()(block)
        block = layers.Activation('relu')(block)
        block = layers.UpSampling2D(transform_size(level))(block)
        blocks.append(block)

    pooled = layers.concatenate([feature_maps, *blocks])

    x = layers.Conv2D(512, 1, padding='same')(pooled)
    x = layers.BatchNormalization()(x)    
    x = layers.Activation("relu")(x)

    # Upsample block ---------------------------------------------------------
    if use_unet_decoder:
        resnet = False
        if 'ResNet' in encoder:
            resnet = True
        outputs = unet_decoder(x, n_classes, resnet, skip_connections)
    else:
        final_upsample_factor = [input_shape[0] // x.shape[1], input_shape[1] // x.shape[2]]
        outputs = layers.UpSampling2D(final_upsample_factor)(x)

        # Network head without a decoder
        outputs = layers.Conv2D(n_classes, 3, activation='sigmoid', padding="same", dtype=tf.float32)(outputs)

    return keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    from tensorflow import keras
    model = PSPNet((256, 256, 1), 1, encoder='vgg', use_unet_decoder=True)
    model.summary()
