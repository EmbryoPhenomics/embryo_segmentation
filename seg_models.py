from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from models import UNet, UNet2plus, UNet3plus, SegNet, FCN, PSPNet, DeepLabV3, HRNetV2

model_definitions = {
    'UNet': UNet,
    'UNet2plus': UNet2plus,
    'UNet3plus': UNet3plus,
    'SegNet': SegNet,
    'FCN': FCN,
    'DeepLabV3': DeepLabV3,
    'HRNetV2': HRNetV2,
    'PSPNet': PSPNet
}

def build_model(input_shape, n_classes, model='UNet', pretrained_weights=False):
    seg_model = model_definitions[model](input_shape=input_shape, n_classes=n_classes)

    if pretrained_weights:
        url = weights[name]
        pretrained_weights = keras.utils.get_file(origin=url)
        seg_model.load_weights(pretrained_weights)

    return seg_model

if __name__ == '__main__':
    model = build_model(input_shape=(256, 256, 1), n_classes=1, model='UNet3plus')
    model.summary()