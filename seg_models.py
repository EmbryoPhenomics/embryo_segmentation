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

weights = {
    'UNet': 'https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/UNet_lymnaea_binary.h5',
    'UNet2plus': 'https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/UNet2plus_lymnaea_binary.h5',
    'UNet3plus': 'https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/UNet3plus_lymnaea_binary.h5',
    'SegNet': 'https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/SegNet_lymnaea_binary.h5',
    'FCN': 'https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/FCN_lymnaea_binary.h5',
    'DeepLabV3': 'https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/PSPNet_lymnaea_binary.h',
    'HRNetV2': 'https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/DeepLabV3_lymnaea_binary.h5',
    'PSPNet': 'https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/HRNetV2_lymnaea_binary.h'
}


def build_model(input_shape, n_classes, model='UNet', pretrained_weights=False):
    seg_model = model_definitions[model](input_shape=input_shape, n_classes=n_classes)

    if pretrained_weights:
        url = weights[model]
        pretrained_weights = keras.utils.get_file(origin=url)
        seg_model.load_weights(pretrained_weights)

    return seg_model

if __name__ == '__main__':
    model = build_model(input_shape=(256, 256, 1), n_classes=1, model='UNet', pretrained_weights=True)
    model.summary()