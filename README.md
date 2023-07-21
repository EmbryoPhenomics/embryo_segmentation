# Semantic segmentation of Embryos in microscopy images

This repository includes training and inference code, as well as pre-trained weights for some early life history stages of some species. Semantic segmentation is achieved through popular encoder-decoder architectures such as UNet and DeepLab V3. Currently there are only weights available for the detection of eggs in Lymnaea stagnalis, though this could be extended to other species with suitable training data. Results of these models are shown below as well as the links to the pre-trained models:

Performance different models on Lymnaea stagnalis embryo segmentation:

| name | resolution | binary iou | #params | model |
|:---:|:---:|:---:|:---:| :---:|
| UNet | 256x256 | 96.2 | 8M | [model](https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/UNet_lymnaea_binary.h5) 
| UNet++ | 256x256 | 88.5 | 5M | [model](https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/UNet2plus_lymnaea_binary.h5)
| UNet3+ | 256x256 | 95.3 | 11M | [model](https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/UNet3plus_lymnaea_binary.h5)
| SegNet | 256x256 | 96.1 | 18M | [model](https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/SegNet_lymnaea_binary.h5)
| FCN | 256x256 | 93.9 | 33M | [model](https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/FCN_lymnaea_binary.h5)
| PSPNet | 256x256 | 95.5 | 7M | [model](https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/PSPNet_lymnaea_binary.h5)
| DeepLabV3+ | 256x256 | 95.8 | 11M | [model](https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/DeepLabV3_lymnaea_binary.h5)
| HRNetV2 | 256x256 | 95.9 | 9M | [model](https://github.com/EmbryoPhenomics/embryo_segmentation/releases/download/v0.1/HRNetV2_lymnaea_binary.h5)

### Lymnaea stagnalis image dataset

Images and annotations for training are included in this repository in the release with model weights. Note that the images used for this dataset were captured with the [OpenVim](https://github.com/otills/openvim) phenotyping platform.

