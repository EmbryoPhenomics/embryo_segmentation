import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
from tqdm import tqdm

# Parameters
images_in = './images/'
annotations_in = './annotations/'

ia.seed(1)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Sometimes(
        0.5,
        iaa.Crop(percent=(0, 0.1))
    ),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.OneOf([
        iaa.Multiply((0.8, 3)),
    ]),
], random_order=True) # apply augmenters in random order

images_fn = glob.glob(f'{images_in}*.png')
annotations_fn = [re.sub(images_in, annotations_in, f) for f in images_fn]

# Note this is a 80:10:10 train:val:test ratio - adjust if you would prefer a different ratio
train_samples = 0.8*len(images_fn)
val_samples = 0.1*len(images_fn)
test_samples = 0.1*len(images_fn)

counter = 0
for imf,annf in tqdm(zip(images_fn, annotations_fn), total=len(images_fn)):

    image, mask = map(cv2.imread, (imf, annf))

    image = image[:,:,0]
    mask = mask[:,:,0]

    segmap = SegmentationMapsOnImage(mask, shape=image.shape)

    if counter < train_samples:
        images_out = './train_images/'
        annotations_out = './train_annotations/'

        # Augment images and segmaps.
        for i in range(10):
            images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)

            imf_at = re.sub(images_in, images_out, imf)
            imf_at = re.sub('.png', f'_{i}.png', imf_at)
            annf_at = re.sub(annotations_in, annotations_out, annf)
            annf_at = re.sub('.png', f'_{i}.png', annf_at)

            cv2.imwrite(imf_at, images_aug_i)
            cv2.imwrite(annf_at, segmaps_aug_i.draw(images_aug_i.shape[:2])[0][:,:,0])
    else:
        if counter > train_samples and counter < train_samples + val_samples:
            images_out = './val_images/'
            annotations_out = './val_annotations/'

            imf_at = re.sub(images_in, images_out, imf)
            annf_at = re.sub(annotations_in, annotations_out, annf)

            cv2.imwrite(imf_at, image)
            cv2.imwrite(annf_at, mask)

        elif counter > train_samples + val_samples and counter < train_samples + val_samples + test_samples :
            images_out = './test_images/'
            annotations_out = './test_annotations/'

            imf_at = re.sub(images_in, images_out, imf)
            annf_at = re.sub(annotations_in, annotations_out, annf)

            cv2.imwrite(imf_at, image)
            cv2.imwrite(annf_at, mask)

    counter += 1
