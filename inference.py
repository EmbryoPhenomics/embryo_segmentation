import tensorflow as tf
import cv2
import vuba
import matplotlib.pyplot as plt
import numpy as np

# Initiate UNet model
unet = tf.keras.models.load_model('./trained_models/UNet_lymnaea_binary.h5')

# Load in an images for inference
x1,x2 = (270, 605)
y1,y2 = (280, 560)

# Read in frames from 
video = vuba.Video('./example.avi')
frames = video.read(start=0, stop=len(video), grayscale=True)

# Perform inference
for frame in frames:
    frame = frame[y1:y2, x1:x2]
    frame = np.expand_dims(frame, axis=-1)
    frame = tf.image.resize_with_pad(frame, 256, 256)
    frame = np.expand_dims(frame, axis=0)

    # Segment and detect outlines using UNet
    result = unet.predict_on_batch(frame)

    result = result[0,:,:,0]
    r = result.copy()
    r[r > 0.6] = 1
    r[r < 1] = 0
    r = (r * 255).astype(np.uint8)

    frame = vuba.bgr(frame[0,:,:,0].astype(np.uint8))

    contours, _ = cv2.findContours(r, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours):
        contour = vuba.largest(contours)
        vuba.draw_contours(frame, contour, -1, (0,255,0), 1)    

    cv2.imshow('test', np.hstack((frame, vuba.bgr((result * 255).astype(np.uint8)))))
    cv2.waitKey()
