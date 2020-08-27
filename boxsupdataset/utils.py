import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float


def show_x_images(image, label):
    """Show images with labelimages"""
    if len(image.shape) == 2:
        imageRGB = np.dstack([image]*3)
    else:
        imageRGB = image

    # stack images
    stacked_image = np.vstack((img_as_float(imageRGB), img_as_float(label)))

    plt.imshow(stacked_image)
    plt.pause(0.001)
