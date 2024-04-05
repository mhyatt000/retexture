# from google.colab import drive
from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt

from itertools import product

import cv2
import numpy as np

# drive.mount("/content/drive")


cn = "/content/drive/MyDrive/Luke Baumel Carbon Fellowship/Official Versions/2D Mapped/"
categories = [
    "bathtub",
    "bear",
    "bird",
    "butterfly",
    "elephant",
    "fish",
    "helmet",
    "mailbox",
    "mug",
    "phone",
]
angles = [30 * i for i in range(12)]
angles = [f"{30*i}.0" for i in range(12)]


def map_2d(shape, tex, shape_prefix, tex_prefix, out_dir):
    shape_paths = f"{shape_prefix}{shape}.png"
    tex_paths = f"{tex_prefix}{tex}.jpg"

    # redefine shape tex
    shape_im = cv2.imread(shape_path)
    tex_im = cv2.imread(texture_path)

    shape_bn = cv2.cvtColor(
        shape_im, cv2.COLOR_BGR2GRAY
    )  # convert shape to a binary image
    shape_bn = cv2.threshold(shape_im, 20, 255, cv2.THRESH_BINARY)[
        1
    ]  # set pix > 20 to white else black

    y, x, c = shape_bn.shape
    tex_im = cv2.resize(tex_im, (x, y))

    original = shape_bn == 255
    tex_im[original] = [255, 255, 255]

    cv2.imwrite(f"{outdir}{shape}_{texture}.jpg", tex)


def image_histogram(image):
    flattened_image = image.flatten()
    histogram = {value: np.sum(flattened_image == value) for value in np.unique(flattened_image)}
    return histogram


def main():

    original = "/Users/matthewhyatt/Downloads/original.png" 
    original = cv2.imread(original)
    tex = "/Users/matthewhyatt/Downloads/tex.png"
    tex = cv2.imread(tex)
    tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
    tex = cv2.resize(tex, (original.shape[1], original.shape[0]))
    thresh = 20
    original = original
    original[original > thresh] = 255
    original[original <= thresh] = tex[original <= thresh]

    Image.fromarray(original).save('output.png', quality=100)  
    quit()

    print(original.shape)

    quit()

    original = np.all(original == 0, axis=-1)
    new = np.copy(original)
    new[original] = tex[original]

    plt.imshow(new)
    plt.show()

    hist = image_histogram(original)
    print(hist)
    quit()

    # make a new image where pixels are 0 if they were 206
    new = original[original == 206] 
    import time
    for row in original:
        print(list(row))
        time.sleep(1)
    print(list(new))
    quit()
    print(original)
    plt.imshow(original)
    plt.show()

    quit()
    shape_prefix = input("input shape prefix: ")
    tex_prefix = input("input tex prefix: ")
    out_dir = input("input output directory: ")

    shape_paths = [
        f"{shape}{num}_black_{angle}"
        for shape, num, angle in product(categories, range(1, 11), angles)
    ]
    tex_paths = [
        f"{shape}{num}"
        for shape, num, angle in product(categories, range(1, 11), angles)
    ]

    pairs = product(shape_paths, tex_paths)
    for shape_path, texture_path in pairs:
        map_2d(shape_path, texture_path, shape_prefix, tex_prefix, out_dir)


if __name__ == "__main__":
    main()
