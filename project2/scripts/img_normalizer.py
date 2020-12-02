import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import sys

from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import expand_dims

def resize_images(img_dir, x, y):
    img_dir = os.path.join(os.getcwd(), img_dir)
    images = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    for i in tqdm(images):
        img_path = os.path.join(img_dir, i)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (y, x))
        cv2.imwrite(img_path, img)

def keras_augmentation(img_dir):
    img_dir = os.path.join(os.getcwd(), img_dir)
    images = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    for i in tqdm(images):
        img = load_img(os.path.join(img_dir, i))
        img = img_to_array(img)
        img = expand_dims(img, axis=0)

        aug = ImageDataGenerator(
            rotation_range=0,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest')
        new_img_dir = os.path.join(os.path.join(os.getcwd(), 'output/sega/'), 'augmented/')
        name = i.split('_')
        name = name[1].split('.png')
        imgGen = aug.flow(img, batch_size=1, save_to_dir=new_img_dir, save_prefix='aug_'+name[0], save_format='png')
        total = 0
        for image in imgGen:
            total += 1
            if total == 9:
                total = 0
                break


def cropBorder(img, left, right, up, down, crop_path, img_name):
    """ Crop image with the given values.

    Args:
        img: openCV image
        left: starting pixel on left side.
        right: starting pixel on right side.
        up: starting pixel on up side.
        down: starting pixel on down side.
        crop_path: path of directory where the cropped image is saved.
        img_name: name used for the cropped image.

    Returns:
        No value returned. Images are saved in a sub-folder "cropped/" in the same folder of the images.
    """

    crop = img[up:down, left:right]
    cv2.imwrite(crop_path+img_name, crop)

def getXMedianValue(img, y, x, w):
    """ Get the median value of pixels on the X axis.

    Args:
        img: openCV image
        y: Coordinate Y of the center pixel.
        x: Coordinate X of the center pixel.
        w: Width of the array to check.

    Returns:
        Median of pixel values between x-(w/2) and x+(w/2).
    """
    _, cols, _ = img.shape
    sub_val = []
    if x < int(w/2):
        for i in range(int(w/2)-x):
            sub_val.append(1)
        start = x-int(w/2)+len(sub_val)
    else:
        start = x-int(w/2)
    if x+int(w/2) > cols-1:
        end = cols
    else:
        end = x+int(w/2)+1
    for x_id in range(start, end):
        sub_val.append(sum(img[y,x_id]))
    while len(sub_val) < w:
        sub_val.append(1)
    return np.median(sub_val)

def getYMedianValue(img, y, x, w):
    """ Get the median value of pixels on the Y axis.

    Args:
        img: openCV image
        y: Coordinate Y of the center pixel.
        x: Coordinate X of the center pixel.
        w: Width of the array to check.

    Returns:
        Median of pixel values between y-(w/2) and y+(w/2).
    """
    rows, _, _ = img.shape
    sub_val = []
    if y < int(w/2):
        for i in range(int(w/2)-x):
            sub_val.append(1)
        start = y-int(w/2)+len(sub_val)
    else:
        start = y-int(w/2)
    if y+int(w/2) > rows-1:
        end = rows
    else:
        end = y+int(w/2)+1
    for y_id in range(start, end):
        sub_val.append(sum(img[y_id,x]))
    while len(sub_val) < w:
        sub_val.append(1)
    return np.median(sub_val)

def getFrameValues(img, blacks):
    """ Use the Pillarbox effect to get the pixel coordinates used to crop the image.

    Args:
        img: openCV image
        blacks: X coordinates of the black pillars due to the Pillarbox effect.

    Returns:
        left, right, up, down pixel coordinates used to crop the image.
    """
    rows, cols, _ = img.shape
    up = []
    down = []
    # Use computed black pillars for left and right coordinates.
    left = min(blacks)
    right = max(blacks)
    for x in blacks:
        for y in range(rows):
            m = getYMedianValue(img, y, x, 11)
            if (m == 0):
                up.append(y-1)
                break
    for x in blacks:
        for y in range(rows-1, 0, -1):
            m = getYMedianValue(img, y, x, 11)
            if (m == 0):
                down.append(y+1)
                break
    return left, right, int(np.median(up)), int(np.median(down))

def getBlackBorders(img):
    """ Compute X coordinates of black pillars generated by Pillarbox effect.

    Args:
        img: openCV image

    Returns:
        Array of all X coordinates of the black pillars generated by the Pillarbox effect.
    """
    rows, cols, _ = img.shape
    blacks = {}
    for y in tqdm(range(rows)):
        for x in range(cols):
            m = getXMedianValue(img, y, x, 11)
            if (m == 0):
                if x in blacks:
                    blacks[x] = blacks[x]+1
                else:
                    blacks[x] = 1
    res = []
    for k in blacks:
        if blacks[k] > int(rows/2):
            res.append(k)
    # Sort list of X coordinates
    res.sort()
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract images from videos')
    parser.add_argument('-ifolder', type=str, required=True, help='Folder containing images to process')
    parser.add_argument('-b', '--blackpillars', type=bool, required=False, default=True, help='Cropped additional borders using black pillars. (default: True)')
    parser.add_argument('-r', '--resize', type=bool, required=False, default=True, help='Resize image with -x and -y values. (default: True)')
    parser.add_argument('-f', '--iformat', type=str, required=False, default='png', help='Format of generated images. (default: png)')
    parser.add_argument('-g', '--goodbatch', type=bool, required=False, default=True, help='All images have same coordinates for black pillars. If True, will compute the black pillar coordinates only for the first image. (default: True)')
    parser.add_argument('-x', '--x_resize', type=int, required=False, default=224, help='Resize X value of images. (default: 224)')
    parser.add_argument('-y', '--y_resize', type=, required=False, default=320, help='Resize Y value of images. (default: 320)')

    args = parser.parse_args(sys.argv[1:])

    resize = args.resize
    blackpillars = args.blackpillars

    cwd_path = os.path.join(os.getcwd(), args.ifolder)
    img_format = args.iformat
    good_batch = args.goodbatch

    x_val = args.x_resize
    y_val = args.y_resize

    images = [f for f in os.listdir(cwd_path) if os.path.isfile(os.path.join(cwd_path, f)) and f.find(img_format) != -1]

    crop_dir = cwd_path
    # If directory doesn't exist, create it
    if (os.path.isdir(crop_dir) == False):
        os.mkdir(crop_dir, 0o777)

    blacks = None
    left = 0
    right = 0
    up = 0
    down = 0
    if blackpillars == True:
        if good_batch == True:
            for i in tqdm(images):
                img = cv2.imread(os.path.join(cwd_path, i))
                if blacks == None:
                    blacks = getBlackBorders(img)
                    left, right, up, down = getFrameValues(img, blacks)
                cropBorder(img, left, right, up, down, crop_dir, i)
        else:
            for i in images:
                img = cv2.imread(os.path.join(cwd_path, i))
                blacks = getBlackBorders(img)
                left, right, up, down = getFrameValues(img, blacks)
                cropBorder(img, left, right, up, down, crop_dir, i)
    if resize == True:
        resize_images(cwd_path, x_val, y_val)
