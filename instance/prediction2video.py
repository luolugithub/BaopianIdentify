"""
-*- coding: utf-8 -*-
@Author  : LuoLu
@Time    : 2020-11-03 14:29
@File: prediction2video.py
@Github    : https://github.com/luolugithub
@Email    : argluolu@gmail.com
"""
import os
import sys
import random
import math
import time
import traceback

import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
from skimage.measure import find_contours
from matplotlib import patches, lines
from matplotlib.patches import Polygon
import IPython.display
import cv2
import matplotlib
from PIL import Image
from instance.mrcnn.visualize import random_colors, apply_mask
import tensorflow as tf
matplotlib.use("TkAgg")
# Root directory of the project
from instance import coco


def pad_images_to_same_size(img2pad):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    # for img in images:
    h, w = img2pad.shape[:2]
    # print("img2pad:", img2pad.shape[:2])
    desired_size = max(h, w)

    delta_w = desired_size - w
    delta_h = desired_size - h
    top, bottom = (int)(delta_h / 2), (int)(delta_h - (delta_h / 2))
    left, right = (int)(delta_w / 2), (int)(delta_w - (delta_w / 2))

    color = [0, 0, 0]
    img_padded = cv2.copyMakeBorder(img2pad, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    # print("img_padded:", img_padded.shape[:2])

    return img_padded







ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from instance.mrcnn import utils
import instance.mrcnn.model as modellib
from instance.mrcnn import visualize


np.set_printoptions(threshold=np.inf)

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "../"))  # To find local version
# import coco

# %matplotlib inline

# Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "checkpoint")


# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = "/home/xkjs/PycharmProjects/instance2video" \
                  "/checkpoint/" \
                  "mask_rcnn_coco_0044.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = "/home/xkjs/Downloads/data/BoPian/" \
            "dataset_ning71x-16-0672/7img/" \
            "c1m116.jpg"
image_file = IMAGE_DIR

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
print("config:\n", config)
# config.display()

# 3
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# 4 Class Names
class_names = ['BG', '_background_', 'keli']

# 5 Run Object Detection

# Load a random image from the images folder
image = skimage.io.imread(image_file)
img = cv2.imread(image_file)



# print("file_names:\n", file_names)
print("image_file:\n", image_file)
name_img_file = image_file.split("/")[-1]
name_root = image_file.split(".")[-2]
print("name_img_file:\n", name_img_file)
print("name_root:\n", name_root)
# zhengjiaopianguang img path
z0_img_path = name_root + "_0.jpg"
z15_img_path = name_root + "_15.jpg"
z30_img_path = name_root + "_30.jpg"
z45_img_path = name_root + "_45.jpg"
z60_img_path = name_root + "_60.jpg"
z75_img_path = name_root + "_75.jpg"

z0_img = cv2.imread(z0_img_path)
z15_img = cv2.imread(z15_img_path)
z30_img = cv2.imread(z30_img_path)
z45_img = cv2.imread(z45_img_path)
z60_img = cv2.imread(z60_img_path)
z75_img = cv2.imread(z75_img_path)

videoPathOut = "/home/xkjs/Downloads/data/BoPian/" \
               "dataset_ning71x-16-0672/video/"

# create video param
fps = 1


# print("z45_img:\n", z45_img_path)

# Run detection
results = model.detect([image], verbose=2)

# Visualize results
r = results[0]


# Number of instances
boxes = r['rois']
masks = r['masks']
class_ids = r['class_ids']
scores = r['scores']
class_names = class_names
N = boxes.shape[0]
ax = None
colors = None
captions = None
show_bbox = True
show_mask = True
title = ""

if not N:
    print("\n*** No instances to display *** \n")
else:
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

# If no axis is passed, create one and automatically call show()
auto_show = False
height, width = image.shape[:2]
print("image.shape: " + str(height) + "X" + str(width))
figsize = (width / 100, height / 100)
if not ax:
    _, ax = plt.subplots(1, figsize=figsize)
    auto_show = True

# Generate random colors
colors = colors or random_colors(N)

# Show area outside image boundaries.
height, width = image.shape[:2]
ax.set_ylim(height + 10, -10)
ax.set_xlim(-10, width + 10)
ax.axis('off')
# ax.set_title(title)
area_total = height*width
print("area total:", area_total)
masked_image = image.astype(np.uint32).copy()

# cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
#                     "/dataset_ning71x-16-0672/crop/cv_" + name_img_file
#                     , img)
# cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
#                     "/dataset_ning71x-16-0672/crop/src_" + name_img_file
#                     , image)


# save mask
# mask_temp = np.zeros([height, width, 3], dtype=np.uint32)
# mask_temp = mask_temp.astype(np.uint32).copy()

# with tf.device('/cpu:0'):

for i in range(N):
    print("keli: " + str(i))

    color = colors[i]

    # Bounding box
    if not np.any(boxes[i]):
        # Skip this instance. Has no bbox. Likely lost in image cropping.
        continue
    y1, x1, y2, x2 = boxes[i]
    if show_bbox:
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

    # Label
    if not captions:
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        caption = "{} {:.3f}".format(label, score) if score else label
    else:
        caption = captions[i]
    #     label and score text
    ax.text(x1, y1 + 8, caption,
            color='w', size=11, backgroundcolor="none")

    # Mask
    mask = masks[:, :, i]
    print("type mask", type(mask))


    # print("size mask", mask.shape)
    # print("mask:\n", mask)
    mask_temp_area = np.zeros([height, width, 3], dtype=np.uint32)
    mask_temp = mask_temp_area.astype(np.uint32).copy()
    if show_mask:
        masked_image = apply_mask(masked_image, mask, color)
        mask_temp = apply_mask(mask_temp.astype(np.uint8), mask, color)

        gray_mask_temp = cv2.cvtColor(mask_temp, cv2.COLOR_BGR2GRAY)
        area = cv2.countNonZero(gray_mask_temp)
        # print("type gray_mask_temp", type(gray_mask_temp))
        print("area gray_mask_temp", area)
        # cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
        #             "/dataset_ning71x-16-0672/crop/gray_" + name_img_file
        #             , gray_mask_temp)
        # print("gray_mask_temp:\n", gray_mask_temp)

    #
    # # Mask Polygon
    # # Pad to ensure proper polygons for masks that touch image edges.
    # padded_mask = np.zeros(
    #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    # padded_mask[1:-1, 1:-1] = mask
    # contours = find_contours(padded_mask, 0.1)
    #
    # #
    # frame_array = []
    # size = ()
    # points = []
    # verts = []
    #
    # for verts in contours:
    #     # Subtract the padding and flip (y, x) to (x, y)
    #     verts = np.fliplr(verts) - 1
    #     p = Polygon(verts, facecolor="none", edgecolor=color)
    #     ax.add_patch(p)
    #
    # # crop
    # points = verts
    # points[points < 0] = 0
    # # print("points:\n", points)
    # points = np.around(points)
    # points = points.astype(int)
    #
    # # print("int points:\n", points)
    #
    # # # (1) Crop the bounding rect
    # rect = cv2.boundingRect(points)
    # x, y, w, h = rect
    # croped = img[y:y + h, x:x + w].copy()
    #
    # croped_0 = z0_img[y:y + h, x:x + w].copy()
    # croped_15 = z15_img[y:y + h, x:x + w].copy()
    # croped_30 = z30_img[y:y + h, x:x + w].copy()
    # croped_45 = z45_img[y:y + h, x:x + w].copy()
    # croped_60 = z60_img[y:y + h, x:x + w].copy()
    # croped_75 = z75_img[y:y + h, x:x + w].copy()
    #
    #
    #
    # # # (2) make mask
    # points = points - points.min(axis=0)
    #
    # mask = np.zeros(croped.shape[:2], np.uint8)
    # # for zhengjiao
    # mask_0 = np.zeros(croped_0.shape[:2], np.uint8)
    # mask_15 = np.zeros(croped_15.shape[:2], np.uint8)
    # mask_30 = np.zeros(croped_30.shape[:2], np.uint8)
    # mask_45 = np.zeros(croped_45.shape[:2], np.uint8)
    # mask_60 = np.zeros(croped_60.shape[:2], np.uint8)
    # mask_75 = np.zeros(croped_75.shape[:2], np.uint8)
    #
    #
    # cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # cv2.drawContours(mask_0, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # cv2.drawContours(mask_15, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # cv2.drawContours(mask_30, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # cv2.drawContours(mask_45, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # cv2.drawContours(mask_60, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # cv2.drawContours(mask_75, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    #
    # # # (3) do bit-op
    # dst = cv2.bitwise_and(croped, croped, mask=mask)
    #
    # dst_0 = cv2.bitwise_and(croped_0, croped_0, mask=mask_0)
    # dst_15 = cv2.bitwise_and(croped_15, croped_15, mask=mask_15)
    # dst_30 = cv2.bitwise_and(croped_30, croped_30, mask=mask_30)
    # dst_45 = cv2.bitwise_and(croped_45, croped_45, mask=mask_45)
    # dst_60 = cv2.bitwise_and(croped_60, croped_60, mask=mask_60)
    # dst_75 = cv2.bitwise_and(croped_75, croped_75, mask=mask_75)
    #
    # # pad odd 2 even
    # # new image H, W
    # height, width, layers = dst.shape
    # # print("crop dst size: ", dst.shape[:2])
    # new_height = 0
    # new_width = 0
    #
    # # pad=ceil(iw/2)*2:ceil(ih/2)*2
    #
    # if (height % 2) == 0:
    #     new_height = height
    # else:
    #     new_height = math.ceil(height / 2) * 2
    # if (width % 2) == 0:
    #     new_width = width
    # else:
    #     new_width = math.ceil(width / 2) * 2
    # print(new_width, new_height)
    # top, bottom = new_height, 0
    # left, right = new_width, 0
    # color = [0, 0, 0]
    # newsize = (new_width, new_height)
    # # resize image
    # dst = cv2.resize(dst, newsize)
    # dst_0 = cv2.resize(dst_0, newsize)
    # dst_15 = cv2.resize(dst_15, newsize)
    # dst_30 = cv2.resize(dst_30, newsize)
    # dst_45 = cv2.resize(dst_45, newsize)
    # dst_60 = cv2.resize(dst_60, newsize)
    # dst_75 = cv2.resize(dst_75, newsize)
    #
    # # pad img
    # dst = pad_images_to_same_size(dst)
    # dst_0 = pad_images_to_same_size(dst_0)
    # dst_15 = pad_images_to_same_size(dst_15)
    # dst_30 = pad_images_to_same_size(dst_30)
    # dst_45 = pad_images_to_same_size(dst_45)
    # dst_60 = pad_images_to_same_size(dst_60)
    # dst_75 = pad_images_to_same_size(dst_75)
    #
    #
    #
    # # img for video
    # # use size in video
    #
    # print("dst_30 shape: ", dst_30.shape)
    # print("dst_30 type: ", type(dst_30))
    # size = dst.shape[:2]
    # print("dst size: ", dst.shape[:2])
    #
    # frame_array.append(dst)
    # frame_array.append(dst_0)
    # frame_array.append(dst_15)
    # frame_array.append(dst_30)
    # frame_array.append(dst_45)
    # frame_array.append(dst_60)
    # frame_array.append(dst_75)

    # # save video
    # video_path_out = videoPathOut + "k_" + str(i) + ".avi"
    # out = cv2.VideoWriter(None,
    #                       cv2.VideoWriter_fourcc(*'DIVX'),
    #                       fps,
    #                       size)
    # print("frame len: ", len(frame_array))
    # for i in range(len(frame_array)):
    #     # writing to a image array
    #     # print("frame size: ", frame_array[i].shape)
    #     out.write(np.uint8(frame_array[i]))
    # out.release()

    # try:
    #     time.sleep(0.5)
    # except Exception:
    #     traceback.print_exc()


        # # (4) add the white background
        # bg = np.ones_like(croped, np.uint8) * 255
        # cv2.bitwise_not(bg, bg, mask=mask)
        # dst2 = bg + dst

        # save crop img
        # cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
        #             "/dataset_ning71x-16-0672/crop/" + name_img_file
        #             + "_k" +
        #             str(i), dst)

    # save crop video
    # cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
    #             "/dataset_ning71x-16-0672/crop/"
    #             + "k_" +
    #             str(i)
    #             + "_s.jpg", dst)
    # cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
    #             "/dataset_ning71x-16-0672/crop/"
    #             + "k_" +
    #             str(i)
    #             + "_0.jpg", dst_0)
    # cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
    #             "/dataset_ning71x-16-0672/crop/"
    #             + "k_" +
    #             str(i)
    #             + "_15.jpg", dst_15)
    # cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
    #             "/dataset_ning71x-16-0672/crop/"
    #             + "k_" +
    #             str(i)
    #             + "_30.jpg", dst_30)
    # cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
    #             "/dataset_ning71x-16-0672/crop/"
    #             + "k_" +
    #             str(i)
    #             + "_45.jpg", dst_45)
    # cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
    #             "/dataset_ning71x-16-0672/crop/"
    #             + "k_" +
    #             str(i)
    #             + "_60.jpg", dst_60)
    # cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
    #             "/dataset_ning71x-16-0672/crop/"
    #             + "k_" +
    #             str(i)
    #             + "_75.jpg", dst_75)




# print("shape masked image:", masked_image.astype(np.uint8).shape)
# ax.imshow(masked_image.astype(np.uint8))

# save pure mask
# cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
#             "/dataset_ning71x-16-0672/predictions/mask_" + name_img_file
#             , mask_temp.astype(np.uint8))



# plt.savefig("/home/xkjs/Downloads/data/BoPian"
#             "/dataset_ning71x-16-0672/predictions/f_" + name_img_file
#             )

# if auto_show:
#     plt.show()
#     print("out type:", type(out))
