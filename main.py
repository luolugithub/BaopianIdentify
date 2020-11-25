"""
-*- coding: utf-8 -*-
@Author  : LuoLu
@Time    : 2020-11-05 10:20
@File:      main.py
@Github    : https://github.com/luolugithub
@Email    : argluolu@gmail.com
"""
import math
from heapq import nlargest

import matplotlib
import matplotlib.pyplot as plt
import skimage.io
from matplotlib import patches
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from instance.mrcnn.visualize import random_colors, apply_mask

matplotlib.use("TkAgg")
from instance import coco
# video import
import sys
import torchvision.models as torch_models
from video.utils import *
from video.models import *
from torchvision import transforms

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def pad_images_to_same_size(img2pad):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    # for img in images:
    h, w = img2pad.shape[:2]
    desired_size = max(h, w)

    delta_w = desired_size - w
    delta_h = desired_size - h
    top, bottom = (int)(delta_h / 2), (int)(delta_h - (delta_h / 2))
    left, right = (int)(delta_w / 2), (int)(delta_w - (delta_w / 2))

    color = [0, 0, 0]
    img_padded = cv2.copyMakeBorder(img2pad, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
    return img_padded


def predict_video(video_model, video, device):
    video_model.eval()
    inputs, _ = collate_fn([[video, torch.tensor([0])]])
    videos = inputs.to(device)

    with torch.no_grad():
        pred_labels = video_model(videos).cpu()
    # add reliable info
    smax = nn.Softmax(1)
    smax_out = smax(pred_labels)
    max_index = list(map(list(smax_out[0]).index, nlargest(3, smax_out[0])))
    reliable = nlargest(3, smax_out[0].cpu().numpy())
    prediction = pred_labels.numpy().argmax()
    return prediction, reliable


def main(model_type,
         cnn_model,
         frames_cnt,
         bilstm,
         model_path,
         video_path,
         class_dict):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
    device = torch.device("cuda:0")
    cnn_model = torch_models.resnet34(pretrained=True) if cnn_model == "resnet34" else torch_models.resnet18(
        pretrained=True)
    if model_type == "cnn-rnn":
        model = CNNtoRNNModel(cnn_model, frames_cnt=frames_cnt, bidirectional=bilstm)
    else:
        model = AvgCNNModel(cnn_model, frames_cnt=frames_cnt)
    with open(model_path, "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)
    model.to(device)
    set_frames_cnt(frames_cnt)

    video = read_video(video_path, train_transforms)

    prediction = predict_video(model, video, device)
    print("predict result:", class_dict[prediction + 1])


# predict video para
model_type = "cnn-avg"
cnn_model = "resnet34"
frames_cnt = 7
bilstm = True
model_path = "/home/xkjs/PycharmProjects/instance2video/" \
             "checkpoint/" \
             "baseline_best.pth"

class_dict = {}
with open("/home/xkjs/PycharmProjects/"
          "instance2video/"
          "video/classInd.txt") as file:
    for line in file:
        (key, val) = line.split()
        class_dict[int(key)] = val

device = torch.device("cuda:0")
cnn_model = torch_models.resnet34(pretrained=True) if cnn_model == "resnet34" else torch_models.resnet18(
    pretrained=True)
if model_type == "cnn-rnn":
    video_model = CNNtoRNNModel(cnn_model, frames_cnt=frames_cnt, bidirectional=bilstm)
else:
    video_model = AvgCNNModel(cnn_model, frames_cnt=frames_cnt)
with open(model_path, "rb") as fp:
    best_state_dict = torch.load(fp, map_location="cpu")
    video_model.load_state_dict(best_state_dict)
video_model.to(device)
set_frames_cnt(frames_cnt)

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transforms = train_transforms
in_memory = False

ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from instance.mrcnn import utils
import instance.mrcnn.model as modellib

np.set_printoptions(threshold=np.inf)

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "../"))  # To find local version
# Directory MODEL_DIR
MODEL_DIR = os.path.join(ROOT_DIR, "checkpoint")

# Local path to trained weights file
COCO_MODEL_PATH = "/home/xkjs/PycharmProjects/instance2video" \
                  "/checkpoint/" \
                  "int_best.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = "/home/xkjs/Downloads/data/BoPian/" \
            "dataset_ning71x-16-0672/7img/" \
            "ning71x-16-0672_c1m116_s.jpg"
image_file = IMAGE_DIR


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
print("config:", config)
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

# print("file_names:", file_names)
print("image_file:", image_file)
name_img_file = image_file.split("/")[-1]
name_root = image_file.split("_s.")[0]
print("name_img_file:", name_img_file)
print("name_root:", name_root)
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

# for vision
img_v = img
img_v0 = z0_img
img_v15 = z15_img
img_v30 = z30_img
img_v45 = z45_img
img_v60 = z60_img
img_v75 = z75_img

videoPathOut = "/home/xkjs/Downloads/data/BoPian/" \
               "dataset_ning71x-16-0672/video/"

# create video param
fps = 1

# print("z45_img:", z45_img_path)

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
    print("*** No instances to display *** ")
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
HEIGHT, WIDTH = image.shape[:2]
ax.set_ylim(height + 10, -10)
ax.set_xlim(-10, width + 10)
ax.axis('off')
# ax.set_title(title)

masked_image = image.astype(np.uint32).copy()
# create empty final mask
mask_img = np.zeros([height, width, 3], dtype=np.uint32)
mask_img = mask_img.astype(np.uint32).copy()
minera_dist_mask_img = mask_img


def img_size_odd2even(image):
    # pad odd 2 even
    # new image H, W
    height, width, layers = image.shape
    # print("crop dst size: ", dst.shape[:2])
    if (height % 2) == 0:
        new_height = height
    else:
        new_height = math.ceil(height / 2) * 2
    if (width % 2) == 0:
        new_width = width
    else:
        new_width = math.ceil(width / 2) * 2
    # print(new_width, new_height)
    top, bottom = new_height, 0
    left, right = new_width, 0
    color = [0, 0, 0]
    newsize = (new_width, new_height)
    return newsize


if __name__ == '__main__':
    # area stats
    area_total = HEIGHT * WIDTH

    area_shiying = 0
    area_changshi = 0
    area_jiadasy = 0
    area_bianzhiyan = 0
    area_suanxingpenchuyan = 0
    area_fangjieshi = 0
    area_yunmu = 0
    area_chenjiyan = 0
    area_baiyunshi = 0
    area_zishengliantukuangwu = 0
    # color_dict of mineral
    color_dict = {'shiying': (244, 5, 1),
                  'changshi': (10, 5, 255),
                  'jiadasy': (0, 153, 0),
                  'bianzhiyan': (204, 102, 153),
                  'suanxingpenchuyan': (255, 255, 102),
                  'fangjieshi': (102, 204, 255),
                  'yunmu': (0, 153, 255),
                  'chenjiyan': (194, 245, 51),
                  'baiyunshi': (255, 0, 102),
                  'zishengliantukuangwu': (51, 153, 102)}
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
        cv_mask = mask
        # for compute area
        mask_temp_area = np.zeros([HEIGHT, WIDTH, 3], dtype=np.uint32)
        mask_temp = mask_temp_area.astype(np.uint32).copy()
        area = 0
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)
            mask_temp = apply_mask(mask_temp.astype(np.uint8), mask, color)

            # get for final mask
            mask_img = apply_mask(mask_img.astype(np.uint8), mask, color)

        gray_mask_temp = cv2.cvtColor(mask_temp, cv2.COLOR_BGR2GRAY)
        contours_cv, hierarchy = cv2.findContours(gray_mask_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = cv2.countNonZero(gray_mask_temp)
        print("area:", area)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.1)

        #
        frame_array = []
        size = ()
        points = []
        verts = []

        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

        # crop
        points = verts
        points[points < 0] = 0
        # print("points:", points)
        points = np.around(points)
        points = points.astype(int)

        # print("int points:", points)

        # # (1) Crop the bounding rect
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        croped = img[y:y + h, x:x + w].copy()

        croped_0 = z0_img[y:y + h, x:x + w].copy()
        croped_15 = z15_img[y:y + h, x:x + w].copy()
        croped_30 = z30_img[y:y + h, x:x + w].copy()
        croped_45 = z45_img[y:y + h, x:x + w].copy()
        croped_60 = z60_img[y:y + h, x:x + w].copy()
        croped_75 = z75_img[y:y + h, x:x + w].copy()

        # # (2) make mask
        points = points - points.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        # for zhengjiao
        mask_0 = np.zeros(croped_0.shape[:2], np.uint8)
        mask_15 = np.zeros(croped_15.shape[:2], np.uint8)
        mask_30 = np.zeros(croped_30.shape[:2], np.uint8)
        mask_45 = np.zeros(croped_45.shape[:2], np.uint8)
        mask_60 = np.zeros(croped_60.shape[:2], np.uint8)
        mask_75 = np.zeros(croped_75.shape[:2], np.uint8)

        cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.drawContours(mask_0, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.drawContours(mask_15, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.drawContours(mask_30, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.drawContours(mask_45, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.drawContours(mask_60, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.drawContours(mask_75, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

        # # (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)

        dst_0 = cv2.bitwise_and(croped_0, croped_0, mask=mask_0)
        dst_15 = cv2.bitwise_and(croped_15, croped_15, mask=mask_15)
        dst_30 = cv2.bitwise_and(croped_30, croped_30, mask=mask_30)
        dst_45 = cv2.bitwise_and(croped_45, croped_45, mask=mask_45)
        dst_60 = cv2.bitwise_and(croped_60, croped_60, mask=mask_60)
        dst_75 = cv2.bitwise_and(croped_75, croped_75, mask=mask_75)

        # pad odd 2 even
        # new image H, W
        height, width, layers = dst.shape
        # print("crop dst size: ", dst.shape[:2])
        new_height = 0
        new_width = 0

        # pad=ceil(iw/2)*2:ceil(ih/2)*2

        if (height % 2) == 0:
            new_height = height
        else:
            new_height = math.ceil(height / 2) * 2
        if (width % 2) == 0:
            new_width = width
        else:
            new_width = math.ceil(width / 2) * 2
        # print(new_width, new_height)
        top, bottom = new_height, 0
        left, right = new_width, 0
        color = [0, 0, 0]
        newsize = (new_width, new_height)
        # resize image
        dst = cv2.resize(dst, newsize)
        dst_0 = cv2.resize(dst_0, newsize)
        dst_15 = cv2.resize(dst_15, newsize)
        dst_30 = cv2.resize(dst_30, newsize)
        dst_45 = cv2.resize(dst_45, newsize)
        dst_60 = cv2.resize(dst_60, newsize)
        dst_75 = cv2.resize(dst_75, newsize)

        # pad img
        dst = pad_images_to_same_size(dst)
        dst_0 = pad_images_to_same_size(dst_0)
        dst_15 = pad_images_to_same_size(dst_15)
        dst_30 = pad_images_to_same_size(dst_30)
        dst_45 = pad_images_to_same_size(dst_45)
        dst_60 = pad_images_to_same_size(dst_60)
        dst_75 = pad_images_to_same_size(dst_75)

        # img for video
        # use size in video

        # print("dst_30 shape: ", dst_30.shape)
        # print("dst_30 type: ", type(dst_30))
        size = dst.shape[:2]
        # print("dst size: ", dst.shape[:2])

        dst = transforms(dst)
        frame_array.append(dst)
        dst_0 = transforms(dst_0)
        frame_array.append(dst_0)
        dst_15 = transforms(dst_15)
        frame_array.append(dst_15)
        dst_30 = transforms(dst_30)
        frame_array.append(dst_30)
        dst_45 = transforms(dst_45)
        frame_array.append(dst_45)
        dst_60 = transforms(dst_60)
        frame_array.append(dst_60)
        dst_75 = transforms(dst_75)
        frame_array.append(dst_75)

        if in_memory:
            samples = list(zip(range(len(frame_array)), frame_array))
            frame_array = random.sample(samples, k=frames_cnt)
            frame_array.sort(key=lambda x: x[0])
            frame_array = [image for _, image in frame_array]

        # prediction for frame array
        # print("type frame_array:", type(frame_array))
        # print("len frame_array:", len(frame_array))
        v_prediction, reliable = predict_video(video_model, frame_array, device)
        print("predict result:", class_dict[v_prediction + 1])

        # save visual img to local
        # draw box
        cv2.rectangle(img=img,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      lineType=cv2.LINE_4,
                      color=color_dict[class_dict[v_prediction + 1]],
                      thickness=1)
        cv2.rectangle(img=z0_img,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      lineType=cv2.LINE_4,
                      color=color_dict[class_dict[v_prediction + 1]],
                      thickness=1)
        cv2.rectangle(img=z15_img,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      lineType=cv2.LINE_4,
                      color=color_dict[class_dict[v_prediction + 1]],
                      thickness=1)
        cv2.rectangle(img=z30_img,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      lineType=cv2.LINE_4,
                      color=color_dict[class_dict[v_prediction + 1]],
                      thickness=1)
        cv2.rectangle(img=z45_img,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      lineType=cv2.LINE_4,
                      color=color_dict[class_dict[v_prediction + 1]],
                      thickness=1)
        cv2.rectangle(img=z60_img,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      lineType=cv2.LINE_4,
                      color=color_dict[class_dict[v_prediction + 1]],
                      thickness=1)
        cv2.rectangle(img=z75_img,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      lineType=cv2.LINE_4,
                      color=color_dict[class_dict[v_prediction + 1]],
                      thickness=1)

        # ready for Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_dict[v_prediction + 1]
            # replace score to reliable
            caption = "{} {:.3f}".format(label, reliable[0]) if score else label
        else:
            caption = captions[i]
        # draw Contours for each mineral
        cv2.drawContours(image=img,
                         contours=contours_cv,
                         contourIdx=-1,
                         lineType=0,
                         thickness=4,
                         color=color_dict[class_dict[v_prediction + 1]])
        cv2.drawContours(image=z0_img,
                         contours=contours_cv,
                         contourIdx=-1,
                         lineType=0,
                         thickness=4,
                         color=color_dict[class_dict[v_prediction + 1]])
        cv2.drawContours(image=z15_img,
                         contours=contours_cv,
                         contourIdx=-1,
                         lineType=0,
                         thickness=4,
                         color=color_dict[class_dict[v_prediction + 1]])
        cv2.drawContours(image=z30_img,
                         contours=contours_cv,
                         contourIdx=-1,
                         lineType=0,
                         thickness=4,
                         color=color_dict[class_dict[v_prediction + 1]])
        cv2.drawContours(image=z45_img,
                         contours=contours_cv,
                         contourIdx=-1,
                         lineType=0,
                         thickness=4,
                         color=color_dict[class_dict[v_prediction + 1]])
        cv2.drawContours(image=z60_img,
                         contours=contours_cv,
                         contourIdx=-1,
                         lineType=0,
                         thickness=4,
                         color=color_dict[class_dict[v_prediction + 1]])
        cv2.drawContours(image=z75_img,
                         contours=contours_cv,
                         contourIdx=-1,
                         lineType=0,
                         thickness=4,
                         color=color_dict[class_dict[v_prediction + 1]])
        #  label and score text
        img_v = cv2.putText(img=img,
                            text=caption,
                            org=(x1, y1 + 8),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(255, 255, 255),
                            thickness=2,
                            lineType=cv2.LINE_AA)
        img_v0 = cv2.putText(img=z0_img,
                             text=caption,
                             org=(x1, y1 + 8),
                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=1,
                             color=(255, 255, 255),
                             thickness=2,
                             lineType=cv2.LINE_AA)
        img_v15 = cv2.putText(img=z15_img,
                              text=caption,
                              org=(x1, y1 + 8),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=1,
                              color=(255, 255, 255),
                              thickness=2,
                              lineType=cv2.LINE_AA)
        img_v30 = cv2.putText(img=z30_img,
                              text=caption,
                              org=(x1, y1 + 8),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=1,
                              color=(255, 255, 255),
                              thickness=2,
                              lineType=cv2.LINE_AA)
        img_v45 = cv2.putText(img=z45_img,
                              text=caption,
                              org=(x1, y1 + 8),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=1,
                              color=(255, 255, 255),
                              thickness=2,
                              lineType=cv2.LINE_AA)
        img_v60 = cv2.putText(img=z60_img,
                              text=caption,
                              org=(x1, y1 + 8),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=1,
                              color=(255, 255, 255),
                              thickness=2,
                              lineType=cv2.LINE_AA)
        img_v75 = cv2.putText(img=z75_img,
                              text=caption,
                              org=(x1, y1 + 8),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=1,
                              color=(255, 255, 255),
                              thickness=2,
                              lineType=cv2.LINE_AA)
        # get for minera_dist_mask_img mask
        minera_dist_mask_img = apply_mask(minera_dist_mask_img.astype(np.uint8), cv_mask,
                                          color_dict[class_dict[v_prediction + 1]])

        # result stats
        if v_prediction == 0:
            area_shiying = area_shiying + area
        elif v_prediction == 1:
            area_changshi = area_changshi + area
        elif v_prediction == 2:
            area_jiadasy = area_jiadasy + area
        elif v_prediction == 3:
            area_bianzhiyan = area_bianzhiyan + area
        elif v_prediction == 4:
            area_suanxingpenchuyan = area_suanxingpenchuyan + area
        elif v_prediction == 5:
            area_fangjieshi = area_fangjieshi + area
        elif v_prediction == 6:
            area_yunmu = area_yunmu + area
        elif v_prediction == 7:
            area_chenjiyan = area_chenjiyan + area
        elif v_prediction == 8:
            area_baiyunshi = area_baiyunshi + area
        elif v_prediction == 9:
            area_zishengliantukuangwu = area_zishengliantukuangwu + area

    # save final mask to local
    cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
                "/dataset_ning71x-16-0672/crop/mask_v_" + name_img_file
                , mask_img)
    cv2.imwrite("/home/xkjs/Downloads/data/BoPian"
                "/dataset_ning71x-16-0672/crop/m_distribution_" + name_img_file
                , minera_dist_mask_img)
    # save image slices to gif or video for vision
    images_gif = []
    # # pad odd 2 even
    newsize = img_size_odd2even(img_v0)
    # resize image
    img = cv2.resize(img, newsize)
    img_v0 = cv2.resize(img_v0, newsize)
    img_v15 = cv2.resize(img_v15, newsize)
    img_v30 = cv2.resize(img_v30, newsize)
    img_v45 = cv2.resize(img_v45, newsize)
    img_v60 = cv2.resize(img_v60, newsize)
    img_v75 = cv2.resize(img_v75, newsize)
    # padding
    img = pad_images_to_same_size(img)
    img_v0 = pad_images_to_same_size(img_v0)
    img_v15 = pad_images_to_same_size(img_v15)
    img_v30 = pad_images_to_same_size(img_v30)
    img_v45 = pad_images_to_same_size(img_v45)
    img_v60 = pad_images_to_same_size(img_v60)
    img_v75 = pad_images_to_same_size(img_v75)
    # image_for_gif = ['img', 'img_v0', 'img_v15',
    #                  'img_v30', 'img_v45', 'img_v60', 'img_v75']
    images_gif.append(img)
    images_gif.append(img_v0)
    images_gif.append(img_v15)
    images_gif.append(img_v30)
    images_gif.append(img_v45)
    images_gif.append(img_v60)
    images_gif.append(img_v75)

    # save video
    video_path_out = "/home/xkjs/Downloads/data/" \
                     "BoPian/dataset_ning71x-16-0672" \
                     "/crop/gif_" + name_img_file.split('.')[-2] + ".avi"
    out = cv2.VideoWriter(video_path_out,
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          fps,
                          img_v0.shape[:2])
    # print("frame len: ", len(frame_array))
    for i in range(len(images_gif)):
        out.write(np.uint8(images_gif[i]))
    out.release()

    # print("area_shiying:", area_shiying)
    # print("area_changshi:", area_changshi)
    # print("area_jiadasy:", area_jiadasy)
    # print("area_bianzhiyan:", area_bianzhiyan)
    # print("area_suanxingpenchuyan:", area_suanxingpenchuyan)
    # print("area_fangjieshi:", area_fangjieshi)
    # print("area_yunmu:", area_yunmu)
    # print("area_chenjiyan:", area_chenjiyan)
    # print("area_baiyunshi:", area_baiyunshi)
    # print("area_zishengliantukuangwu:", area_zishengliantukuangwu)

    # miankonglv

    print("\n################### miankonglv #####################\n")
    miankonglv = (area_total - (area_shiying + area_changshi +
                                area_jiadasy + area_bianzhiyan +
                                area_suanxingpenchuyan + area_fangjieshi +
                                area_fangjieshi + area_yunmu +
                                area_chenjiyan + area_baiyunshi +
                                area_zishengliantukuangwu)) / area_total
    print("miankonglv:", str(np.round(miankonglv * 100, 4)) + "%")

    # area ratio
    print("\n\n################### fina result #####################\n\n")
    print("shiying:", str(np.round((area_shiying / area_total) * 100, 4)) + "%")
    print("changshi:", str(np.round((area_changshi / area_total) * 100, 4)) + "%")
    print("jiadasy:", str(np.round((area_jiadasy / area_total) * 100, 4)) + "%")
    print("bianzhiyan:", str(np.round((area_bianzhiyan / area_total) * 100, 4)) + "%")
    print("suanxingpenchuyan:", str(np.round((area_suanxingpenchuyan / area_total) * 100, 4)) + "%")
    print("fangjieshi:", str(np.round((area_fangjieshi / area_total) * 100, 4)) + "%")
    print("yunmu:", str(np.round((area_yunmu / area_total) * 100, 4)) + "%")
    print("chenjiyan:", str(np.round((area_chenjiyan / area_total) * 100, 4)) + "%")
    print("baiyunshi:", str(np.round((area_baiyunshi / area_total) * 100, 4)) + "%")
    print("zishengliantukuangwu:", str(np.round((area_zishengliantukuangwu / area_total) * 100, 4)) + "%")

    # ####################### pore segmentation  ####################

    from pore_segmentation import model_builder
    from pore_segmentation import utils, helpers
    import tensorflow as tf

    # para for model
    class_names_list, label_values = helpers.get_label_info("../pore_segmentation/class_dict.csv")
    num_classes = len(label_values)
    checkpoint_path = "../pore_segmentation/checkpoints_pore/model.ckpt"

    model_seg = "FC-DenseNet103"
    crop_width = 800
    crop_height = 800

    # Initializing   network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    net_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    net_output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])

    network, _ = model_builder.build_model(model_seg, net_input=net_input,
                                           num_classes=num_classes,
                                           crop_width=crop_width,
                                           crop_height=crop_height,
                                           is_training=False)

    sess.run(tf.global_variables_initializer())

    print('Loading seg_model checkpoint weights')
    saver = tf.train.Saver(max_to_keep=1000)
    saver.restore(sess, checkpoint_path)

    # print("Testing image " + image_path)

    loaded_image = utils.load_image(IMAGE_DIR)
    resized_image = cv2.resize(loaded_image, (crop_width, crop_height))
    input_image = np.expand_dims(np.float32(resized_image[:crop_height, :crop_width]), axis=0) / 255.0

    # st = time.time()
    output_image = sess.run(network, feed_dict={net_input: input_image})

    # run_time = time.time() - st

    output_image = np.array(output_image[0, :, :, :])
    output_image = helpers.reverse_one_hot(output_image)

    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
    segmented_pore_img = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite("segmented_pore_img.png", segmented_pore_img)




