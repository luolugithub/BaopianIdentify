"""
-*- coding: utf-8 -*-
@Author  : LuoLu
@Time    : 2020-11-25 09:53
@File: demo_segmentation.py
@Github    : https://github.com/luolugithub
@Email    : argluolu@gmail.com
"""
import cv2
import os
import time

import numpy as np
import tensorflow as tf
from pore_segmentation import model_builder
from pore_segmentation import utils, helpers

print("tf ver:", tf.__version__)

# para for model
class_names_list, label_values = helpers.get_label_info("../pore_segmentation/class_dict.csv")
num_classes = len(label_values)
checkpoint_path = "../pore_segmentation/checkpoints_pore/model.ckpt"
image_path = "/home/xkjs/PycharmProjects/" \
             "Semantic-Segmentation-Suite-master/" \
             "Dataset_pore/test/" \
             "hu27-16-1343_m001_s.png"
model_seg = "FC-DenseNet103"
crop_width = 800
crop_height = 800


# Initializing network
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

print('Loading model checkpoint weights')
saver = tf.train.Saver(max_to_keep=1000)
saver.restore(sess, checkpoint_path)

print("Testing image " + image_path)

loaded_image = utils.load_image(image_path)
resized_image = cv2.resize(loaded_image, (crop_width, crop_height))
input_image = np.expand_dims(np.float32(resized_image[:crop_height, :crop_width]), axis=0) / 255.0

st = time.time()
output_image = sess.run(network, feed_dict={net_input: input_image})

run_time = time.time() - st

output_image = np.array(output_image[0, :, :, :])
output_image = helpers.reverse_one_hot(output_image)

out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
file_name = utils.filepath_to_name(image_path)
cv2.imwrite("%s_pred.png" % (file_name), cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
