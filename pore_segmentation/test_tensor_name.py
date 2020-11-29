"""
-*- coding: utf-8 -*-
@Author  : LuoLu
@Time    : 2020-11-27 14:39
@File: test_tensor_name.py
@Github    : https://github.com/luolugithub
@Email    : argluolu@gmail.com
"""
import os
from tensorflow.python import pywrap_tensorflow

model_dir = "/home/xkjs/PycharmProjects/instance2video/pore_segmentation/checkpoints_pore"
checkpoint_path = os.path.join(model_dir, "model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
if __name__ == '__main__':
    for key in var_to_shape_map:
        # print("tensor_name: ", key, end=' ')
        print("tensor_name: ", key)
        # print(reader.get_tensor(key))

