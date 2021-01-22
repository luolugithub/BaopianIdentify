"""
-*- coding: utf-8 -*-
@Author  : LuoLu
@Time    : 2021-01-22 12:43
@File: config.py
@Github    : https://github.com/luolugithub
@Email    : argluolu@gmail.com
"""
'''
    configure model para
    image path
    output path
    in this file
'''


class Config(object):
    # predict video para
    model_type = "cnn-avg"
    cnn_model = "resnet34"
    frames_cnt = 7
    bilstm = True
    # ###########  directory of video model ##################
    model_path = "checkpoint/" \
                 "T-Resnet34_best.pth"
    # class index
    class_ind_path = "video/classInd.txt"
    # GPU device
    device = "cuda:1"
    # instance model path
    COCO_MODEL_PATH = "checkpoint/" \
                      "inst_e0090.h5"
    # #############  Directory of images #########################
    # 单偏光
    IMAGE_DIR = "/home/xkjs/Downloads/data/BoPian/dataset_ning71x-16-0672/" \
                "7img/gao30-245X-14-0202_m008_s.png"
    # class name
    class_names = ['BG', '_background_', 'keli']
    # zhengjiaopianguang img
    j0_extension = "_0.png"
    j15_extension = "_15.png"
    j30_extension = "_30.png"
    j45_extension = "_45.png"
    j60_extension = "_60.png"
    j75_extension = "_75.png"
    # create video param
    fps = 1
    # segmentation class csv
    path_pore_seg_class_dict = "pore_segmentation/class_dict.csv"
    # path of pore segmentation model
    path_pore_seg = "pore_segmentation/" \
                    "checkpoints_pore/model.ckpt"
    # type of segmentaion model
    type_seg_model = "FC-DenseNet103"
    # crop size of segmentaion
    crop_seg_width = 800
    crop_seg_height = 800
    # 结构分类
    # 胶结类型
    path_model_jiaojie = "structure/best_model-jiaojie.tar"
    # 接触关系
    path_model_contact = "structure/best_model-jiechu.tar"
    # 磨圆度
    path_model_moyuand = "structure/best_model-moyuan.tar"

    # color_dict of mineral
    color_dict = {'shiying': (204, 51, 0),
                  'changshi': (255, 255, 0),
                  'suanxingpenchuyan': (0, 153, 0),
                  'zjxpengchuyan': (204, 0, 255),
                  'chenjiyan': (204, 102, 153),
                  'bianzhiyan': (0, 102, 255),
                  'qinruyan': (255, 204, 153),
                  'yunmu': (102, 0, 204),
                  'zajinizhi': (102, 255, 204),
                  'fangjieshi': (255, 204, 0),
                  'baiyunshi': (102, 204, 255),
                  'zishengliantukuangwu': (255, 51, 204)}
    # *************************** mineral dingming dict******************
    dict_mineral_cf = {1: "石英砂岩",
                       2: "长石石英砂岩",
                       3: "岩屑石英砂岩",
                       4: "长石砂岩",
                       5: "岩屑长石砂岩",
                       6: "长石岩屑砂岩",
                       7: "岩屑砂岩"}
    # 粒径分布
    class_keli_distribute = ['泥', '细粉砂', '粗粉砂', '极细砂', '细砂', '中砂', '粗砂', '细砾', '中砾', '粗砾']

    # #############   extracted keli image ###################
    # ################# 鉴定报告: 颗粒提取图 ############
    path_keli_extract = "/home/xkjs/Downloads/data/BoPian" \
                        "/dataset_ning71x-16-0672/crop/extracted_"

    # #############   mineral distribution image  ###################
    # ################# 鉴定报告: 矿物分布图 ############
    path_mineral_distributed = "/home/xkjs/Downloads/data/BoPian" \
                               "/dataset_ning71x-16-0672/crop/m_distribution_"

    # output path of mp4
    path_out_video = "/home/xkjs/Downloads/data/" \
                     "BoPian/dataset_ning71x-16-0672" \
                     "/crop/"
    # pore segmented image
    # default path = root of project
    path_seged_img = "800segmented_pore_img.png"
    # ################## 鉴定报告结果图: 孔径分布图 ############
    # default path = root of project
    path_fig_kongjing = '孔径分布图.jpg'
    # ################## 鉴定报告结果图: 粒径分布图 ############
    # default path = root of project
    path_fig_lijing = '粒径分布图.jpg'
    # baopian identify report result json output path
    bp_report = 'bp_report.json'
