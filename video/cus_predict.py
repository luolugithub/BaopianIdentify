"""
-*- coding: utf-8 -*-
@Author  : LuoLu
@Time    : 2020-11-04 16:23
@File: cus_predict.py
@Github    : https://github.com/luolugithub
@Email    : argluolu@gmail.com
"""
import glob
import sys
from argparse import ArgumentParser
from torchvision import transforms
import torchvision.models as models
from video.utils import *
from video.models import *
from torch import nn
from heapq import nlargest

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# cus pre
def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--model-path", "-m", help="Path to a trained model.")
    parser.add_argument("--video-path", "-v", help="Path to a video file.")
    parser.add_argument("--frames-cnt", "-f", help="Number of video frames for random selection.", default=7, type=int)
    parser.add_argument("--model-type", help="Model to run. Two options: 'cnn-avg' or 'cnn-rnn'.",
                        default="cnn-avg")
    parser.add_argument("--bilstm", action="store_true", help="Whether the LSTM is bidirectional")
    parser.add_argument("--cnn-model", help="CNN pretrained Model to use. Two options: 'resnet18' or 'resnet34'.",
                        default="resnet18")
    parser.add_argument("--gpu", action="store_true", help="Whether to run using GPU or not.")
    return parser.parse_args()


def predict(model, video, device):
    model.eval()
    inputs, _ = collate_fn([[video, torch.tensor([0])]])
    videos = inputs.to(device)

    with torch.no_grad():
        pred_labels = model(videos).cpu()
    # add reliable info
    smax = nn.Softmax(1)
    smax_out = smax(pred_labels)
    max_index = list(map(list(smax_out[0]).index, nlargest(3, smax_out[0])))
    reliable = nlargest(3, smax_out[0].cpu().numpy())
    print("max_index: ", max_index)
    # print("reliable: ", reliable)
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
    cnn_model = models.resnet34(pretrained=True) if cnn_model == "resnet34" else models.resnet18(pretrained=True)
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
    print("type video:", type(video))
    print("len video:", len(video))

    prediction, reliable = predict(model, video, device)
    print("predict result:", class_dict[prediction + 1])
    print("reliable:", reliable[0])


if __name__ == '__main__':
    model_type = "cnn-avg"
    cnn_model = "resnet18"
    frames_cnt = 7
    bilstm = True
    model_path = "../checkpoint/baseline_best.pth"
    video_path = "/home/xkjs/Downloads/data" \
                 "/BoPian/dataset_ning71x-16-0672" \
                 "/video/k_8.avi"

    class_dict = {}
    with open("/home/xkjs/PycharmProjects/"
              "instance2video/"
              "video/classInd.txt") as file:
        for line in file:
            (key, val) = line.split()
            class_dict[int(key)] = val
    print("class_dict:", class_dict)
    # print("class_dict:", class_dict[1])
    # main(model_type,
    #      cnn_model,
    #      frames_cnt,
    #      bilstm,
    #      model_path,
    #      video_path,
    #      class_dict)

    # test
    # enumerate_for_gif = ['img', 'img_v0', 'img_v15',
    #                      'img_v30', 'img_v45', 'img_v60', 'img_v75']
    # for iterm in enumerate_for_gif:
    #     print(iterm)
    image_folder = '/home/xkjs/Downloads/data/BoPian/' \
                   'dataset_ning71x-16-0672/crop/annm_*.jpg'
    video_name = '/home/xkjs/Downloads/data/BoPian' \
                 '/dataset_ning71x-16-0672/' \
                 'crop/annm_keli.avi'

    images = [img for img in sorted(glob.glob(image_folder))]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
