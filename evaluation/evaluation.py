import unittest
import torch
from coco_eval import run_eval
import sys
sys.path.append(r'C:\Users\maddy\OneDrive - Knights - University of Central Florida\CAP5415-KeypointDetection\version1\pytorch_Realtime_Multi-Person_Pose_Estimation\network')
from rtpose_vgg import get_model, use_vgg
from torch import load

# Notice, if you using the
with torch.autograd.no_grad():
    # this path is with respect to the root of the project
    weight_name = './network/weight/best_pose.pth'
    state_dict = torch.load(weight_name)
    model = get_model(trunk='vgg19')

    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    model.float()
    model = model.cuda()

    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in
    # this repo used 'vgg' preprocess
    run_eval(image_dir='/data/coco/images/', anno_dir='/data/coco', vis_dir='/data/coco/vis',
             image_list_txt='./evaluate/image_info_val2014_1k.txt',
             model=model, preprocess='vgg')