from coco_eval import run_eval
from multipose_utils.multipose_model import get_model
from torch import load
import sys
sys.path.append('..')
import torch
import cv2
from collections import OrderedDict


# Notice, if you using the
with torch.autograd.no_grad():
    # this path is with respect to the root of the project
    main_dir = r'C:\Users\maddy\OneDrive - Knights - University of Central Florida\CAP5415-KeypointDetection'
    model_path = main_dir + r'\coco_pose_iter_440000.pth.tar'

    state_dict = torch.load(model_path)['state_dict']
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        new_state_dict['module.'+key] = state_dict[key]
    model = get_model(trunk='vgg19')
    model = torch.nn.DataParallel(model)# .cuda()
    model.load_state_dict(new_state_dict)
    model.eval()
    model.float()
    model = model.cuda()

    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in
    # this repo used 'vgg' preprocess
    image_dir = main_dir + r'\dataset\COCO\images'
    model_path = main_dir + r'\coco_pose_iter_440000.pth.tar'
    output_dir = r'\results'
    vis_dir = 'd'
    anno_path = main_dir + r'\dataset\COCO\annotations'
    run_eval(image_dir=image_dir, anno_dir=anno_path, vis_dir='/data/coco/vis',
             image_list_txt='image_info_val2014_1k.txt',
             model=model, preprocess='rtpose')
