import unittest
import torch
from coco_eval import run_eval
from rtpose_vgg import get_model, use_vgg
from torch import load
import sys
sys.path.append('..')
import torch
import pose_estimation
import cv2
from collections import OrderedDict
import os
import argparse
import torchvision.models as models
import graphviz
from graphviz import Digraph
from torchviz import make_dot, make_dot_from_trace
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'


# make_dot was moved to https://github.com/szagoruyko/pytorchviz
from torchviz import make_dot


def construct_model(args):

	model = pose_estimation.PoseModel(num_point=19, num_vector=19)
	state_dict = torch.load(args.model)['state_dict']
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k
		new_state_dict[name] = v
	state_dict = model.state_dict()
	state_dict.update(new_state_dict)
	model.load_state_dict(state_dict)
	model = model.cuda()
	model.eval()

	return model

# Notice, if you using the
with torch.autograd.no_grad():
    # this path is with respect to the root of the project

    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in
	# this repo used 'vgg' preprocess

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #print(torch.cuda.get_device_name(0))
    #parser = argparse.ArgumentParser()
   # parser.add_argument('--model', type=str, default='C:\Users\Brandon\OneDrive\UCF\Fall_2018\Computer_Vision\Project\EXP1\coco_pose_iter_440000.pth.tar', help='path to the weights file')
	
    main_dir = 'C:/Users/Brandon/OneDrive/UCF/Fall_2018/Computer_Vision/Project/EXP1/'
    image_dir = 'C:/Users/Brandon/Documents/COCO/'
    model_path = main_dir + 'coco_pose_iter_440000.pth.tar'
	
    #args = parser.parse_args()
    state_dict = torch.load(model_path)['state_dict']
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        new_state_dict['module.'+key] = state_dict[key]
    model = get_model(trunk='vgg19')
    # model = construct_model(model_path)
    #model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(new_state_dict)
    model2 = nn.Sequential(models.vgg19(True).features[1], model)
    print(model2)
	
    x = torch.randn(3, 3, 128, 128).requires_grad_(True)
   # model3 = model2(x)
    with torch.onnx.set_training(model2, False):
        trace, _ = torch.jit.get_trace_graph(model2, args=(x,))
    dot = make_dot_from_trace(trace)
    dot.format = 'svg'
    dot.render('C:/Users/Brandon/OneDrive/UCF/Fall_2018/Computer_Vision/Project/EXP1/Evaluate/test2.svg', view = True)
	
    wait = input("enter")
    # #model = model.cuda()
    # #model.eval().cuda()
    # model.eval()
	# #model = construct_model(args)
	
    # output_dir = 'C:/Users/Brandon/OneDrive/UCF/Fall_2018/Computer_Vision/Project/EXP1/results/'
    # anno_path = main_dir
    # run_eval(image_dir=image_dir, anno_dir=anno_path, image_list_txt = r'C:/Users/Brandon/OneDrive/UCF/Fall_2018/Computer_Vision/Project/EXP1/Evaluate/image_info_val2014_1k.txt', model=model, preprocess='rtpose', vis_dir = 'C:/Users/Brandon/OneDrive/UCF/Fall_2018/Computer_Vision/Project/EXP1/vis/')

	# # run_eval(image_dir=image_dir, anno_dir=anno_path, vis_dir='/data/coco/vis',
		 # # image_list_txt='image_info_val2014_1k.txt',
		 # # model=model, preprocess='rtpose')

