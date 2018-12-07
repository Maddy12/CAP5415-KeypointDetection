import os
from coco_eval import run_eval
import sys
sys.path.append('..')
from multipose_utils.multipose_model import get_model
import torch
from collections import OrderedDict

# Notice, if you using the
with torch.autograd.no_grad():
    # this path is with respect to the root of the project
    # main_dir = r'C:\Users\maddy\OneDrive - Knights - University of Central Florida\CAP5415-KeypointDetection'
    main_dir = '/home/CAP5415-KeypointDetection/'
    image_dir = os.path.join(main_dir, 'dataset/COCO_data/images')
    model_path = os.path.join(main_dir, 'multipose_utils/multipose_model/coco_pose_iter_440000.pth.tar')
    output_dir = os.path.join(main_dir, 'results')
    anno_dir = os.path.join(main_dir, 'dataset/COCO_data/')
    vis_dir = os.path.join(main_dir, 'dataset/COCO_data/vis')
    preprocess = 'rtpose'
    # post_model_path = os.path.join(main_dir, 'classifier_utils/model_best.pth.tar')
    post_model_path = '/home/model_best.pth.tar'
    image_list_txt = os.path.join(main_dir, 'evaluate/image_info_val2014_1k.txt')


    state_dict = torch.load(model_path)['state_dict']
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        new_state_dict['module.'+key] = state_dict[key]
    model = get_model(trunk='vgg19')
    # model = construct_model(model_path)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(new_state_dict)
    model.eval()
    model.float()
    model = model.cuda()

    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in
    # this repo used 'vgg' preprocess
    run_eval(image_dir=image_dir, anno_dir=anno_dir, vis_dir=vis_dir, image_list_txt=image_list_txt,
    #         image_list_txt='image_info_val2014_10.txt',
             model=model, preprocess='rtpose')


