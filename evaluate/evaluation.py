from coco_eval import run_eval
from multipose_utils.multipose_model import get_model
import sys
sys.path.append('..')
import torch
from collections import OrderedDict

# Notice, if you using the
with torch.autograd.no_grad():
    # this path is with respect to the root of the project
    # main_dir = r'C:\Users\maddy\OneDrive - Knights - University of Central Florida\CAP5415-KeypointDetection'
    main_dir = '/home/CAP5415-KeypointDetection'
    model_path = main_dir + '/coco_pose_iter_440000.pth.tar'

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
    image_dir = main_dir + '/dataset/COCO/images'
    model_path = main_dir + '/coco_pose_iter_440000.pth.tar'
    output_dir = '/results'
    anno_path = main_dir + '/dataset/COCO/'
    vis_dir = main_dir + '/dataset/COCO/vis'
    run_eval(image_dir=image_dir, anno_dir=anno_path, vis_dir='/data/coco/vis',
             image_list_txt='image_info_val2014_1k.txt',
    #         image_list_txt='image_info_val2014_10.txt',
             model=model, preprocess='rtpose')


