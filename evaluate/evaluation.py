from coco_eval import run_eval
from multipose_utils.multipose_model import get_model
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
import os
import json

sys.path.append('..')
import torch
import cv2
from collections import OrderedDict


class Evaluate:
    def __init__(self, model_path, image_dir, output_dir, anno_path, vis_dir,
                 image_list_txt='image_info_val2014_1k.txt'):
        self.model = self._get_model(model_path)
        self._image_dir = image_dir
        self._output_dir = output_dir
        self._anno_path = anno_path
        self.results = None
        self._vis_dir = vis_dir
        self.image_list_txt = image_list_txt

    @staticmethod
    def _get_model(model_path):
        # Notice, if you using the
        with torch.autograd.no_grad():
            # this path is with respect to the root of the project
            state_dict = torch.load(model_path)['state_dict']
            new_state_dict = OrderedDict()
            for key in state_dict.keys():
                new_state_dict['module.' + key] = state_dict[key]
            model = get_model(trunk='vgg19')
            model = torch.nn.DataParallel(model)  # .cuda()
            model.load_state_dict(new_state_dict)
            model.eval()
            model.float()
            model = model.cuda()
        return model

    def run(self):
        self.results = run_eval(image_dir=self._image_dir, anno_dir=self._anno_path, vis_dir=self._vis_dir,
                                image_list_txt=self.image_list_txt,
                                model=self.model, preprocess='rtpose')


if __name__ == '__main__':
    os.path.join(os.getcwd(), '..')
    main_dir = r'C:\Users\maddy\OneDrive - Knights - University of Central Florida\KeyPointDetection\CAP5415-KeypointDetection\multipose_utils'
    image_dir = main_dir + r'\dataset\COCO\images'
    model_path = main_dir + r'\coco_pose_iter_440000.pth.tar'
    output_dir = r'\results'
    anno_path = main_dir + r'\dataset\COCO\annotations'
    vis_dir = '/data/coco/vis'
    preprocess = 'rtpose'
    eval = Evaluate(model_path, image_dir, output_dir, anno_path, vis_dir)
    eval.run()

