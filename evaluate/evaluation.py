from multipose_utils.multipose_model import get_model
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
import os
import numpy as np
from torch import nn

sys.path.append('..')
import torch
import cv2
from collections import OrderedDict
from classifier_utils import classifier_model
from evaluate.coco_eval import *
from multipose_utils.layers import RegionProposal
from multipose_utils import multipose_model


class Evaluate(nn.Module):
    def __init__(self, model, classifier, image_dir, output_dir, anno_path, vis_dir,
                 image_list_txt='image_info_val2014_1k.txt'):
        super(Evaluate, self).__init__()
        self.model = model
        self._image_dir = image_dir
        self._output_dir = output_dir
        self._anno_path = anno_path
        self.results = None
        self._vis_dir = vis_dir
        self.image_list_txt = image_list_txt
        self.classifier = classifier

    def forward(self, img_path):
        """Run the evaluation on the test set and report mAP score
            Get output of model for original image and flipped image than average them.
            :param model: the model to test
            :returns: float, the reported mAP score
            """
        oriImg = cv2.imread(os.path.join(image_dir, 'val2014/' + img_path))
        multiplier = get_multiplier(oriImg)
        orig_paf, orig_heat = get_outputs(multiplier, oriImg, self.model, preprocess='vgg')

        # Get results of flipped image
        swapped_img = oriImg[:, ::-1, :]
        flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img, self.model, preprocess)

        # compute averaged heatmap and paf
        paf, heatmap = handle_paf_and_heat(orig_heat, flipped_heat, orig_paf, flipped_paf)
        # regions = RegionProposal(paf, heatmap, oriImg, self.classifier_model)
        # outputs = regions.forward()
        return paf, heatmap

    def run(self, outputs):
        self.results = run_eval(image_dir=self._image_dir, anno_dir=self._anno_path, vis_dir=self._vis_dir,
                                image_list_txt=self.image_list_txt,
                                model=self.model, preprocess='rtpose')


def run_evaluation(model_path, post_model_path, image_dir, output_dir, anno_path, vis_dir, image_list_txt):
    # https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose/blob/master
    img_ids, img_paths, img_heights, img_widths = get_coco_val(image_list_txt)

    print("Total number of validation images {}".format(len(img_ids)))
    # iterate all val images
    outputs = []
    print("Processing Images in validation set")
    model = multipose_model.get_multipose_model(model_path)
    classifier = classifier_model.get_model(post_model_path)
    eval = Evaluate(model, classifier, image_dir, output_dir, anno_path, vis_dir)
    outputs = list()
    for i in range(len(img_ids)):
        if i % 10 == 0 and i != 0:
            print("Processed {} images".format(i))
        paf, heatmap = eval.forward(img_paths[i])
        regions = RegionProposal(paf, heatmap, img_paths[i], classifier)
        oriImg_path, filtered, joint_list = regions.forward()
        append_result(oriImg_path, filtered, joint_list, outputs)
    eval.run(outputs)
    # eval_coco(outputs=outputs, dataDir=anno_path, imgIds=img_ids)


if __name__ == '__main__':
    os.chdir('..')
    main_dir = r'multipose_utils\\'
    image_dir = main_dir + r'\dataset\COCO\images'
    model_path = main_dir + r'multipose_model\coco_pose_iter_440000.pth.tar'
    output_dir = r'\results'
    anno_path = main_dir + r'\dataset\COCO\annotations'
    vis_dir = '/data/coco/vis'
    preprocess = 'rtpose'
    post_model_path = 'classifier_utils/model_best.pth.tar',
    image_list_txt = 'image_info_val2014_1k.txt'
    run_evaluation(model_path, post_model_path, image_dir, output_dir, anno_path, vis_dir, image_list_txt)
