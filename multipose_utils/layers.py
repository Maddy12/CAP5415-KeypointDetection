import torch
import torch.nn as nn
import cv2

# Local
from multipose_utils.generate_pose import *
from multipose_utils import generate_pose
from evaluate.coco_eval import *


class RegionProposal:
    def __init__(self, output1, output2, img_orig, model):
        super(RegionProposal, self).__init__()
        """
        To get heatmaps and pafs:
            heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)
            pafs = output1.cpu().data.numpy().transpose(0, 2, 3, 1)
        :param output1: Multi-pose model output 1
        :param output2: Multi-pose model output 2
        """
        self.pafs = output1.cpu().detach().numpy()
        self.heatmaps = output2.cpu().detach().numpy()  # heatmap
        self.param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        self.num_joints = NUM_JOINTS
        self.num_limbs = NUM_LIMBS
        self.oriImg = cv2.imread(img_orig)
        self.model = model
        self.preprocess = 'vgg'

    def preprocess_img(self):
        shape_dst = np.min(self.oriImg.shape[0:2])

        # Get results of original image
        multiplier = get_multiplier(self.oriImg)
        orig_paf, orig_heat = get_outputs(
            multiplier, self.oriImg, self.model, self.preprocess)

        # Get results of flipped image
        swapped_img = self.oriImg[:, ::-1, :]
        flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
                                                self.model, self.preprocess)

        # compute averaged heatmap and paf
        paf, heatmap = handle_paf_and_heat(
            orig_heat, flipped_heat, orig_paf, flipped_paf)

        # choose which post-processing to use, our_post_processing
        # got slightly higher AP but is slow.
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        canvas, to_plot, candidate, subset = decode_pose(
            self.oriImg, param, heatmap, paf)
        return {'paf': paf, 'heatmap': heatmap, 'canvas':canvas, 'to_plot':to_plot,
                'candidate':candidate, 'subset':subset}

    def forward(self):
        joint_list_per_joint_type = NMS(self.param, self.heatmaps)
        joint_list = np.array([tuple(peak) + (joint_type,) for joint_type, joint_peaks in
                               enumerate(joint_list_per_joint_type) for peak in joint_peaks])
        paf_upsamp = cv2.resize(self.pafs, (self.img_orig.shape[1], self.img_orig.shape[0]),
                                interpolation=cv2.INTER_CUBIC)
        connected_limbs = find_connected_joints(self.param, paf_upsamp, joint_list_per_joint_type)
        person_to_joint_assoc = group_limbs_of_same_person(
            connected_limbs, joint_list)
        return person_to_joint_assoc