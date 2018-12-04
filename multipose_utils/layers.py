import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from PIL import Image
# Local
from multipose_utils.generate_pose import *
from evaluate.coco_eval import *


class RegionProposal:
    def __init__(self, output1, output2, img_origs, model):
        super(RegionProposal, self).__init__()
        """
        To get heatmaps and pafs:
            heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)
            pafs = output1.cpu().data.numpy().transpose(0, 2, 3, 1)
        :param output1: Multi-pose model output 1, PAF - batch
        :param output2: Multi-pose model output 2, Heatmap - batch
        :param img_origs: List of image paths
        :param model: Model
        """
        self.pafs = output1.transpose(1, 2).transpose(2, 3)
        self.heatmaps = output2.transpose(1, 2).transpose(2, 3)
        self.param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        self.num_joints = NUM_JOINTS
        self.num_limbs = NUM_LIMBS
        self.oriImgs = img_origs
        self.model = model
        self.preprocess = 'vgg'
        self.joint_to_limb_heatmap_relationship = [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
            [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
            [2, 16], [5, 17]]

    def forward(self):
        """
        Iterates through the batch and gets the individual peoples then a bounding box around the person.
        Then transforms the resulting image region in the appropriate input dimensions for the resnet classifier.
        :return:
        """
        for i, (paf, heatmap) in enumerate(zip(self.pafs, self.heatmaps)):
            img = cv2.imread(self.oriImgs[i])
            persons, joint_list = get_person_to_join_assoc(img, self.param,
                                               heatmap.detach().numpy(), paf.detach().numpy())
            persons = torch.from_numpy(persons).float()
            regions, bounds = self.find_regions(img, joint_list, persons)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            final_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            regions = torch.stack([final_transform(Image.fromarray(region)) for region in regions])
            output = self.model(regions)
            # TODO

    def find_regions(self, img_orig, joint_list, person_to_joint_assoc):
        """
        Gets a bounded box of a person and the image.
        :param img_orig:
        :param joint_list:
        :param person_to_joint_assoc:
        :return:
        """
        # For Each person
        regions = list()
        bounds = list()
        for person_joint_info in person_to_joint_assoc:
            # For Each Limb
            maxX = 0
            minX = img_orig.shape[1]
            maxY = 0
            minY = img_orig.shape[0]

            for limb_type in range(19):
                # The Indidieces of this joint
                joint_indices = person_joint_info[
                    self.joint_to_limb_heatmap_relationship[limb_type]].int().detach().numpy()  # .astype(int)
                joint_coords = joint_list[joint_indices, 0:2]

                for joint in joint_coords:
                    maxX = int(max(maxX, joint[0]))
                    minX = int(min(minX, joint[0]))
                    maxY = int(max(maxY, joint[1]))
                    minY = int(min(minY, joint[1]))

            regions.append(img_orig[minY:maxY, minX:maxX, :])
            bounds.append([minX, maxX, minY, maxY])
        return regions, bounds