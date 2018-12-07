import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from PIL import Image
# Local
from multipose_utils.generate_pose_RETIRED import *
from evaluate.coco_eval_RETIRED import *
import pdb

class RegionProposal(nn.Module):
    def __init__(self, output1, output2, oriImg, model):
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
        self.paf =output1  # .transpose(0,1 ).transpose(0,2)  # .transpose(0,2 ).transpose(0,1)  # .transpose(1, 2).transpose(2, 3)
        self.heatmap = output2  #.transpose(0,1 ).transpose(0,2)  # .transpose(0,2 ).transpose(0,1).shape  # .transpose(1, 2).transpose(2, 3)
        self.param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        self.num_joints = NUM_JOINTS
        self.num_limbs = NUM_LIMBS
        self.oriImg = oriImg
        self.model = model
        self.oriImg_path = oriImg
        self.oriImg = cv2.imread(oriImg)
        self.preprocess = 'vgg'
        self.buffer_vert = 10
        self.buffer_horiz = 5
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
        persons, joint_list = get_person_to_join_assoc(self.oriImg, self.param,
                                           self.heatmap, self.paf)
        persons = torch.from_numpy(persons).float()
        regions, bounds = self.find_regions(self.oriImg, joint_list, persons)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        final_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        regions = torch.stack([torch.autograd.Variable(final_transform(Image.fromarray(region))) for region in regions])
        y_pred = self.model(regions)
        preds = np.argmax(y_pred.cpu().detach().numpy(), axis=1)
        idx = np.argwhere(np.asarray(preds))
        try:
            filtered = persons[idx[0]]
        except Exception as e:
            error = e
            import pdb; pdb.set_trace()
        filtered = filtered.reshape(filtered.shape[0], filtered.shape[-1])
        # to_plot, canvas = plot_pose(self.oriImg, joint_list, filtered)
        # append_result(self.oriImg_path, filtered, joint_list, outputs)
        # return outputs  # this goes into coco_eval
        return self.oriImg_path, filtered, joint_list

    def find_regions(self, img_orig, joint_list, person_to_joint_assoc):
        """
        Find regions of potential humans by fisding the the max and min points with respect  to the x and y direction
        for each delcareed human then produce the region of the image associated with those points
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
                    # Put a buffer around the potential humans
                    maxX = maxX + self.buffer_horiz
                    minX = minX - self.buffer_horiz
                    maxY = maxY + self.buffer_vert
                    minY = minY - self.buffer_vert

                    if maxX > img_orig.shape[1]:
                        maxX = img_orig.shape[1]
                    if minX < 0:
                        minX = 0
                    if maxY > img_orig.shape[0]:
                        maxY = img_orig.shape[0]
                    if minY < 0:
                        minY = 0

                    if maxX == minX:
                        maxX = maxX + 1
                    if maxY == minY:
                        maxY = maxY + 1

            regions.append(img_orig[minY:maxY, minX:maxX, :])
            bounds.append([minX, maxX, minY, maxY])
        return regions, bounds
