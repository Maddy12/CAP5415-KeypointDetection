import torch
import torch.nn as nn
from multipose_utils.generate_pose import *


class RegionProposal():
    def __init__(self, output1, output2):
        """
        To get heatmaps and pafs:
            heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)
            pafs = output1.cpu().data.numpy().transpose(0, 2, 3, 1)
        :param output1: Multipose model output 1
        :param output2: Multipose model output 2
        """
        self.output1 = output1
        self.output2 = output2

    def forward(self):
