import numpy as np
import cv2
import torch
import os

# Local
from evaluate.coco_eval import get_multiplier
from multipose_utils.dataset_utils.coco_data.preprocessing import vgg_preprocess
from multipose_utils import im_transform
from multipose_utils.multipose_model import get_model


def test_dimensions(cuda=False):
    img = 'ski.jpg'
    img = cv2.imread(img)
    multiplier = get_multiplier(img)
    m = 1
    scale = multiplier[m]
    max_scale = multiplier[-1]
    max_size = max_scale * img.shape[0]
    # padding
    max_cropped, _, _ = im_transform.crop_with_factor(
        img, max_size, factor=8, is_ceil=True)
    batch_images = np.zeros(
        (len(multiplier), 3, max_cropped.shape[0], max_cropped.shape[1]))
    inp_size = scale * img.shape[0]
    im_croped, im_scale, real_shape = im_transform.crop_with_factor(
        img, inp_size, factor=8, is_ceil=True)
    im_data = vgg_preprocess(im_croped)
    batch_images[m, :, :im_data.shape[1], :im_data.shape[2]] = im_data
    model = get_model()  # dont care about the weights just want out dimensions
    batch_var = torch.from_numpy(batch_images) # .cuda().float()
    if cuda:
        model = model.cuda()
        batch_var = batch_var.cuda()
    batch_var = batch_var.float()
    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
    return output1, output2