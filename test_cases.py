import numpy as np
import cv2
import torch
import os
from torchvision import transforms
# Local
from evaluate.coco_eval import *
from multipose_utils.dataset_utils.coco_data.preprocessing import vgg_preprocess
from multipose_utils import im_transform
from multipose_utils.multipose_model import get_model
from multipose_utils.generate_pose import *

person_to_joint_assoc = [[-1, 3, 7, 11, 14, 18,
                          21, 23, 26, 31, -1, 38,
                          41, -1, -1, -1, 49, 53,
                          19.51260906, 13],
                         [0,     1, 5, 9, 13, 16,
                          19, 22, 24, 28, 34, 36,
                          39, 42, 45, 46, 47, 51,
                          25.41747306, 18],
                         [-1, 2, 6, 10, -1, 17,
                          20, -1, 25, 29, 35, 37,
                          40, 44, -1, -1, 48, 52,
                          16.6448012, 13],
                         [-1, 4, 8, 12, 15, -1,
                          -1, -1, 27, 32, -1, -1,
                          -1, -1, -1, -1, 50, -1,
                          7.68753984, 7]]

def test_outputs(model):
    # heatmap = torch.load('output2.pt')
    # paf = torch.load('output1.pt')
    # # model = get_model()
    img = cv2.imread('ski.jpg')
    # means = [0.485, 0.456, 0.406]
    # stds = [0.229, 0.224, 0.225]
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, stds)])
    # batch = torch.zeros([1, 3, img.shape[0], img.shape[1]])
    # batch[0, :, :, :] = transform(img)
    # (output1, output2), _ = model(batch)
    output2 = torch.load('output2.pt')
    output1 = torch.load('output1.pt')
    output2 = output2.transpose(1, 2).transpose(2, 3)
    output1 = output1.transpose(1, 2).transpose(2, 3)
    one_heatmap = output2[0, :, :, :]
    one_paf = output1[0, :, :, :]
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    joint_list_per_joint_type = NMS(param, one_heatmap.detach().numpy())
    persons, joint_list = get_person_to_join_assoc(img, param, one_heatmap.detach().numpy(), one_paf.detach().numpy())
    torch.from_numpy(persons[0]).float()
    return output1, output2



def test_decode_pose(cuda=False):
    oriImg = cv2.imread('ski.jpg')
    model = get_model()
    if cuda:
        model = model.cuda()
    preprocess = 'vgg'
    # Get the shortest side of the image (either height or width)
    shape_dst = np.min(oriImg.shape[0:2])

    # Get results of original image
    multiplier = get_multiplier(oriImg)
    orig_paf, orig_heat = get_outputs(
        multiplier, oriImg, model, preprocess)

    # Get results of flipped image
    swapped_img = oriImg[:, ::-1, :]
    flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
                                            model, preprocess, cuda)

    # compute averaged heatmap and paf
    paf, heatmap = handle_paf_and_heat(
        orig_heat, flipped_heat, orig_paf, flipped_paf)

    # choose which post-processing to use, our_post_processing
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    person_to_joint_assoc, joint_list = get_person_to_join_assoc(oriImg, param, heatmap, paf)
    return person_to_joint_assoc, joint_list


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