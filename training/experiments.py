import sys
sys.path.append('..')
import torch
from torch import nn
from collections import OrderedDict
import torchvision.models as models
sys.path.append('../evaluate/')
from rtpose_vgg import get_model, use_vgg


main_dir = r'C:\Users\maddy\OneDrive - Knights - University of Central Florida\CAP5415-KeypointDetection'
MODEL_PATH = main_dir + r'\coco_pose_iter_440000.pth.tar'

"""
(module): rtpose_model(
    (model0): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace)
      (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace)
      (23): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace)
      (25): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (26): ReLU(inplace)
    )
    (model1_1): Sequential(
      (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): ReLU(inplace)
      (6): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
      (7): ReLU(inplace)
      (8): Conv2d(512, 38, kernel_size=(1, 1), stride=(1, 1))
    )
    (model2_1): Sequential(
      (0): Conv2d(185, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (1): ReLU(inplace)
      (2): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (3): ReLU(inplace)
      (4): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (5): ReLU(inplace)
      (6): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (7): ReLU(inplace)
      (8): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (9): ReLU(inplace)
      (10): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(128, 38, kernel_size=(1, 1), stride=(1, 1))
    )
    (model3_1): Sequential(
      (0): Conv2d(185, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (1): ReLU(inplace)
      (2): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (3): ReLU(inplace)
      (4): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (5): ReLU(inplace)
      (6): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (7): ReLU(inplace)
      (8): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (9): ReLU(inplace)
      (10): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(128, 38, kernel_size=(1, 1), stride=(1, 1))
    )
    (model4_1): Sequential(
      (0): Conv2d(185, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (1): ReLU(inplace)
      (2): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (3): ReLU(inplace)
      (4): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (5): ReLU(inplace)
      (6): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (7): ReLU(inplace)
      (8): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (9): ReLU(inplace)
      (10): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(128, 38, kernel_size=(1, 1), stride=(1, 1))
    )
    (model5_1): Sequential(
      (0): Conv2d(185, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (1): ReLU(inplace)
      (2): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (3): ReLU(inplace)
      (4): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (5): ReLU(inplace)
      (6): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (7): ReLU(inplace)
      (8): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (9): ReLU(inplace)
      (10): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(128, 38, kernel_size=(1, 1), stride=(1, 1))
    )
    (model6_1): Sequential(
      (0): Conv2d(185, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (1): ReLU(inplace)
      (2): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (3): ReLU(inplace)
      (4): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (5): ReLU(inplace)
      (6): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (7): ReLU(inplace)
      (8): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (9): ReLU(inplace)
      (10): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(128, 38, kernel_size=(1, 1), stride=(1, 1))
    )
    (model1_2): Sequential(
      (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
      (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): ReLU(inplace)
      (6): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
      (7): ReLU(inplace)
      (8): Conv2d(512, 19, kernel_size=(1, 1), stride=(1, 1))
    )
    (model2_2): Sequential(
      (0): Conv2d(185, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (1): ReLU(inplace)
      (2): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (3): ReLU(inplace)
      (4): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (5): ReLU(inplace)
      (6): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (7): ReLU(inplace)
      (8): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (9): ReLU(inplace)
      (10): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(128, 19, kernel_size=(1, 1), stride=(1, 1))
    )
    (model3_2): Sequential(
      (0): Conv2d(185, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (1): ReLU(inplace)
      (2): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (3): ReLU(inplace)
      (4): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (5): ReLU(inplace)
      (6): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (7): ReLU(inplace)
      (8): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (9): ReLU(inplace)
      (10): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(128, 19, kernel_size=(1, 1), stride=(1, 1))
    )
    (model4_2): Sequential(
      (0): Conv2d(185, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (1): ReLU(inplace)
      (2): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (3): ReLU(inplace)
      (4): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (5): ReLU(inplace)
      (6): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (7): ReLU(inplace)
      (8): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (9): ReLU(inplace)
      (10): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(128, 19, kernel_size=(1, 1), stride=(1, 1))
    )
    (model5_2): Sequential(
      (0): Conv2d(185, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (1): ReLU(inplace)
      (2): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (3): ReLU(inplace)
      (4): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (5): ReLU(inplace)
      (6): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (7): ReLU(inplace)
      (8): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (9): ReLU(inplace)
      (10): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(128, 19, kernel_size=(1, 1), stride=(1, 1))
    )
    (model6_2): Sequential(
      (0): Conv2d(185, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (1): ReLU(inplace)
      (2): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (3): ReLU(inplace)
      (4): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (5): ReLU(inplace)
      (6): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (7): ReLU(inplace)
      (8): Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (9): ReLU(inplace)
      (10): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(128, 19, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  """


def construct_model(model_path=MODEL_PATH, cuda=False):
    """
    The authors re-scale the image to be 368x654.
    The output of the final layers of the two split networks are
    * in channel: 128
    * out channel: 19
    * kernel size: (1, 1)
    * stride: (1, 1)
    :param model_path:
    :param cuda:
    :return:
    """
    with torch.autograd.no_grad():
        state_dict = torch.load(model_path)['state_dict']
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            new_state_dict['module.'+key] = state_dict[key]
        model = get_model(trunk='vgg19')
        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(new_state_dict)
        model.eval()
        model.float()
        if cuda:
            model = model.cuda()
    return model


def add_layers(model, cuda=False):
    """
    Output of original model is:
    * Conv2d(128, 19, kernel_size=(1, 1), stride=(1, 1))
    * Conv2d(128, 19, kernel_size=(1, 1), stride=(1, 1))
    :param cuda:
    :return:
    """
    # freeze current layers in training
    for param in model.parameters():
        param.requires_grad = False

    model_update = nn.Sequential(model, models.vgg19(True).features[7:])


def new_model():
    return nn.Sequential(
        torch.nn.Conv2d(128, 19, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),

    )

def init_model(model_path, cuda=False):
    """

    :param model_path:
    :param bool cuda:
    :return:
    """
    model = construct_model(model_path)  # This constructs the original model
    if cuda:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)





    model2 = nn.Sequential(model)
    print(model2)

    x = torch.randn(3, 3, 128, 128).requires_grad_(True)
    with torch.onnx.set_training(model2, False):
        trace, _ = torch.jit.get_trace_graph(model2, args=(x,))
    dot = make_dot_from_trace(trace)
    dot.format = 'svg'
    dot.render('C:/Users/Brandon/OneDrive/UCF/Fall_2018/Computer_Vision/Project/EXP1/Evaluate/test2.svg', view=True)

    wait = input("enter")

    return model