import torchvision.models as models
from torchvision.models import vgg19, alexnet, resnet152, resnet18, resnet101
from torch import nn, load
from collections import OrderedDict


def get_model(classifier, pretrained=False, weight_path=None, cuda=True):
    """
    :param str classifier:
    :param bool pretrained:
    :param str weight_path:
    :param bool cuda:
    :return:
    """
    assert classifier in ['resnet101', 'resnet18', 'vgg', 'alexnet'], "Please provide a model: resnet, vgg, alexnet"
    if classifier == 'resnet152':
        model = resnet152(pretrained=pretrained)
    if classifier == 'resnet101':
        model = resnet101(pretrained=pretrained)
    if classifier == 'resnet18':
        model = resnet18(pretrained=pretrained)
    if classifier == 'vgg':
        model = vgg19(pretrained=pretrained)
    if classifier ==  'alexnet':
        model = alexnet(pretrained=pretrained)
    # Don't update non-classifier learned features in the pretrained networks
    model.fc = nn.Linear(2048, 2)
    if weight_path is not None:
        state_dict = load(weight_path)['state_dict']
    if cuda:
        model = nn.DataParallel(model).cuda()
        if weight_path is not None:
            model.load_state_dict(state_dict)
    else:
        model = model.to('cpu')
        if weight_path is not None:
            new_state_dict = OrderedDict()
            for key in state_dict.keys():
                new_state_dict[key.replace('module.', '')] = state_dict[key]
            model.load_state_dict(new_state_dict)
    model.eval()
    return model