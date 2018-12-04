import torchvision.models as models
from torch import nn, load
from collections import OrderedDict


def get_model(weight_path='classifier_utils/model_best.pth.tar', cuda=True):
        model = models.__dict__["resnet152"](pretrained=False)
        # Don't update non-classifier learned features in the pretrained networks
        model.fc = nn.Linear(2048, 2)
        state_dict = load(weight_path)['state_dict']
        if cuda:
            model = nn.DataParallel(model).cuda()
            model.load_state_dict(state_dict)
        else:
            model = model.to('cpu')
            new_state_dict = OrderedDict()
            for key in state_dict.keys():
                new_state_dict[key.replace('module.', '')] = state_dict[key]
            model.load_state_dict(new_state_dict)
        model.eval()
        return model