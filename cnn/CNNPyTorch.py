from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
import torch
from torch import nn
import torchvision.models as models
import os
from config.CONFIG import Config

cfg = Config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class CNN(nn.Module):
    def __init__(self, classifier_type, is_load=True):
        self.data = None
        # self.model = None
        self.model = models.resnet152(pretrained=False)
        model_weight_path = 'resnet152-b121ed2d.pth'
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        self.model.load_state_dict(torch.load(model_weight_path))
        self.classifier_type = classifier_type
        self.image_shape = (32, 32, 3)
        self.class_number = None
        if is_load:
            self.load(classifier_type)
        # self.model=

    def setmodel(self, epoch_num, is_compile=True):
        net = models

        # base_model =
resnet=models.resnet152(pretrained=False)
print(resnet)
model_weight_path = 'resnet152-b121ed2d.pth'
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
resnet=resnet.load_state_dict(torch.load(model_weight_path))
# print(resnet.children())