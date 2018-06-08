from torchvision import models
import torch.nn as nn
class VGG16Feature(nn.Module):
    def __init__(self):
        super(VGG16Feature, self).__init__()
        
        origin = models.vgg16(pretrained=True)
        features = origin.features
        childrens = list(features.children())
        self.feature1 = nn.Sequential(*childrens[:5])#nn.Sequential(*childrens[:7])
        self.feature2 = nn.Sequential(*childrens[5:10])#nn.Sequential(*childrens[7:14])
        self.feature3 = nn.Sequential(*childrens[10:17])#nn.Sequential(*childrens[14:24])
        self.feature4 = nn.Sequential(*childrens[17:24])#nn.Sequential(*childrens[24:34])
        self.feature5 = nn.Sequential(*childrens[24:])#nn.Sequential(*childrens[34:])
        self.features = [self.feature1, self.feature2, self.feature3, self.feature4, self.feature5]
        
    def forward(self, x):
        ret_list = []
        for feature in self.features:
            x = feature(x)
            ret_list.append(x)
            
        return ret_list