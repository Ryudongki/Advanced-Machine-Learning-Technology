
# coding: utf-8

# In[ ]:

import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data as data
import torch

class ImageFeatureFolder(dset.ImageFolder):
    def __init__(self, root, landmark_file, imageSize):
        super(ImageFeatureFolder, self).__init__(root=root)
        self.feature_input_transform = transforms.Compose([
            transforms.CenterCrop((112, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.net_input_transform = transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        with open(landmark_file, 'r') as f:
            data = f.read()
            
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
                
        return self.net_input_transform(img), self.feature_input_transform(img)#, self.attrs[index]

