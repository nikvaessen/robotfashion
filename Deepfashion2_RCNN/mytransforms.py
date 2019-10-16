import random
import torch

from torchvision.transforms import functional as F

#################################################################
# Transformations that can be passed to pycocotools for software
# augmentation. Transformations have to be created as objects
#################################################################

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            if target is None:
                image = t(image, target)
            else:
                image, target = t(image, target)
        if target is None:
            return image
        else:
            return image, target

class ToTensor(object):
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is None:
            return image
        else:
            return image, target
