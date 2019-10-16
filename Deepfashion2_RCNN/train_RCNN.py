# import necessary libraries
from PIL import Image
from train_epoch import train_one_epoch, evaluate

import matplotlib.pyplot as plt
import torch
import mytransforms as T
import torchvision.utils
import torchvision
import torch
import numpy as np
import cv2
import random
import myutils
import coco_utils

from myconfig import *
from torch import nn

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.rpn import AnchorGenerator

#################################################################
# Main function for creating and training (Mask/Keypoint) RCNN
#################################################################


def train_RCNN(model, path2data, path2json, weight_path = None):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    # see if pretrained weights are available
    load_pretrained = False
    if weight_path is not None:
        load_pretrained = True
    # get coco style dataset
    dataset = coco_utils.get_coco(path2data, path2json, T.ToTensor())
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-1])
    dataset_test = torch.utils.data.Subset(dataset, indices[-1:])

    # define training and validation data loaders(use num_workers for multi-gpu)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        collate_fn=myutils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        collate_fn=myutils.collate_fn)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    # load the dataset in case of pretrained weights
    start_epoch = 0
    if load_pretrained:
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch + start_epoch, print_freq=100)
        # update the learning rate
        #lr_scheduler.step()
        # evaluate on the test dataset
        # Find a way around the broken pytorch nograd keypoint evaluation
        # evaluate(model, data_loader_test, device=device)

        # save weights when done
        torch.save({
                'epoch': num_epochs + start_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_weights_to)
    
    
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def get_model_keypoint_detection(num_classes, num_keypoints):
    # load a keypoiny detection model pre-trained on COCO
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_keypoint = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels

    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features_keypoint, num_keypoints)
    return model

def get_model_keypoint_detection_custom(num_classes, num_keypoints):
    #backbone = myutils.VGG16()
    backbone = torchvision.models.vgg16().features
    # KeypointRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    

    ################# USE PRETRAINED DEEP FASHION 1 #################
    pretrained_dict = torch.load(dp1_PATH + 'stage3_hard.pt')
    pretrained_dict2 = torch.load(dp1_PATH +'fashion_detector.pt')
    model_dict = backbone.state_dict()
    # 0. rename keys to use in torchvision
    pretrained_dict['0.weight'] = pretrained_dict.pop('conv1_1.weight')
    pretrained_dict['2.weight'] = pretrained_dict.pop('conv1_2.weight')
    pretrained_dict['5.weight'] = pretrained_dict.pop('conv2_1.weight')
    pretrained_dict['7.weight'] = pretrained_dict.pop('conv2_2.weight')
    pretrained_dict['10.weight'] = pretrained_dict.pop('conv3_1.weight')
    pretrained_dict['12.weight'] = pretrained_dict.pop('conv3_2.weight')
    pretrained_dict['14.weight'] = pretrained_dict.pop('conv3_3.weight')
    pretrained_dict['17.weight'] = pretrained_dict.pop('conv4_1.weight')
    pretrained_dict['19.weight'] = pretrained_dict.pop('conv4_2.weight')
    pretrained_dict['21.weight'] = pretrained_dict.pop('conv4_3.weight')
    pretrained_dict['24.weight'] = pretrained_dict.pop('conv5_1.weight')
    pretrained_dict['26.weight'] = pretrained_dict.pop('conv5_2.weight')
    pretrained_dict['28.weight'] = pretrained_dict.pop('conv5_3.weight')
    pretrained_dict['31.weight'] = pretrained_dict.pop('fc6.weight')

    pretrained_dict['0.bias'] = pretrained_dict.pop('conv1_1.bias')
    pretrained_dict['2.bias'] = pretrained_dict.pop('conv1_2.bias')
    pretrained_dict['5.bias'] = pretrained_dict.pop('conv2_1.bias')
    pretrained_dict['7.bias'] = pretrained_dict.pop('conv2_2.bias')
    pretrained_dict['10.bias'] = pretrained_dict.pop('conv3_1.bias')
    pretrained_dict['12.bias'] = pretrained_dict.pop('conv3_2.bias')
    pretrained_dict['14.bias'] = pretrained_dict.pop('conv3_3.bias')
    pretrained_dict['17.bias'] = pretrained_dict.pop('conv4_1.bias')
    pretrained_dict['19.bias'] = pretrained_dict.pop('conv4_2.bias')
    pretrained_dict['21.bias'] = pretrained_dict.pop('conv4_3.bias')
    pretrained_dict['24.bias'] = pretrained_dict.pop('conv5_1.bias')
    pretrained_dict['26.bias'] = pretrained_dict.pop('conv5_2.bias')
    pretrained_dict['28.bias'] = pretrained_dict.pop('conv5_3.bias')
    pretrained_dict['31.bias'] = pretrained_dict.pop('fc6.bias')
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    backbone.load_state_dict(model_dict)
    #################################################################
    backbone.out_channels = 512

    # put the pieces together inside a FasterRCNN model
    model = KeypointRCNN(backbone,
                        num_classes=num_classes,
                        num_keypoints=num_keypoints)
    return model
