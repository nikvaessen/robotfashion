# import necessary libraries
from PIL import Image
from train_epoch import train_one_epoch, evaluate

import matplotlib.pyplot as plt
import torch
import transforms as T
import torchvision.utils
import torchvision
import torch
import numpy as np
import cv2
import random
import utils
import coco_utils

from config import *
from torch import nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

#################################################################
# Main function for creating and training faster RCNN
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
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        collate_fn=utils.collate_fn)

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
    
    
def get_model_bbox_detection(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

