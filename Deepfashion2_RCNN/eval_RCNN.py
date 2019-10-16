import matplotlib
import matplotlib.pyplot as plt

import mytransforms as T
import torch
import numpy as np
import cv2
import random
import datetime
import pickle
import time
import errno
import os
import re
import json
import itertools
import myutils
from myconfig import *

#############################################################################
# Main function for testing and evaluating images using (Mask/Keypoint) RCNN
#############################################################################

def eval_RCNN(model, api):
    if weight_path is not None:
        device = None
        if use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if use_cuda:
        model = model.cuda()
    #Instance segmentation api to illustrate masks
    for img in images_to_eval:
        api(model, img, COCO_INSTANCE_CATEGORY_NAMES, 0.75)

def instance_segmentation_api(model, img_path, cat_names, threshold=0.5, rect_th=3, text_size=1, text_th=2):
    masks, boxes, pred_cls, pred_id = myutils.get_prediction(model, img_path, cat_names, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = myutils.random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    if is_interactive:
        matplotlib.use('TkAgg')
        plt.figure(figsize=(20,30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show(block=True)
    save_annos([[[]] for i in range(len(boxes))], [masks], boxes, pred_cls, pred_id, img_path, img)

def keypoint_detection_api(model, img_path, cat_names, threshold=0.5, rect_th=3, text_size=1, text_th=2):
    keypoints, boxes, pred_cls, pred_id = myutils.get_prediction(model, img_path, cat_names, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    keypoints_dest = []
    for i in range(len(keypoints)):
        keypoints_viz = []
        keypoints_show = keypoints[i]
        if num_keypoints == 294:
            keypoints_show = classify_keypoints(keypoints_show, pred_id[i])
        for point in keypoints_show:
            if point[2] > 0.9:
                keypoints_viz.append([point[0], point[1]])
        rgb_col = colours[random.randrange(0,10)]
        for point_viz in keypoints_viz:
            cv2.circle(img, tuple(point_viz), 8, tuple(rgb_col), -1)
        for j in range(len(keypoints_viz) - 1):
            cv2.line(img, tuple(keypoints_viz[j]), tuple(keypoints_viz[j + 1]) , tuple(rgb_col), 2)
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        keypoints_dest.append(keypoints_show)
    if is_interactive:
        matplotlib.use('TkAgg')
        plt.figure(figsize=(20,30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show(block=True)
    save_annos(keypoints_dest, [ [[[]]] for i in range(len(boxes))], boxes, pred_cls, pred_id, img_path, img)

def save_annos(keypoints, masks, boxes, pred_cls, pred_id, img_path, img):
    annos = {"source": "user", "pair_id": 1}
    for i in range(len(boxes)):
        masks[i] = list(itertools.chain.from_iterable(masks[i]))
        boxes[i] = list(itertools.chain.from_iterable(boxes[i]))
        keypoints[i] = list(itertools.chain.from_iterable(keypoints[i]))
        keypoints[i] = [int(val) for val in keypoints[i]]
        for j in range(len(masks[i])):
            masks[i][j] = [int(val) for val in masks[i][j]]
        boxes[i] = [int(val) for val in boxes[i]]
        item_data = {"segmentation": masks[i], "scale": 1, "viewpoint": 2,
                    "zoom_in": 1, "style": 1, "occlusion": 2, "landmarks": keypoints[i],
                    "bounding_box": boxes[i], "category_id": int(pred_id[i]), "category_name": pred_cls[i]}
        annos.update({"item" + str(i + 1): item_data})
    json_name = re.search(".+/(.+)\.(jpg|jpeg)", img_path).group(1)
    ext = re.search(".+/(.+)\.(jpg|jpeg)", img_path).group(2)
    if not os.path.exists(save_annos_dir + 'json/'):
        os.makedirs(save_annos_dir + 'json/')
    if not os.path.exists(save_annos_dir + 'img/'):
        os.makedirs(save_annos_dir + 'img/')
    with open(save_annos_dir + 'json/' + json_name + ".json", 'w') as f:
        json.dump(annos, f)
    plt.imsave(save_annos_dir + 'img/' + json_name + '_annotated.' + ext, img)

def classify_keypoints(keypoints_show, pred_id):
    if pred_id == 1:
        keypoints_f = keypoints_show[0:25]
    elif pred_id ==2:
        keypoints_f = keypoints_show[25:58]
    elif pred_id ==3:
        keypoints_f = keypoints_show[58:89]
    elif pred_id == 4:
        keypoints_f = keypoints_show[89:128]
    elif pred_id == 5:
        keypoints_f = keypoints_show[128:143]
    elif pred_id == 6:
        keypoints_f = keypoints_show[143:158]
    elif pred_id == 7:
        keypoints_f = keypoints_show[158:168]
    elif pred_id == 8:
        keypoints_f = keypoints_show[168:182]
    elif pred_id == 9:
        keypoints_f = keypoints_show[182:190]
    elif pred_id == 10:
        keypoints_f = keypoints_show[190:219]
    elif pred_id == 11:
        keypoints_f = keypoints_show[219:256]
    elif pred_id == 12:
        keypoints_f = keypoints_show[256:275]
    elif pred_id == 13:
        keypoints_f = keypoints_show[275:294]
    return keypoints_f