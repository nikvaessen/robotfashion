import matplotlib
import matplotlib.pyplot as plt

import transforms as T
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
import utils
from config import *

#############################################################################
# Main function for testing and evaluating images using Faster RCNN
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
    for img in images_to_eval:
        try:
            print("annotating: " + img)
            api(model, img, COCO_INSTANCE_CATEGORY_NAMES, detection_confidence)
        except:
            print("oops! No matches found in: " + img)

def instance_bbox_api(model, img_path, cat_names, threshold=0.5, rect_th=3, text_size=1, text_th=2):
    boxes, pred_cls, pred_id = utils.get_prediction(model, img_path, cat_names, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    if is_interactive:
        matplotlib.use('TkAgg')
        plt.figure(figsize=(20,30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show(block=True)
    save_annos(boxes, pred_cls, pred_id, img_path, img)

def save_annos(boxes, pred_cls, pred_id, img_path, img):
    annos = {"source": "user", "pair_id": 1}
    for i in range(len(boxes)):
        boxes[i] = list(itertools.chain.from_iterable(boxes[i]))
        boxes[i] = [int(val) for val in boxes[i]]
        item_data = {"scale": 1, "viewpoint": 2,
                    "zoom_in": 1, "style": 1, "occlusion": 2,
                    "bounding_box": boxes[i], "category_id": int(pred_id[i]), "category_name": pred_cls[i]}
        annos.update({"item" + str(i + 1): item_data})
    json_name = re.search(".+/(.+)\.(jpg|jpeg|png)", img_path).group(1)
    ext = re.search(".+/(.+)\.(jpg|jpeg|png)", img_path).group(2)
    if not os.path.exists(save_annos_dir + 'json/'):
        os.makedirs(save_annos_dir + 'json/')
    if not os.path.exists(save_annos_dir + 'img/'):
        os.makedirs(save_annos_dir + 'img/')
    with open(save_annos_dir + 'json/' + json_name + ".json", 'w') as f:
        json.dump(annos, f)
    plt.imsave(save_annos_dir + 'img/' + json_name + '_annotated.' + ext, img)
