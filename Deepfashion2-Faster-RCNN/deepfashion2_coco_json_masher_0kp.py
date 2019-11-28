# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:15:50 2019

@author: loktarxiao
"""

import json
import os
import re
from PIL import Image
import numpy as np
from tqdm import tqdm

#################################################################
# This function is supposed to pack deepfashion 2 annotations
# into a single coco style annotation file
#################################################################

dataset = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

dataset['categories'].append({
    'id': 1,
    'name': "short_sleeved_shirt",
    'supercategory': "clothes",
    'skeleton': []
})
dataset['categories'].append({
    'id': 2,
    'name': "long_sleeved_shirt",
    'supercategory': "clothes",
    'skeleton': []
})
dataset['categories'].append({
    'id': 3,
    'name': "short_sleeved_outwear",
    'supercategory': "clothes",
    'skeleton': []
})
dataset['categories'].append({
    'id': 4,
    'name': "long_sleeved_outwear",
    'supercategory': "clothes",
    'skeleton': []
})
dataset['categories'].append({
    'id': 5,
    'name': "vest",
    'supercategory': "clothes",
    'skeleton': []
})
dataset['categories'].append({
    'id': 6,
    'name': "sling",
    'supercategory': "clothes",
    'skeleton': []
})
dataset['categories'].append({
    'id': 7,
    'name': "shorts",
    'supercategory': "clothes",
    'skeleton': []
})
dataset['categories'].append({
    'id': 8,
    'name': "trousers",
    'supercategory': "clothes",
    'skeleton': []
})
dataset['categories'].append({
    'id': 9,
    'name': "skirt",
    'supercategory': "clothes",
    'skeleton': []
})
dataset['categories'].append({
    'id': 10,
    'name': "short_sleeved_dress",
    'supercategory': "clothes",
    'skeleton': []
})
dataset['categories'].append({
    'id': 11,
    'name': "long_sleeved_dress",
    'supercategory': "clothes",
    'skeleton': []
})
dataset['categories'].append({
    'id': 12,
    'name': "vest_dress",
    'supercategory': "clothes",
    'skeleton': []
})
dataset['categories'].append({
    'id': 13,
    'name': "sling_dress",
    'supercategory': "clothes",
    'skeleton': []
})

num_images = 10

sub_index = 0 # the index of ground truth instance
    ############ PATH TO IMAGES IS SPECIFIED HERE ############
images_to_eval = [f for f in os.listdir('../set12') if re.match(r'.+\.(jpg|jpeg|png)', f)]
for num in tqdm(range(len(images_to_eval))):
    raw_name = re.search("(.+)\.(jpg|jpeg|png)", images_to_eval[num]).group(1)
    ext = re.search("(.+)\.(jpg|jpeg|png)", images_to_eval[num]).group(2)
    json_name = './annos/json/' + raw_name + '.json'
    image_name = './annos/img/' + raw_name + '_annotated.' + ext
    ##########################################################
    if (num>=0):
        imag = Image.open(image_name)
        width, height = imag.size
        with open(json_name, 'r') as f:
            temp = json.loads(f.read())
            pair_id = temp['pair_id']
            f_name = images_to_eval[num]
            image_url = "http://localhost:8000/" + f_name
            dataset['images'].append({
                'coco_url': '',
                'date_captured': '',
                'file_name': f_name,
                'flickr_url': '',
                'id': num,
                'license': 0,
                'width': width,
                'height': height,
                'url': image_url
            })
            for i in temp:
                if i == 'source' or i=='pair_id':
                    continue
                else:
                    points = np.zeros(294 * 3)
                    sub_index = sub_index + 1
                    box = temp[i]['bounding_box']
                    w = box[2]-box[0]
                    h = box[3]-box[1]
                    x_1 = box[0]
                    y_1 = box[1]
                    bbox=[x_1,y_1,w,h]
                    cat = temp[i]['category_id']
                    style = temp[i]['style']

                    dataset['annotations'].append({
                        'area': w*h,
                        'bbox': bbox,
                        'category_id': cat,
                        'id': sub_index,
                        'pair_id': pair_id,
                        'image_id': num,
                        'iscrowd': 0,
                        'style': style,
                    })


json_name = './coco_annos_split.json'
with open(json_name, 'w') as f:
  json.dump(dataset, f)




