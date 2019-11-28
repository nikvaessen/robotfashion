
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:15:50 2019

@author: loktarxiao
"""

import json
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
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})
dataset['categories'].append({
    'id': 2,
    'name': "long_sleeved_shirt",
    'supercategory': "clothes",
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})
dataset['categories'].append({
    'id': 3,
    'name': "short_sleeved_outwear",
    'supercategory': "clothes",
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})
dataset['categories'].append({
    'id': 4,
    'name': "long_sleeved_outwear",
    'supercategory': "clothes",
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})
dataset['categories'].append({
    'id': 5,
    'name': "vest",
    'supercategory': "clothes",
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})
dataset['categories'].append({
    'id': 6,
    'name': "sling",
    'supercategory': "clothes",
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})
dataset['categories'].append({
    'id': 7,
    'name': "shorts",
    'supercategory': "clothes",
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})
dataset['categories'].append({
    'id': 8,
    'name': "trousers",
    'supercategory': "clothes",
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})
dataset['categories'].append({
    'id': 9,
    'name': "skirt",
    'supercategory': "clothes",
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})
dataset['categories'].append({
    'id': 10,
    'name': "short_sleeved_dress",
    'supercategory': "clothes",
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})
dataset['categories'].append({
    'id': 11,
    'name': "long_sleeved_dress",
    'supercategory': "clothes",
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})
dataset['categories'].append({
    'id': 12,
    'name': "vest_dress",
    'supercategory': "clothes",
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})
dataset['categories'].append({
    'id': 13,
    'name': "sling_dress",
    'supercategory': "clothes",
    'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    'skeleton': []
})

num_images = 30000

sub_index = 0 # the index of ground truth instance
for num in tqdm(range(1, num_images+1)):
    ############ PATH TO IMAGES IS SPECIFIED HERE ############
    json_name = '../data/annos/' + str(num).zfill(6)+'.json'
    image_name = '../data/image/' + str(num).zfill(6)+'.jpg'
    ##########################################################
    if (num>=0):
        imag = Image.open(image_name)
        width, height = imag.size
        with open(json_name, 'r') as f:
            temp = json.loads(f.read())
            pair_id = temp['pair_id']

            dataset['images'].append({
                'coco_url': '',
                'date_captured': '',
                'file_name': str(num).zfill(6) + '.jpg',
                'flickr_url': '',
                'id': num,
                'license': 0,
                'width': width,
                'height': height
            })
            for i in temp:
                if i == 'source' or i=='pair_id':
                    continue
                else:
                    points = np.zeros(39 * 3)
                    sub_index = sub_index + 1
                    box = temp[i]['bounding_box']
                    w = box[2]-box[0]
                    h = box[3]-box[1]
                    x_1 = box[0]
                    y_1 = box[1]
                    bbox=[x_1,y_1,w,h]
                    cat = temp[i]['category_id']
                    style = temp[i]['style']
                    seg = temp[i]['segmentation']
                    landmarks = temp[i]['landmarks']

                    points_x = landmarks[0::3]
                    points_y = landmarks[1::3]
                    points_v = landmarks[2::3]
                    points_x = np.array(points_x)
                    points_y = np.array(points_y)
                    points_v = np.array(points_v)

                    if cat == 1:
                        for n in range(25):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat ==2:
                        for n in range(33):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat ==3:
                        for n in range(31):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat == 4:
                        for n in range(39):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat == 5:
                        for n in range(15):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat == 6:
                        for n in range(15):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat == 7:
                        for n in range(10):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat == 8:
                        for n in range(14):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat == 9:
                        for n in range(8):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat == 10:
                        for n in range(29):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat == 11:
                        for n in range(37):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat == 12:
                        for n in range(19):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat == 13:
                        for n in range(19):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    num_points = len(np.where(points_v > 0)[0])

                    dataset['annotations'].append({
                        'area': w*h,
                        'bbox': bbox,
                        'category_id': cat,
                        'id': sub_index,
                        'pair_id': pair_id,
                        'image_id': num,
                        'iscrowd': 0,
                        'style': style,
                        'num_keypoints':num_points,
                        'keypoints':points.tolist(),
                        'segmentation': seg,
                    })


json_name = './valid_compact.json'
with open(json_name, 'w') as f:
  json.dump(dataset, f)




