import re
import os
# Category names of instances passed over as global variable
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling',
    'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress'
]
# colors that are used for annotations
colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]

# specify the number of classes and keypoints
num_keypoints = 294
num_classes = 14
# define path to our dataset and defined annotations
path2data = "../set8/"
path2json = "./receipts_valid.json"

dp1_PATH = "../"

# DO NOT change this one, uncomment the next one in case of pretrained weights
weight_path = None
# uncomment to load pretrained weights
#weight_path = "../deepfashion2_rcnn_trained_2.pth"
# specify where to save the resulting weights
#save_weights_to = "./deepfashion2_rcnn_trained.pth"

# number of epochs to train
num_epochs = 5

# decide whether we want to use the model to train or to infer
inference_only = False
# configure to whether cuda
use_cuda = True
# configure whether we have a GUI
is_interactive = False
# configure images that must be evaluated
root_dir = '.'

images_to_eval = [root_dir + '/' + f for f in os.listdir(root_dir) if re.match(r'.+\.(jpg|jpeg|png)', f)]
# specify where to save the resulting annotations
save_annos_dir = "./annos/"
# specify the confidence interval of detections
detection_confidence = 0.5