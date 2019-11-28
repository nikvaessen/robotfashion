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
path2data = "../../data/image/"
path2json = "./valid.json"

dp1_PATH = "../"

# DO NOT change this one, uncomment the next one in case of pretrained weights
weight_path = None
# uncomment to load pretrained weights
weight_path = "../deepfashion2_294kp_res_2.pth"
# specify where to save the resulting weights
save_weights_to = "./deepfashion_rcnn_trained.pth"

# number of epochs to train
num_epochs = 5

# decide whether we want to use the model to train or to infer
inference_only = True
# configure to whether cuda
use_cuda = False
# configure whether we have a GUI
is_interactive = False
# configure images that must be evaluated
images_to_eval = ['./' + f for f in os.listdir('.') if re.match(r'.+\.(jpg|jpeg)', f)]
# specify where to save the resulting annotations
save_annos_dir = "./annos/"