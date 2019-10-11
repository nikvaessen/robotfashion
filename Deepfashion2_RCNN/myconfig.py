
# Category names of instances passed over as global variable
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling',
    'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress'
]

# decide whether we want to use the model to train or to infer
inference_only = False
# specify the number of classes and keypoints
num_keypoints = 294
num_classes = 14
# define path to our dataset and defined annotations
path2data = "../../data/image"
path2json = "../../coco_annotations/keypoints_val_vis_segs.json"
weight_path = None

# uncomment to load pretrained weights
#weight_path = "./deepfashion_keypoints_trained.pth"

# number of epochs to train
num_epochs = 5
