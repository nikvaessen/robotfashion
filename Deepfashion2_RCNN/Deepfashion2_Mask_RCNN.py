from train_RCNN import train_RCNN, get_model_instance_segmentation, get_model_keypoint_detection
from myconfig import *
import myutils
import torch
from eval_RCNN import *
#################################################################
# Main function for Mask(Segments) RCNN training and evaluation
#################################################################

def main():
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
   
    if not inference_only:
        train_RCNN(model, path2data, path2json, weight_path)
    else:
        eval_RCNN(model, instance_segmentation_api)
    print("That's it!")


if __name__ == "__main__":
    main()


#################################################################
# To configure the parameters, modify myconfig.py file
#################################################################