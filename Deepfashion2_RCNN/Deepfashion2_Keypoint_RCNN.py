from train_RCNN import train_RCNN, get_model_instance_segmentation, get_model_keypoint_detection, get_model_keypoint_detection_custom
from myconfig import *
import myutils
import torch
from eval_RCNN import *
#################################################################
# Main function for Keypoint RCNN training and evaluation
#################################################################

def main():
    # get the model using our helper function
    model = get_model_keypoint_detection(num_classes, num_keypoints)
    
    if not inference_only:
        train_RCNN(model, path2data, path2json, weight_path)
    else:
        eval_RCNN(model, keypoint_detection_api)
    print("That's it!")


if __name__ == "__main__":
    main()


#################################################################
# To configure the parameters, modify myconfig.py file
#################################################################
