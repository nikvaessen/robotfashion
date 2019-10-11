from train_RCNN import train_RCNN, get_model_instance_segmentation, get_model_keypoint_detection
from myconfig import *
import myutils

#################################################################
# Main function for Keypoint RCNN training and evaluation
#################################################################

def main():
    # get the model using our helper function
    model = myutils.get_model_keypoint_detection(num_classes, num_keypoints)
    
    if not inference_only:
        train_RCNN(model, path2data, path2json, weight_path)
    else:
        model.eval()
        # We should replace this line with keypoint detection api
        #myutils.instance_segmentation_api(model, './penguins.jpg', COCO_INSTANCE_CATEGORY_NAMES, 0.75)
    print("That's it!")


if __name__ == "__main__":
    main()


#################################################################
# To configure the parameters, modify myconfig.py file
#################################################################