--------------------------------------------------------------------------------
<details>
<summary>CTU Color and Depth Image Dataset of Spread Garments</summary>
<br>

#### Links
[Link to paper](./pdf/ctu_color_depth.pdf)  
[Link to repository]((https://github.com/CloPeMa/garment_dataset)

#### Summary
 * contains colour and depth images of spread garments
 * Color images - segmentation, recognition and model fitting
 * Depth images - wrinkle detection and spreading strategy estimation.
#### Purpose
 Clothes PErception and Manipulation. 

#### Tasks
 1. cloth segmentation
 2. garment recognition
 3. FOLD DETECTION
 4. model fitting

#### Dataset details
## Part I - Segmentation, recognition and model fitting
* 1. Color and Depth images of 17 garments in different configurations.
* 2. Ground Truth - Class of garments, state of garments (folded, wrinkled or flat) and front/back facing.
* 3. Each sample has:
	 * color image - resolution 1280 x 1024 pixels -  3 x 8 bits channel
	 * depth map - resolution 640 x 480 pixels - one 16 bit channel
	 * annotation file - .yaml file with ground truth variables and path to color and depth, name and position of corners and moves to change state of model.

## Part II - Folded Garments
*1. classified to 9 classes.
*2. keypoints along the outline of the garment.
#### read by
* Deepika
</details>
