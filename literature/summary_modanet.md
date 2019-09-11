--------------------------------------------------------------------------------
<details>
<summary>ModaNet: A Large-scale Street Fashion Dataset with Polygon Annotations</summary>
<br>

#### Links
[Link to paper](./pdf/eBayModanet.pdf)  
Link to repository - https://github.com/eBay/modanet

#### Summary
 * 55, 176 street images, fully annotated with polygons
 * 1 million weakly annotated street images in Paperdoll
 * Previously available datasets has not many pose variations, hence work well only in limited scenarios.
 * Does not take occlusion into account.
 * provide baseline results on FCNs, CRFasRNN, DeepLabv3+, DeepLab and Polygon-RNN++
 
#### Tasks
 1. object detection
 2. semantic segmentation
 3.  polygon prediction

#### Dataset details

* classified using ResNet followed by human annotation, to allow only one person images.
* not a good fit for occlusion related tasks.
* polygon annotations for individual objects in the image and assign
a label from a pre-defined set of fashion object categories
	
#### read by
* Deepika
</details>
