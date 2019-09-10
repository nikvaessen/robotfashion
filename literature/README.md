# Literature research

## Research  articles

Some important/relevant papers:

--------------------------------------------------------------------------------
<details>
<summary>DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations</summary>
<br>

### Links
[link to paper](./pdf/deepfashion.pdf)  
[link to repository](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

### Summary
Present a novel dataset with 800k images, labeled with:
* categories (50)
* attributes (1000)
* land mark bounding boxes of key points in clothing (4~8 per image)
* cross-domain/cross-pose image pairs (300k)

Dataset contains:
* images of professional photo shoot for retail (in-shop)
* images of normal people in the clothes, such as selfies (consumer)

Design a state-of-the-art architecture called FashionNet, which combines the task of attribute prediction and landmark prediction in 1 network.   

This dataset proposes three benchmarks:
1. clothing category and attribute prediction
    * category prediction uses top-k classification accuracy
    * attribute prediction uses top-k recall rate
2. in-shop clothes retrieval
    * decide whether two images the same clothing item
    * only includes "nice" in-shop images
    * metric: top-k retrieval accuracy (retrieval = exact item in top-k results)
3. cross-domain clothes retrieval
    * same as 2, but match consumer picture to shopping picture

### Tasks
* clothing category and attribute prediction
* in-shop clothes retrieval
* cross-domain clothes retrieval

### Datasets
* DeepFashion

### read by
* Nik
</details>

--------------------------------------------------------------------------------
<details>
<summary>DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images</summary>
<br>

#### Links
[Link to paper](./pdf/deepfashion2.pdf)  
[Link to repository](https://github.com/switchablenorms/DeepFashion2 )

#### Summary

Preset a novel dataset with 801k clothing items, where each item has annotations for:
* 13 categories (less ambiguous than deepfashion 1 )
* bounding boxes
* pose for each landmark, which has set of landmarks plus contours and skeleton between landmarks
* per-pixel map over clothing item
* style label

#### Tass
* clothes detection (bounding box + category label)
* landmark estimation
* segmentation
* commercial-consumer clothes retrieval

#### Datasets
* DeepFashion2

#### read by
* Nik
</details>

--------------------------------------------------------------------------------
<details>
<summary>Glasgow's Stereo Image Database of Garments</summary>
<br>

#### Links
[Link to paper](./pdf/glasgow_database.pdf)  
[Link to repository](https://sites.google.com/site/ugstereodatabase/)

#### Summary
This is a summary of the paper

#### Tasks
* 1
* 2
* 3

#### Datasets
* 1

#### read by
* 1
</details>

--------------------------------------------------------------------------------
<details>
<summary>CTU color and depth image dataset of spread
garments</summary>
<br>

#### Links
[Link to paper](./pdf/ctu_color_depth.pdf)  
[Link to repository](https://github.com/CloPeMa/garment_dataset)

#### Summary
This is a summary of the paper

#### Tasks
* 1
* 2
* 3

#### Datasets
* 1

#### read by
* 1
</details>

--------------------------------------------------------------------------------
process:  
http://vision.is.tohoku.ac.jp/~kyamagu/research/clothing_parsing/  
http://mvc-datasets.github.io/MVC/you  
https://github.com/eBay/modanet  
https://labicvl.github.io/docs/pubs/Andreas_ICRA_2014.pdf  



## Datasets

## Tasks

Tasks related to clothing.
