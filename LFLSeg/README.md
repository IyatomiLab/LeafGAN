## Implementation of the LFLSeg Module for Segmenting Leaf Area

![Teaser image](Supplement_LFLSeg.png)
The above image shows the heatmap visualization of our LFLSeg on different scenariors (full leaf, non-leaf, partial leaf).
These heatmaps can be used as useful segmentation masks for training our LeafGAN model without the need of pixel-label data.
LFLSeg works well on different in-ﬁeld images with complex backgrounds.
However, when the images contain multiple and overlapping leaves, the LFLSeg fails to correctly segment the leaf area (last image of the "full leaf" cases).
Also, LFLSeg may incorrectly detect "partial leaf" as "full leaf" if the "partial leaf" image has a different shooting distance than images in our training dataset (last column).
### Suggestion
Applying data augmentation techniques such as random resize/scale is expected to increase the robustness of this module and thus, boosting the performance of the system.

## Datasets
Dataset will have 3 classes:
- full leaf: image that contains a full leaf
- partial leaf: image that contains a part of a full leaf
- non-leaf: image that does not contain any part of a leaf

For more details about how to create dataset, please refer to our paper.

Please provide a `.txt` file that contain training info with label as follow:
- full leaf: label 0
- partial leaf: label 1
- non-leaf: label 2
```
/path/to/full_leaf/full_leaf_1.JPG, 0
/path/to/full_leaf/full_leaf_2.png, 0
/path/to/full_leaf/full_leaf_3.jpg, 0
... ... ...
/path/to/partial_leaf/partial_leaf_1.JPG, 1
/path/to/partial_leaf/partial_leaf_2.jpg, 1
/path/to/partial_leaf/partial_leaf_3.png, 1
... ... ...
/path/to/non_leaf/non_leaf_1.png, 2
/path/to/non_leaf/non_leaf_2.jpg, 2
/path/to/non_leaf/non_leaf_3.JPG, 2
```

## Train LFLSeg

```bash
python train_LFLSeg.py --train /path/to/train_data.txt --test /path/to/train_data.txt
```

After training, please replace the trained model path at line 90 of the [leaf_gan_model.py](https://github.com/IyatomiLab/LeafGAN/blob/master/models/leaf_gan_model.py#L90)
```
load_path = '/path/to/LFLSeg_model.pth'
```

## Test LFLSeg

Coming soon ...

## Citation

```
@article{cap2020leafgan,
  title   = {LeafGAN: An Effective Data Augmentation Method for Practical Plant Disease Diagnosis},
  author  = {Quan Huu Cap and Hiroyuki Uga and Satoshi Kagiwada and Hitoshi Iyatomi},
  journal = {CoRR},
  volume  = {abs/2002.10100},
  year    = {2020},
}
```