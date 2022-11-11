# SphKpt
SphKpt is a uniform framework for detecting keypoints on both omnidirectional and perspective images.

## Dependencies
pip install -r requirements.txt

## Train the framework
* Run pretrain/vp_pretrain.py to pre-train the perspective module.
* Run pretrain/erp_pretrain.py to pre-train the spherical module.
* run train/train.py to train the two modules jointly.

## Test
* Run test/test.py to detect the keypoints.

## Datasets
* PanoContext Dataset

Please download the PanoContext dataset from this [link](https://panocontext.cs.princeton.edu).

* Multimodal Panoramic 3D Outdoor Dataset

Please download the Multimodal Panoramic 3D Outdoor Dataset from this [link](http://robotics.ait.kyushu-u.ac.jp/kyushu_datasets/outdoor_dense.html).

## Acknowledgements
We thank the reviewers for their valuable feedback that has helped us improve our paper.

We use the public implementation of [fast_soft_sort](https://github.com/google-research/fast-soft-sort) and [GCNN](https://github.com/basveeling/keras-gcnn), we thank the authors.

