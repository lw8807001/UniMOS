# UniMOS: A Universal Framework For Multi-Organ Segmentation Over Label-Constrained Datasets

## Background
The annotation of medical images is very important especially in training machine learning models which may aid medical practitioners in the diagnosis and management of diseases such as cancer. Due to the fact that medical image annotation requires a great deal of manpower and expertise, as well as the fact that clinical departments perform image annotation based on task orientation, there is the problem of having fewer medical image annotation data with more unlabeled data and having many datasets that annotate only a single organ. However, existing methods do not fully utilize all these types of data. 

## Introduction
In this paper, we present UniMOS, the first universal framework for achieving the utilization of fully and partially labeled images as well as unlabeled images. Specifically, we construct a Multi-Organ Segmentation (MOS) module over fully/partially labeled data as the basenet, where multi-view multi-scale contextual features are extracted to enhance representation and a new target adaptive loss is designed for MOS. Furthermore, we incorporate a semi-supervised training module over unlabeled data. This module effectively combines consistent regularization and pseudo-labeling techniques, leveraging both image-level and feature-level interference to notably enhance the segmentation of unlabeled data. 

## Environment
We set up environment by Anaconda.
- python 3.7, torch 1.3.0
- conda install -c simpleitk simpleitk
- conda install -c conda-forge nibabel
- pip install torchvision=0.4
- pip install pillow
- pip install scikit-image

## Dataset
All datasets we used in the training and validation of UniMOS can be downloaded from links below: 
- LiTS: https://aistudio.baidu.com/datasetdetail/10273
- KiTS: https://aistudio.baidu.com/datasetdetail/24582
- MSDSpleen: https://aistudio.baidu.com/datasetdetail/23918

After datas are downloaded, please store them in corresponding folders as follows:
```
├─LiTS
│  ├─label            //labels of labeled data and validation data
│  ├─labeled          //labeled data in training set 
│  ├─unlabeled        //unlabeled data in training set
│  └─validation       //validation data in validation set
├─KiTS
│  └─...
└─MSDSpleen
   └─...
```
## Preprocessing
Please use `resample.py` to resize the images and labels into n\*256\*256 by:
```
resample -p1 './data/raw_LiTS/' -p2 './data/LiTS/' -s1 256 -s2 256
```

## Usage
You can train and validate UniMOS on JupyterLab as:
```
sbatch Final.sh
```
The parameters in `Final.sh` are defined in `Final.py` and can be modified by yourself.
