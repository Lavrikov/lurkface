# Lurkface: С++ GPU Real-time video surveillance face covering(blur) to satisfy European GDPR law
 
### Introduction
 The new European law required cover all faces from unutorized access. So, we sholud find whole bunch of faces
 and cover them from anyone exept authorities. This is c++ object detection implementation for five cams simultaneously
 proceding with GeForce 4 Gb 1050Ti. 
 
### links
This repo uses folowing progects source code and data:
This is a C++ implementation of SORT used with some changes as core of face tracker https://github.com/mcximing/sort-cpp
This caffe model (Faceboxes) used as a pretrained model for the face boxes detection https://github.com/sfzhang15/FaceBoxes.

### Inference
In fact, more fast and thin networks tend to loose faces frecuently than original. In this project, combination SSD with SORT
handle this problem with low price.  



@inproceedings{zhang2017faceboxes,
  title = {Faceboxes: A CPU Real-time Face Detector with High Accuracy},
  author = {Zhang, Shifeng and Zhu, Xiangyu and Lei, Zhen and Shi, Hailin and Wang, Xiaobo and Li, Stan Z.},
  booktitle = {IJCB},
  year = {2017}
}