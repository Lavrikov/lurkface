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

As a matter of fact, to hide all faces on a video we should mere detect and cover them with an any box type. At the first sight, face detection is a well-established task with a lot of "state of art" solutions. Although, here are two lurked problems have to be solved. At the first, our solution have to execute with near to "real time" speed. Because of real applications of secure cameras have to be proceed with in at least 12-25 fps. So, it arises the contradiction between reducing resolution of image and ability to detect all faces including absolutely tiny of them (18x18 px). At the second, many of stunning quality pretrained models falls with continuous face detection on a video. Especially, this becomes quite distinctly with shrinking of image resolution. After a little investigation we can figure out that the confidence of pretrained networks about the particular detected face changing dramatically between frames. For example, look at picture 1. 

picture 1. S3FD network confidence about the same face between different frames. 

Further, to solve the first problem we should use model reduction methods like .... . It is works for particular conditions, because of selected video scene has pure set of backgrounds usually, in contrast to plenty of them at train datasets. So, low capacity network handle it well enough. For instance, "Tiny" network was used instead of vgg16 as feature extractor. As well as the deepwise convolution is used to replace all appropriate layers of the network due to speed reasons. 

As for second problem, it is needed to make a little trick. Generally, despite of falling the confidence value, the network still produces a correct bbox for a particular detected face. If we could once detect area as face with high confidence score, we would continue to attribute this object as face further.

@inproceedings{zhang2017faceboxes,
  title = {Faceboxes: A CPU Real-time Face Detector with High Accuracy},
  author = {Zhang, Shifeng and Zhu, Xiangyu and Lei, Zhen and Shi, Hailin and Wang, Xiaobo and Li, Stan Z.},
  booktitle = {IJCB},
  year = {2017}
}
