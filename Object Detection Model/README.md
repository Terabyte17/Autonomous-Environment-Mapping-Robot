# Object Detection using Deep Learning
In our project, the YOLO algorithm was used for object detection to detect our target object - 'husky'. The ImageAI implementation of the YOLOv3 convolutional neural network was used for this. The model was trained on Google Colab (K80 GPU) using manually generated images of the target object 'husky' with noise objects in the background. These images were generated in the pybullet simulator. 

## Training Images
The image dataset on which the model was trained was prepared manually using the pybullet simulator. Various kinds of images of the target object 'husky' were taken in different orientations, different environments and from different distances. Different objects were put in the background as noise, so that our model is able to differentiate between 'husky and other objects. 

<p align="center">
 <img  width="400" height="400" src="https://github.com/Terabyte17/Autonomous-Room-Mapping-Robot/blob/master/Object%20Detection%20Model/husky%20(2).png">
</p>

## Labelling of images
For object detection using YOLOv3, bounding boxes were to be made around the target object in each image. This was done using the LabelImg tool which can be found at - https://github.com/tzutalin/labelImg#labelimg. This tool was used to generate annotations for each image in the PascalVOC format. Around 1300+ images were made and labelled using this.

<p align="center">
 <img  width="400" height="250" src="https://github.com/Terabyte17/Autonomous-Room-Mapping-Robot/blob/master/Object%20Detection%20Model/LabelImg.png">
</p>


## Model Training
ImageAI is a powerful python library which provides various Computer Vision capabilities using deep learning. Hence, we decided to make use of the YOLOv3 Object Detection algorithm to train our model. Before training, anchor boxes were generated with an IoU(Intersection Over Union) of 0.9. We saw, that transfer learning gave us better results rather than training our model from scratch. Hence, we trained the pre-trained model to detect 'husky'. A batch size of 4 was used with the 5 epochs over which the model was trained. It took around 3 hrs to train the model. Our model was evaluated and it gave us mAP(Mean Average Precision) of 0.9668.

<p align="center">
 <img  width="400" height="400" src="https://github.com/Terabyte17/Autonomous-Room-Mapping-Robot/blob/master/Object%20Detection%20Model/Husky_detected.png">
</p>

You can find the ImageAI github repo here - https://github.com/OlafenwaMoses/ImageAI. The detection_config.json file in which the acnhor boxes are stored has been also provided in the repo. However, due to github size constraints, the model .h5 file could not be loaded. Hence, you can see the model .h5 file along with the training and validation images and annotations here - https://drive.google.com/drive/folders/1KFqMZFUNzZ-NAC3F-uDEITPFsUmESCaP?usp=sharing. Before running the main script, you need to download the model file - detection_model-ex-005--loss-0004.657.h5 present in the models folder of the google drive folder. 

Note:- For training the model, you require the tensorflow-gpu==1.13.1 version and the latest version of imageAI.
