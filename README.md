# Road Segmentation with U-net

#### Overview

The repository includes an application using U-net CNN architecture to segment roads from images captured by vehicle cameras. The U-net concatenate the receptive fields in the convolutional and up-sampling layers. It is a convolutional network architecture that can segment images in a fast and precise manor with relatively less training samples.  

The task for this project is for road segmentation from camera images with Convolutional Neural Networks. 

#### Data preparation and augmentation

This project uses the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) which includes the 289 training and 290 test images in 3 categories: unban unmarked roads, urban marked roads, and urban multiple marked roads.

Data preparation solves following problems. 1) Size of images are not consistent. 2) Certain sizes are not valid for the U-net architecture. Concatenating the convolutionaized and upsampled layers requires the image size to be divisible by 2^n, where n is the number of MaxPooling layers in the architecture. 3) The masks has 3 channels, and it is converted to the gray scale image with road as 255 and background as 0. All these are fixed by zero padding the original images and masks into 1248X384 spatial resolution.

 Image augmentation includes randomly shifting image < 10% in width and height direction, zoom < 10%, fill the empty space with 0.0, horizontally flip image and rescale the image to 1./255.

The images preparation are carried out with programs in prepareImageMask.py.

![um_000006.png (1248Ã384)](https://raw.githubusercontent.com/EricYang3721/RoadSegmentationWithUnet/master/images/validation/val_image/val_image/um_000006.png)

#### Model

The U-net architecture is firstly introduced for image segmentation on medical images on [MICCAI 2015](https://arxiv.org/abs/1505.04597). The medical images are gray scale images and the images, and the convolutionized layers are cropped to concatenate with the upsample layers. 

In our implementation, we revised the input layer into 3 channels to enable it for RGB images, and the images are padded in the convolutional layers to keep spatial sizes of the feature maps. Also, several models with 3, 4, 5 MaxPoolings, kernel regularization on the convolution kernels are also implemented as in [unetModel.py](https://github.com/EricYang3721/RoadSegmentationWithUnet/blob/master/unetModel.py). 

![1524444202426](/images/pics/1524444202426.png)



#### Loss functions

Four different loss functions are used for training the model ([lossFunction.py](https://github.com/EricYang3721/RoadSegmentationWithUnet/blob/master/lossFunction.py)):

1. Binary cross-entropy (BCE): pixel-wised classification
2. Dice or 1- Intersection of Union (IOU). 
3. BCE + Dice: as BCE + 1- log(IOU).
4. Weighted-BCE-DICE: the improve the prediction on the edge of the roads, in calculating the loss function, the weights of pixels close to the edge of the road on ground truth masks are 3 time as those far away from the edges. 

#### Training

The models are trained in [main.py](https://github.com/EricYang3721/RoadSegmentationWithUnet/blob/master/main.py). Models are trained in 200 epochs, with learning rate decay 0.5 for every 20 epochs. The weights with the best validation IOU scores are save, and the training process are recorded with CSVlogger. 

#### Result analysis 

Performance analysis: 

Functions used evaluating the performance of models are implemented in [postProcessing.py](https://github.com/EricYang3721/RoadSegmentationWithUnet/blob/master/postProcessing.py), which analyze the IOU score vs prediction threshold, and the ROC curve on the validation data as following. 

![1524445822379](/images/pics/1524445822379.png)

Results illustration:

[help.py](https://github.com/EricYang3721/RoadSegmentationWithUnet/blob/master/helper.py) includes the implementations to show the images for analysis. showImage could plot 2 images at the same time for comparing original images/masks/predictions in different windows. PlotImgMsk overlays the image with mask/prediction result on the same image. To better illustrate the mask/prediction, the original images are converted into gray scale.

![1524446180406](/images/pics/1524446180406.png)

plotPredTruth function overlays the prediction results and ground truth mask on the same image for visually comparing the prediction with the ground truth.

![1524446247303](/images/pics/1524446247303.png)



#### Some results

The IOU score achieved on validation results are 0.935.

Results on validation samples:

Simpler task (prediction results):

![1524446622493](/images/pics/1524446622493.png)

simpler task (ground truth): 

![1524446658204](/images/pics/1524446658204.png)

More complicated prediction results:

![1524446475049](/images/pics/1524446475049.png)

![1524446516925](/images/pics/1524446516925.png)

