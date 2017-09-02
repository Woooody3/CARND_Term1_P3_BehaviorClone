
# **Behavioral Cloning** 

## 0 Project goal

--- 
This project targets to build a model which is able to learn how to steer during driving, by followed the input driving behavior, in the simulation.

A simulator, drive.py and video.py are provided by Udacity. Here we're going to:
- Collect training data 
- Build the model
- Train the model with the collected data(images and steering angles), and refine it until the train & validation loss is acceptable
- Test the refined model(.h5 file) by runing drive.py. 

## 1 Collecting Data
---
[//]: # (Image References)

[image1]: ./Results/Image_example.png 
[image2]: ./Results/driving-log-output.png



### 1.1 Captured Data Format
By running the simulator under training and recording condition, the information would be recored in following type. 
- a "IMG" file restored images captured by 3 vehicle camera, left, center, right camera. Thus 3 images were recorded at one time.
![text][image1]
- a csv file restored images file location, steering angle, throttle, brake and speed. 
![text][image2]

To this CNN model, the data should be in format of:
- X: 4d matrix (num, height, width, color_channels)
- y: 1d array (steering angle)

### 1.2 Data collection method
To gather data with as good completeness as possible, to cover possible situations, I did 
1. Run 2 clockwise loops;   
2. Run 2 counter-clockwise loops;
3. Run start from side of the road, and record the recover;
4. Repeat "3" many(10-20) times in different start point, in both direction;
5. Repeat many times recording in the place where model went off trail(this is when the model tested fail).

## 2 Build the Model
---
[//]: # (Image References)

[image3]: ./Results/SteerDistribution_Origin.png 
[image4]: ./Results/SteerDistribution_LeftRightAdjust.png
[image5]: ./Results/SteerDistribution_LeftRightAdjust_Dicentralize.png 
[image6]: ./Results/bgr.png 
[image7]: ./Results/hsv.png
[image8]: ./Results/hsv_random.png 
[image9]: ./Results/rgb_random.png 
[image10]: ./Results/rgb_flip.png
[image11]: ./Results/image_BeforeCrop.jpg 
[image12]: ./Results/image_AfterCrop.jpg



### 2.1 Preprocess the data
Preparing the input data into a clean shape before the model built, is critical to an efficient and successful result.  

** 2.1.1 Import the raw input data ** (plot.py)

To plot all the recored steering data, it's clear that more than half of them are with about 0 steering angle.

![text][image3]

** 2.1.2 Steerings: Adjustment in left/right image** (line 56)

We are inclinded to keep the vehicle away from the side lanes. It will help that adjust a +0.2 angle to left camera image, and a -0.2 angle to right camera image.

Thus, the steering distribution will looks like following plot. The bias is still obvious.
![text][image4]

** 2.1.3 Steerings: Filtering ** (line 13-20) 

To avoid the bias caused by this heavy concentrated contribution, we should set up a filter to drop off 90% of data which have less then 0.01 steering. 

The filter should be implemented when reading the csv file, before the left/right image steering adjustment.

The distribution now looks like:
![text][image5]

**2.1.4 Images: Adjustment in color** (line 26-32)

 <i>cv2.imread</i> reads the image in BGR format, while drive.py runs in RGB format. On the other hand, to give the images an arbitrary brightness would help train under more situations.
 ![text][image6]
- transfer from BGR to HSV
![text][image7]
- randomly adjust the value in brightness channel
![text][image8]
- transfer from HSV to RGB
![text][image9]


** 2.1.5 Images: Flip them ** (line 59-60)

Fliping the image horizontally makes great deal, it means totally different training data to the model.
![text][image10]

** 2.1.6 Images: Add 3 data points by one line **

Now we get 3 "images-steering pairs" of data from one recorded csv line:
- left image, adjusted steering angle (randomly flipped)
- center image, steering angle (randomly flipped)
- right image, adjusted steering angle (randomly flipped)

Add all of them into X_data.

** 2.1.7 Generator: to handle large volumn of data ** (line 37-67)

Together with keras fit_generator, the defined generator function process a small batch of data, requiring much less memory space, and would speed up the model. 

** 2.1.8 Image Resize, cropping, normalization are done inside the model ** (line 75-77, line 86-88)
- cropping: remove data of first 75 rows and end 25 rows, will remove irrelevant information;

![text][image11]
![text][image12]
- resize: use tensorflow.image.resize_image to a (64,64,3) shape, will speed up the model;
- normalization: will avoid bias caused by absolute values

### 2.2  The Model
--- 
[//]: # (Image References)

[image13]: ./Results/Train_Val_Loss.png 


** 2.2.1 Architecture **
NIVIDIA architecture is used, with kernel sizes adjusted in convolutional layers according to the input data shape.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image  							|
| Cropping        		| 60x320x3 RGB image  							|
| Resize         		| 64x64x3 RGB image  							|
| Normalize        		| 64x64x3 RGB image  							|
| Convolution 2x2     	| stride 2x2, padding valid, outputs 32x32x24 	|
| ReLU					|												|
| Dropout           	|                                            	|
| Convolution 2x2     	| stride 2x2, padding valid, outputs 16x16x36 	|
| ReLU					|												|
| Dropout           	|                                            	|
| Convolution 2x2	    | stride 2x2, padding valid, outputs 8x8x48     |
| ReLU					|												|
| Dropout           	|                                            	|
| Convolution 2x2	    | stride 1x1, padding valid, outputs 7x7x64     |
| ReLU					|												|
| Convolution 2x2	    | stride 1x1, padding valid, outputs 6x6x64     |
| ReLU					|												|
| Flatten               | outputs 2304                                  |
| Fully connected       | outputs 120        			                |
| Fully connected       | outputs 50        			                |
| Fully connected       | outputs 10        			                |
| Fully connected       | outputs 1        			                    |

**2.2.2 Train the Model**
The parameters:
- samples_per_epoch = 4 * (training data length)
- number of epoch = 6
- dropout probability = 0.6

**2.2.3 Evaluate the Model**
The model's training loss and validation loss shown in below picture:
![text][image13]

## 3 Test the Model

[//]: # (Image References)

[image14]: ./Results/too_right.png
[image15]: ./Results/in_middle.png
[image16]: ./Results/too_left.png
During several rounds of tests, it is found out the model could run off the bridge, run into the muddy road right after the bridge, or run into lake in sharp turns.

The model was finetoned accordingly, e.g. screening out certain amount of near zero steering data.

The final model as described above, is able to drive staying on track in "track 1" situation. 

But it still could off center time to time, an improvement might be needed to enhance the stabalization. Here are some screenshot from "run_0901_08.mp4".
- After going off the bridge, the vehicle leans too right, a big left turn prevents it running to muddy road
![text][image14]
- After the muddy road interaction, the vehicle adjusts right back to the road center
![text][image15]
- Then there's goint to be a sharp left turn, the vehicle leans too left, it almost goes off road until a big right turn saves it back
![text][image16]


