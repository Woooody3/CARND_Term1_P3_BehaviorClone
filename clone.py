import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
#	for row in reader:
#		steering_center = float(row[3])
#		# create adjusted steering measurements
#		correction = 0.2
#		steering_left = steering_center + correction
#		steering_right = steering_center - correction
#		# read images from left, center and right
#		path = 'data/IMG/'
#		img_center = process_img(np.asarray(Image.open(path+row[0])))
#		img_left = process_img(np.asarray(Image.open(path+row[1])))
#		img_right = process_img(np.asarray(Image.open(path+row[2])))
#		# add images and angles to dataset
#		car_images.extend(img_center, img_left, img_right)
#		car_angles.extend(steering_center, steering_left, steering_right)


images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = 'data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		images.append(np.fliplr(image)) # append augmented image
		measurement = float(line[3]) + 0.2 * (5*i - 3*i*i) /2	# if i=0, measurement is center; if i=1, measurement is +0.2 correction, if i=2, measurement is -0.2 correction.
		measurements.append(measurement)
		measurements.append(measurement * (-1)) # append augmented measurement

'''
Follow 6 lines are learnt from course vedio, however, I choose to append the augment data in each line-reading. 
'''
#augmented_images, augmented_measurement = [], []
#for image, measurement in zip(images, measurements):
#	augmented_images.append(image)
#	augmented_measurements.append(measurement)
#	augmented_images.append(cv2.flip(image, 1))
#	augmented_measurements.append(measurement * (-1))

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
model.add(Convolution2D(32,3,3,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(60))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5) 

model.save('model.h5')
exit()


