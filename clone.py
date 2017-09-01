import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle
import matplotlib.pyplot as plt 			

### Preparing Data ###

# Import data
# and drop off about 70% of data in straight road with no steering, to avoid bias
lines = []

with open('0831_1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		probability = np.random.random()
		if probability < 0.5 and float(line[3]) <0.005:
			continue
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Define a input data generator, to deal with large volume of collected image data.
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: 
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset : offset+batch_size]

			images = []
			measurements = []
			for batch_sample in batch_samples:
				for i in range(3): # append center, left, right images and corresponding measurements
					image_path = '0831_1/IMG/' + batch_sample[i].split('/')[-1]
					img = cv2.imread(image_path)
					image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					images.append(image)
					measurement = float(batch_sample[3]) + 0.2 * (5*i - 3*i*i) /2	# if i=0, measurement is center; if i=1, measurement is +0.2 correction, if i=2, measurement is -0.2 correction.
					measurements.append(measurement)
			
			# adding the fliped images and measurements
			augmented_images, augmented_measurements = [], []
			for image, measurement in zip(images, measurements):
				augmented_images.append(image)
				augmented_measurements.append(measurement)
				augmented_images.append(cv2.flip(image, 1))
				augmented_measurements.append(measurement * (-1))

			X_in = np.array(augmented_images)
			y_in = np.array(augmented_measurements)
			yield sklearn.utils.shuffle(X_in, y_in)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


### Training Model ###

def img_resize(X):
	from keras.backend import tf as ktf
	return ktf.image.resize_images(X, (60,60))

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

dropping_prob = 0.60
model = Sequential()
model.add(Cropping2D(cropping=((75,25),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(img_resize))
model.add(Lambda(lambda x: (x/255.0)-0.5))
model.add(Convolution2D(24,2,2, subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Dropout(dropping_prob))
model.add(Convolution2D(36,2,2, subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Dropout(dropping_prob))
model.add(Convolution2D(48,2,2, subsample=(2,2), activation='relu'))
model.add(Dropout(dropping_prob))
model.add(Convolution2D(64,2,2, activation='relu'))
model.add(Convolution2D(64,2,2, activation='relu'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

# Visualizing Loss
'''
Confused: what's the better value of the samples_per_epoch?
'''
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
	validation_data=validation_generator, nb_val_samples=len(validation_samples), 
	nb_epoch=6, verbose=1)
print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model MSE loss')
plt.ylabel('MeanSquareError')
plt.xlabel('epoch')
plt.legend(['training_set', 'validation_set'], loc='upper right')
plt.show()

model.save('model.h5')
exit()


