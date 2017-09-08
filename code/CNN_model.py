from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
import cPickle as pickle
import sys
import time
from Metrics import f1,recall,precision
from matplotlib import pyplot as plt
import keras




def load_data(filename):
	'''
	Load data from pickle file, then split data to training set and test set.
	Input: pickle filename
	Output: training data and testing data
	'''

	start_time = time.time()
	print'-'*20+'Start loading data'+'-'*20
	with open('../data/{}'.format(filename),'r') as f:
		X,y = pickle.load(f)

	print'-'*20+'Finish loading'+'-'*20
	print 'loading time: {}s'.format(round(time.time()-start_time,2))
	X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=100,stratify=y,test_size=.15)
	return 	X_train,X_test,y_train,y_test

def generator(X_train,X_test,y_train,y_test):
	'''
	Initialize data generator for model training
	Input: training data and testing data
	Output: training generator and testing generator
	'''

	train_datagen = ImageDataGenerator(
	        rescale=1./255,
	        horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1./255)
	train_generator = train_datagen.flow(X_train,y_train)
	test_generator = test_datagen.flow(X_test,y_test)
	return train_generator, test_generator


def model_fit(filename,epochs,save=None):
	'''
	load training data and testing data, compile and train CNN model, return training history
	Parameters
	filename: pickle data
	epochs: number of epochs for training
	save(optional): path for saving trained model
	Output: training history
	'''
	
	X_train,X_test,y_train,y_test = load_data(filename)
	train_generator,test_generator = generator(X_train,X_test,y_train,y_test)
	# dimensions of images.
	img_height,img_width = X_train.shape[1],X_train.shape[2]

    # number of class
	num_class = y_train.shape[1]

	if K.image_data_format() == 'channels_first':
	    input_shape = (3, img_height, img_width)
	else:
	    input_shape = (img_height, img_width, 3)
	print 'input shape: ',input_shape
	# layer 1
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# layer 2
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# layer 3
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# layer 4
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# layer 5
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	# layer 6
	model.add(Dense(32))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	# layer 7
	model.add(Dense(num_class))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy',recall,precision])

	batch_size = 32

	hist = model.fit_generator(
		  train_generator,
		  steps_per_epoch=len(X_train) / 32,
	   	  epochs=epochs,
		  validation_data=test_generator,
		#   callbacks=[callback],
		  validation_steps = len(X_test)/32)
	if save:
		model.save(save)
	return hist

if __name__ == '__main__':
	filename = sys.argv[1]
	epochs = int(sys.argv[2])
	hist = model_fit(filename,epochs)

	with open('hist.pkl','wb') as f:
	    pickle.dump(hist.history,f)
