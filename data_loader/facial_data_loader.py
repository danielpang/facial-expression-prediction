from base.base_data_loader import BaseDataLoader
from tensorflow.python.keras.datasets import mnist
import pandas as pd
import numpy as np
from keras.utils import np_utils

class FacialDataLoader(BaseDataLoader):
	def __init__(self, config):
		super(FacialDataLoader, self).__init__(config)
		# Import csv file
		raw = pd.read_csv("data/fer2013.csv")
		train = raw[raw['Usage'] == 'Training']
		train_pixels = train['pixels']
		self.train_y = train['emotion'].values
		self.train_y = np_utils.to_categorical(self.train_y,10)

		test = raw[raw['Usage'] == 'PublicTest']
		test_pixels = test['pixels']
		self.test_y = test['emotion'].values
		self.test_y = np_utils.to_categorical(self.test_y,10)

		# Convert raw training pixel data into numpy array
		train_x = []
		for image in train_pixels:
			temp = [int(n) for n in image.split()]
			train_x.append(temp)
		self.train_x = np.asarray(train_x)
		self.train_x = np.resize(train_x, (len(train_pixels),1, 48, 48))

		# Convert raw testing pixel data into numpy arrays
		test_x = []
		for image in test_pixels:
			temp = [int(n) for n in image.split()]
			test_x.append(temp)
		self.test_x = np.asarray(test_x)
		self.test_x = np.resize(test_x, (len(test_pixels), 1, 48, 48))

	def get_train_data(self):
		return self.train_x, self.train_y

	def get_test_date(self):
		return self.test_x, self.test_y
