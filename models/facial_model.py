from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils
import keras.losses
import keras
import tensorflow as tf

class FacialModel(BaseModel):
	def __init__(self, config):
		super(FacialModel, self).__init__(config)
		self.build_model()

	def build_model(self):
		in_shape = (1, 48, 48)
		self.model = Sequential()
		self.model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', data_format='channels_first', activation='relu', input_shape=in_shape))
		self.model.add(MaxPooling2D())
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.25))

		# Add noise to our data
		model.add(GaussianNoise(0.3))

		self.model.add(Conv2D(128, kernel_size=(5,5), strides=(1,1), padding='same', data_format='channels_first', activation='relu', input_shape=in_shape))
		self.model.add(MaxPooling2D())
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.25))

		self.model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same', data_format='channels_first', activation='relu'))
		self.model.add(MaxPooling2D())
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.25))

		self.model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same', data_format='channels_first', activation='relu'))
		self.model.add(MaxPooling2D())
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.25))

		self.model.add(Flatten())
		self.model.add(Dense(256, activation='relu'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.25))

		self.model.add(Dense(512, activation='relu'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.50))
		self.model.add(Dense(10, activation='softmax'))

		self.model.compile(loss=keras.losses.categorical_crossentropy,
			optimizer=self.config.optimizer,
			metrics=['acc'])
