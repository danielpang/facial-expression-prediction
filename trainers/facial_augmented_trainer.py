from base.base_trainer import BaseTrain
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

class FacialTrainer(BaseTrain):
	def __init__(self, model, data, config):
		super(FacialTrainer, self).__init__(model, data, config)
		self.callbacks = []
		self.loss = []
		self.acc = []
		self.val_loss = []
		self.val_acc = []
		self.init_callbacks()

	def init_callbacks(self):
		self.callbacks.append(
			ModelCheckpoint(
				filepath=os.path.join(self.config.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp_name),
				monitor=self.config.checkpoint_monitor,
				mode=self.config.checkpoint_mode,
				save_best_only=self.config.checkpoint_save_best_only,
				save_weights_only=self.config.checkpoint_save_weights_only,
				verbose=self.config.checkpoint_verbose
			)
		)

		self.callbacks.append(
			TensorBoard(
				log_dir=self.config.tensorboard_log_dir,
				write_graph=self.config.tensorboard_write_graph
			)
		)

	def train(self):
		# Use image augmentation
		datagen = ImageDataGenerator(
			featurewise_center=True,
			featurewise_std_normalization=True,
			rotation_range=20,
			width_shift_range=0.2,
			height_shift_range=0.2,
			horizontal_flip=True,
			data_format='channels_first')

		# compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied)
		datagen.fit(self.data[0])

		# Compute representation of each label in the training set and
		# assigns weights to boost under-represented labels in the training set
		weights = class_weight.compute_class_weight('balanced', np.unique(self.data[1]), self.data[1])
		history = model.fit_generator(datagen.flow(self.data[0], self.data[1], batch_size=32),
			steps_per_epoch=len(self.data[0]) / 32,
			epochs=self.config.num_epochs,
			verbose=self.config.verbose_training,
			batch_size=self.config.batch_size,
			validation_split=self.config.validation_split,
			callbacks=self.callbacks,
			class_weight=weights)

		# with tf.Session() as sess:
		# 	sess.run(tf.global_variables_initializer())
		# 	history = self.model.fit(
		# 		self.data[0], self.data[1],
		# 		epochs=self.config.num_epochs,
		# 		verbose=self.config.verbose_training,
		# 		batch_size=self.config.batch_size,
		# 		validation_split=self.config.validation_split,
		# 		callbacks=self.callbacks
		# 	)
		self.loss.extend(history.history['loss'])
		self.acc.extend(history.history['acc'])
		self.val_loss.extend(history.history['val_loss'])
		self.val_acc.extend(history.history['val_acc'])
