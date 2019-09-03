from keras.models import Sequential
from keras.layers import Dense , Dropout , Conv2D , MaxPooling2D , Activation , Flatten 
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import numpy as np
import keras.backend.tensorflow_backend as backend
import tensorflow as tf
from collections import deque
import time
import random 
from tqdm import tqdm
import os
from PIL import Image
import cv2
from initializers import *

class Circle:
	def __init__(self , circleX , circleY):
		self.circleX = circleX
		self.circleY = circleY
class Rect:
	def __init__(self , left , top , width , height):
		self.left = left
		self.top = top
		self.width = width 
		self.height = height

class State:
	def __init__(self , rect , circle):
		self.rect = rect
		self.circle = circle

class ModifiedTensorBoard(TensorBoard):

	# Overriding init to set initial step and writer (we want one log file for all .fit() calls)
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.step = 1
		self.writer = tf.summary.FileWriter(self.log_dir)

	# Overriding this method to stop creating default log writer
	def set_model(self, model):
		pass

	# Overrided, saves logs with our step number
	# (otherwise every .fit() will start writing from 0th step)
	def on_epoch_end(self, epoch, logs=None):
		self.update_stats(**logs)

	# Overrided
	# We train for one batch only, no need to save anything at epoch end
	def on_batch_end(self, batch, logs=None):
		pass

	# Overrided, so won't close writer
	def on_train_end(self, _):
		pass

	# Custom method for saving own metrics
	# Creates writer, writes custom metrics and closes writer
	def update_stats(self, **stats):
		self._write_logs(stats, self.step)

# DEEP Q-Learning model
class DQNAgent:
	def __init__(self):
		# Main Models
		self.model = self.create_model_ann()
		# Target Model
		self.target_model = self.create_model_ann()
		self.target_model.set_weights(self.model.get_weights())

		# Reply memory to get last n steps  memory
		self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)

		# Custom tensorboard Object for visualisation
		self.tensorboard = ModifiedTensorBoard(log_dir = "logs/{}-{}".format(MODEL_NAME , int(time.time())))
		self.target_update_counter = 0

	
	def create_model_ann(self):
		model = Sequential()
		model.add(Dense(16 , input_dim = ENV_INPUT , activation = 'relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(12, activation='relu'))
		model.add(Dense(ENV_OUTPUT, activation='linear'))
		model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
		return model

	# Conv2d Model that acccpts the Image and return the model
	def create_model(self):
		model = Sequential()
		model.add(Conv2D(256 , (3,3) , input_shape = env.OBSERVATION_SPACE_VALUES))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(2,2))
		model.add(Dropout(0.2))

		model.add(Conv2D(256, (3,3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(2,2))
		model.add(Dropout(0.2))

		model.add(Flatten())
		model.add(Dense(64))

		model.add(Dense(env.ACTION_SPACE_SIZE ,activation = 'linear'))
		model.compile(loss = 'mse' , optimizer = Adam(lr = 0.001) , metrics = ['accuracy'])
		return model

	def update_replay_memory(self , transition):
		self.replay_memory.append(transition)

	def get_qs(self , state):
		# print(np.array([state.rect.left , state.circle.circleX , state.circle.circleY ]))
		return self.model.predict(np.array([[state.rect.left/WINDOW_WIDTH , state.circle.circleX/WINDOW_WIDTH , state.circle.circleY/WINDOW_HEIGHT]]))[0]

	def train(self , terminal_state , step):
		if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
			return
		# reshuffling the batch size 
		minibatch = random.sample(self.replay_memory , MINIBATCH_SIZE)
		
		current_states =np.array([transition[0] for transition in minibatch])
		current_qs_list = self.model.predict(np.array(current_states))

		new_current_states = np.array([transition[3] for transition in minibatch ])
		future_qs_list = self.target_model.predict(np.array(new_current_states))

		X = []
		y = []

		for index , (current_state , action , reward, new_current_states , done) in enumerate(minibatch):
			if not done:
				max_feature_q = np.max(future_qs_list[index])
				new_q = reward + DISCOUNT * max_feature_q
			else:
				new_q = reward
			current_qs = current_qs_list[index]
			current_qs[action] = new_q
			X.append(current_state)
			y.append(current_qs)

		self.model.fit(np.array(X) , np.array(y) , batch_size = MINIBATCH_SIZE , verbose = 0 , shuffle = False , 
			callbacks=[self.tensorboard] if terminal_state else None)

		if terminal_state:
			self.target_update_counter += 1
		if self.target_update_counter > UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0
