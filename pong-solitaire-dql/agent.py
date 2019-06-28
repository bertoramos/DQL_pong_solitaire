
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense

from collections import deque

import random

class Agent:
	def __init__(self, state_size, n_actions):
		self.state_size = state_size
		self.n_actions = n_actions

		self.replay_memory = deque(maxlen=1000)
		self.gamma = 0.90 # discount rate
		self.epsilon = 1.0 # Exploration rate
		self.d_epsilon = 0.99
		self.lr = 0.001

		self.model = self._model_builder()

	def _model_builder(self):
		model = Sequential()
		model.add(Dense(5, input_dim=self.state_size, activation='relu'))
		model.add(Dense(15, activation='relu'))
		model.add(Dense(self.n_actions, activation='linear'))
		model.compile(loss='mse', optimizer=keras.optimizers.Adadelta())
		return model

	def remember(self, state, action, reward, next_state, done):
		self.replay_memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.n_actions)
		else:
			prediction = self.model.predict(state)
			return np.argmax(prediction)

	def replay(self, batch_size):
		if len(self.replay_memory) > batch_size:
			minibatch = random.sample(self.replay_memory, batch_size)
		else:
			minibatch = [e for e in self.replay_memory]
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
				prediction = self.model.predict(state)
				prediction[0][action] = target
				self.model.fit(state, prediction, epochs=1, verbose=0)
			self.epsilon *= self.d_epsilon
