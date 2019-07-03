from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
import sys
import subprocess as sp
import math

class DQNAgent(object):

    def __init__(self, size):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        #self.learning_rate = 0.0005
        self.learning_rate = 0.01
        self.size = int(math.pow(size / 20, 2))
        self.epsilon = 0
        self.actual = []
        self.memory = []
        self.model = self.network()
        #self.model = self.network("weights.hdf5")

    def get_state(self, game, player, food):
        state = [0] * self.size

        index = int((player.x / 20) + (((player.y - 40) / 20) * (game.game_width / 20)))
        if index >= 0 and index < self.size:
            state[index] = 1

        index = int((food.x_food / 20) + (((food.y_food - 40) / 20) * (game.game_width / 20)))
        if index >= 0 and index < self.size:
            state[index] = 2

        if player.food > 1:
            for i in range(1, player.food):
                tail_index = len(player.position) - 1 - i
                index = int((player.position[tail_index][0] / 20) + (((player.position[tail_index][1] - 40) / 20) * (game.game_width / 20)))
                if index >= 0 and index < self.size:
                    state[index] = 3

        #sp.call('clear', shell=True)
        #for i in range(len(state)):
        #    sys.stdout.write(str(state[i]))
        #    sys.stdout.write(" ")
        #    if i % (game.game_width / 20) == 0:
        #        sys.stdout.write("\n")

        return np.asarray(state)

    def set_reward(self, player, crash):
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(units=self.size, activation='relu', input_dim=self.size))
        model.add(Dropout(0.15))
        model.add(Dense(units=int(self.size / 2), activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(units=3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * \
                np.amax(self.model.predict(
                    next_state.reshape((1, self.size)))[0])
        target_f = self.model.predict(state.reshape((1, self.size)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, self.size)),
                       target_f, epochs=1, verbose=0)
