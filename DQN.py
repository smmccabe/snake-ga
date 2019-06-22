from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
import sys

class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network()
        # self.model = self.network("weights.hdf5")
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def get_state(self, game, player, food):
        state = [0] * 400

        index = int(player.x + ((player.y - 40) * 20)) / 20
        if index >= 0 or index > 399:
            print(index)
        state[index] = 1

        index = (food.x_food + ((food.y_food - 40) * 20)) / 20
        if index < 0 or index > 399:
            print(index)
        state[index] = 2

        if player.food > 1:
            for i in range(1, player.food):
                tail_index = len(player.position) - 1 - i
                index = int(player.position[tail_index][0] + ((player.position[tail_index][1] - 40) * 20)) / 20
                if index < 0 or index > 399:
                    print(index)
                state[index] = 3

        #for i in range(len(state)):
        #    sys.stdout.write(str(state[i]))
        #    if i % 20 == 0:
        #        sys.stdout.write("\n")

        return np.asarray(state)

    def set_reward(self, player, crash):
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10 - (player.turns * 0.01)
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(units=120, activation='relu', input_dim=400))
        model.add(Dropout(0.15))
        model.add(Dense(units=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(units=120, activation='relu'))
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
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 400)))[0])
        target_f = self.model.predict(state.reshape((1, 400)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 400)), target_f, epochs=1, verbose=0)
