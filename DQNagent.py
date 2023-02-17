from collections import deque
from keras.models import save_model
import numpy as np
import random
import tensorflow as tf
import csv


class DQNagent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.ID = 1
        self.IDres = 0

        self.memory = deque(maxlen= 2000)
        self.last_iteration = []

        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01

        self.learning_rate = 0.001
        # agent.model = agent._build_model()
        self.episode = 0
        self.max_episodes = 500

    def remember(self, iteration):
        self.memory.append(iteration[1:])


def replay(agent, batch_size, model):
    minibatch = random.sample(agent.memory, batch_size)
    for state, action, next_state, reward in minibatch:
        target = (reward + agent.gamma *
                  np.amax(model.predict(next_state.astype('float32'))[0]))
        target = reward
        target_f = model.predict(state.astype('float32'))
        target_f[0][action] = target
        model.fit(state.astype('float32'), target_f, epochs=1, verbose=0)
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    save_model(model, r'C:\Users\DELL\PycharmProjects\pythonProject\model6.h5')


def act(agent, model, state):
    if np.random.rand() <= agent.epsilon:
        return random.randrange(agent.action_size)
    act_values = model.predict(state.astype('float32'))
    return np.argmax(act_values[0])  # returns action


def clean_state(state):
    if state [2] is None:
        state[2] = 0
    elif '2p' in state[2]:
        state[2] = 0.33
    elif '4p' in state[2]:
        state[2] = 0.66
    elif '6p' in state[2]:
        state[2] = 0.99
    if state[3] is None:
        state[3] = 0
    elif '2p' in state[3]:
        state[3] = 0.33
    elif '4p' in state[3]:
        state[3] = 0.66
    elif '6p' in state[3]:
        state[3] = 0.99
    return  state

"""
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from keras.optimizers import Adam

def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
        
        
def GetlastID(file):
    with open(file, 'r') as csv_file:
        rowlist = []
        csvreader = csv.reader(file)
        for row in csvreader:
            rowlist.append(row)
        print(int(rowlist[-1][0]))


import csv

with open("E:\pfe\Done.csv", 'r') as file:
    rowlist=[]
    csvreader = csv.reader(file)
    for row in csvreader:
        rowlist.append(row)
    print(int(rowlist[-1][0]))

def _build_model(agent):
    model = Sequential()
    model.add(Dense(24, input_dim = agent.state_size, activation = relu))
    model.add(Dense(24, activation = relu))
    model.add(Dense(agent.action_size, activation = linear))
    model.compile(loss="mse", optimizer=Adam(lr=agent.learning_rate))
    return model
    
def replay(agent, batch_size):
    minibatch = random.sample(agent.memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + agent.gamma *
                      np.amax(agent.model.predict(next_state)[0]))
        target_f = agent.model.predict(state)
        target_f[0][action] = target
        agent.model.fit(state, target_f, epochs=1, verbose=0)
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
"""
