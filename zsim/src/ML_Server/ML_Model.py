import tensorflow as tf
from collections import deque
import random
import os
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import numpy as np
    from numpy import savetxt
    from numpy import loadtxt
    from keras import models
    from keras import layers
    from keras import optimizers
    from keras.callbacks import ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0


class DQN:
    def __init__(self, _num_features, _num_output, _rep_size, _app_name, _checkpt="none"):
        self.gamma        = 0.95    # discounting factor
        self.alpha        = 0.05    # learning rate
        self.eps          = 0.05    # exploration rate
        self.num_actions  = _num_output
        self.num_features = _num_features
        self.num_retrains = 0
        self.network = self.createModel(self.num_actions*self.num_features, _num_output, _checkpt)
        self.replay_buffer = deque(maxlen=_rep_size)
        if _checkpt == "none":
            self.mcp_save = ModelCheckpoint('weights/'+_app_name+'.hdf5', save_best_only=True, monitor='loss', mode='min')
        else:    
            self.mcp_save = ModelCheckpoint(_checkpt, save_best_only=True, monitor='loss', mode='min')
        print("New model created\n")

    def createModel(self, _num_input, _num_output, _checkpoint):
        model = models.Sequential()
        model.add(layers.Dense(32, activation='tanh', input_shape=(_num_input,)))
        model.add(layers.Dense(32, activation='tanh'))
        model.add(layers.Dense(_num_output))
        if _checkpoint != "none":
            model.load_weights(_checkpoint)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def predict(self, state):
        if np.random.randint(0,99) < int(self.eps * 100):
            action = np.random.randint(0,self.num_actions)
            return action
        else:
            x_input = [state]
            x_input = np.array(x_input)
            action = np.argmax(self.network.predict(x_input))
            return action

    def addSample(self, data):
        state      = []
        next_state = []
        action     = data[-2]
        reward     = data[-1]

        state_length = self.num_features*self.num_actions
        for index in range(state_length):
            state.append(data[index])
        for index in range(state_length, 2*state_length):
            next_state.append(data[index])
        
        self.replay_buffer.append((state, int(action), next_state, reward))


    def retrain(self, draw_size=1000, train_batch_size=32, num_epochs=3):
        # sanity check: there must be something in the replay buffer
        assert(len(self.replay_buffer))
        
        # sanity check: draw size always > batch size
        assert(draw_size >= train_batch_size)

        if len(self.replay_buffer) >= draw_size:
            batch = random.sample(self.replay_buffer, draw_size)
        else:
            batch = self.replay_buffer
            if len(self.replay_buffer) < train_batch_size:
                train_batch_size = len(self.replay_buffer)

        states_batch, actions_batch, next_states_batch, rewards_batch = map(np.array, zip(*batch))
        current_Q_batches = self.network.predict(states_batch)
        next_Q_batches = self.network.predict(next_states_batch)

        # update new Q-value: Q(s,a) = Q(s,a) + alpha * (reward + gamma * maxQ(s',a') - Q(s,a))
        for index in range(len(current_Q_batches)):
            current_Q_batches[index][actions_batch[index]] += self.alpha * \
                                                              (rewards_batch[index] + \
                                                              self.gamma * np.max(next_Q_batches[index]) - \
                                                              current_Q_batches[index][actions_batch[index]])  
        
        self.network.fit(states_batch, current_Q_batches, batch_size=train_batch_size, \
                        verbose=False, epochs=num_epochs, callbacks=[self.mcp_save])
        self.num_retrains += 1