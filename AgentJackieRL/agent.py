import numpy as np
from helper import KungFu
import random
from keras.models import load_model

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = 0.99 # Discount factor
        self.epsilon = 1.0 # Initial exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 1e-3
        self.batch_size = 32
        self.train_start = 50000
        self.learn_rate = 0.00025

        self.possible_actions = list(range(action_size))
        self.model = KungFu(self)
        self.target_model = KungFu(self)
        self.update_target_network()

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state): # Chooses action : random action or best predicted Q-value
        if np.random.rand() <= self.epsilon:
            return random.choice(self.possible_actions)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        return np.argmax(q_values)

    def train(self, replay_buffer): # Training the Q-network using randomly sampled experiences from the replay buffer

        if len(replay_buffer) < self.train_start:
            return

        minibatch = replay_buffer.sample(self.batch_size)  

        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0] # Predict current Q-values
            if done:
                target[action] = reward# No future reward if episode ended
            else: # Predicts the future Q-values from the target model
                t = self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0]
                target[action] = reward + self.gamma * np.max(t)
            states.append(state)
            targets.append(target)

        # Fitting model to minimize loss between predicted Q-values and target Q-values
        self.model.fit(np.array(states), np.array(targets), batch_size=self.batch_size, verbose=0)

        # Manual Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)


    def save(self, path): # Saves the model to disk
        self.model.save(path)

    def load(self, path): # Loads the model from disk and updates the target network
        self.model = load_model(path) 
        self.update_target_network()

