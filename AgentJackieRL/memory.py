from collections import deque
import random

class Replay:
    def __init__(self, capacity): # Initializes the buffer with a maximum size
        self.buffer = deque(maxlen=capacity)

    def push(self, experience): # Adds a new experience
        self.buffer.append(experience)

    def __len__(self): # Returns the current number of experiences stored
        return len(self.buffer)

    def sample(self, batch_size): # Randomly samples a batch of experiences for training
        return random.sample(self.buffer, batch_size)
