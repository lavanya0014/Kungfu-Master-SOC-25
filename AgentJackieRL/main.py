import gymnasium as gym
import ale_py
import numpy as np
import cv2
import tensorflow as tf
from collections import deque
import random
from agent import DQNAgent
from memory import Replay
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
import os


# Setting seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("videos", exist_ok=True)

# Creating and preprocessing the environment
env = gym.make("ALE/KungFuMaster-v5", frameskip=1,render_mode="rgb_array")
env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda e: e % 100 == 0, name_prefix="agent_play")
env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4, screen_size=84)
env = FrameStackObservation(env, 4)

# Environment details
state_shape = env.observation_space.shape
n_actions = env.action_space.n

# Initializing agent and replay buffer
agent = DQNAgent(state_shape, n_actions)
memory = Replay(capacity=100_000)

# Training hyperparameters
EPISODES = 5000
BATCH_SIZE = 32
START_TRAINING_AFTER = 100
TARGET_UPDATE_EVERY = 1000
MAX_STEPS_PER_EPISODE = 10000
total_steps = 0
best_reward = -float("inf")  # saving best model

for episode in range(1, EPISODES + 1):
    state, _ = env.reset()
    state = np.array(state)
    state = np.transpose(np.array(state), (1, 2, 0))
    total_reward = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.transpose(np.array(next_state), (1, 2, 0))
        done = terminated or truncated
        next_state = np.array(next_state)

        memory.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        total_steps += 1


        if len(memory) > START_TRAINING_AFTER:
            agent.train(memory)

        if total_steps % TARGET_UPDATE_EVERY == 0:
            agent.update_target_network()

        if done:
            break

    print(f"Episode {episode} - Reward: {total_reward:.1f} - Epsilon: {agent.epsilon:.3f}")
    # Manual epsilon decay for debugging
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon -= agent.epsilon_decay
        agent.epsilon = max(agent.epsilon, agent.epsilon_min)


    #Saving the best model
    if total_reward > best_reward:
        agent.save("best_agent.keras")
        best_reward = total_reward


env.close()