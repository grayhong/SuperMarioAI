import gym
import sys, os

import ppaquette_gym_super_mario
import readchar
import tensorflow as tf
import numpy as np

from dqn import DQN
from utils import get_copy_var_ops
from collections import deque

from wrappers import MarioActionSpaceWrapper, ProcessFrame84
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

def main():

    MAX_BUFFER_SIZE = 100000
    MAX_EPISODES = 10000
    TRAIN_EPISODE = 100
    TARGET_UPDATE_EPS = 1000

    batch_size = 32
    n_size = 84
    discount = 0.99

    checkpoint_dir = './checkpoints'
    save_file_name = 'mario_weight.ckpt'
    
    # 1. Create gym environment
    env = gym.make("ppaquette/SuperMarioBros-1-1-v0")
    # 2. Apply action space wrapper
    env = MarioActionSpaceWrapper(env)
    # 3. Apply observation space wrapper to reduce input size
    env = ProcessFrame84(env)

    sess = tf.Session()
    targetDQN = DQN(sess, name="target")
    dqn_var_list = targetDQN.var_list

    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(var_list = dqn_var_list)
    saver.restore(sess, os.path.join(checkpoint_dir, save_file_name))


    for eps in range(MAX_EPISODES):
        done = False
        step_count = 0
        state = env.reset()

        state_queue = deque(maxlen=4)
        state_queue.append(state)

        
        while not done:
            step_count += 1

            # cumulate 4 frames
            if step_count < 4:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                state_queue.append(next_state)
                continue

            action = np.argmax(targetDQN.predict(np.reshape(np.array(state_queue), [1, n_size, n_size, 4])))

            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(action)
            state_queue.append(next_state)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    main()