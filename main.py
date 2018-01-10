import gym
import sys, os

import ppaquette_gym_super_mario
import readchar
import tensorflow as tf
import numpy as np

from dqn import DQN
from utils import get_copy_var_ops

from wrappers import MarioActionSpaceWrapper, ProcessFrame84
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# MACROS
LEFT = 3
DOWN = 2
RIGHT = 7
UP = 1

# Key mapping
arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT
}


def main():

    MAX_BUFFER_SIZE = 100000
    MAX_EPISODES = 10000
    TRAIN_EPISODE = 100
    TARGET_UPDATE_EPS = 1000

    batch_size = 32
    n_size = 84
    discount = 0.99
    
    # 1. Create gym environment
    env = gym.make("ppaquette/SuperMarioBros-1-1-v0")
    # 2. Apply action space wrapper
    env = MarioActionSpaceWrapper(env)
    # 3. Apply observation space wrapper to reduce input size
    env = ProcessFrame84(env)

    #replay_buffer = PrioritizedReplayBuffer(MAX_BUFFER_SIZE, alpha=prioritized_replay_alpha)
    replay_buffer = ReplayBuffer(MAX_BUFFER_SIZE)
    sess = tf.Session()

    mainDQN = DQN(sess, name="main")
    targetDQN = DQN(sess, name="target")

    sess.run(tf.global_variables_initializer())

    copy_ops = get_copy_var_ops(dest_scope_name="target",
                                src_scope_name="main")
    sess.run(copy_ops)


    for eps in range(MAX_EPISODES):
        # decaying epsilon greedy
        e = 1. / ((eps / 10) + 1)
        done = False
        step_count = 0
        state = env.reset()
        state_s = state

        
        while not done:
            step_count += 1
            print(step_count)
            if step_count == 1 or np.random.rand() < e:
                action = env.action_space.sample()
            else:
                # Choose an action by greedily from the Q-network
                action = np.argmax(mainDQN.predict(state_s))

            # Get new state and reward from environment
            state_stack = []
            for _ in range(4):
                next_state, reward, done, _ = env.step(action)

                if done:  # Penalty
                    reward = -1
                state_stack.append(next_state)
                replay_buffer.add(state, action, reward, next_state, done)

            if step_count % TRAIN_EPISODE == 0:
                states, actions, rewards, next_states, _ = replay_buffer.sample(batch_size * 4)
                states, next_states = np.reshape(states, [batch_size, n_size, n_size, 4]), np.reshape(next_states, [batch_size, n_size, n_size, 4])
                Q_t = targetDQN.predict(next_states)
                Q_m = mainDQN.predict(states)
                Q_t = np.max(Q_t, axis=1)
                rewards = np.mean(np.reshape(rewards, [batch_size, 4]), 1)
                estimates = rewards + discount * Q_t
                actions = np.reshape(actions, [batch_size, 4])
                actions = np.reshape(actions[::4], [batch_size])
                Q_m[np.arange(batch_size), actions] = estimates
                loss = mainDQN.update(states, Q_m)
                print("eps: {} step: {} loss: {}".format(eps, step_count, loss))

            if step_count % TARGET_UPDATE_EPS == 0:
                sess.run(copy_ops)

            state_s = np.reshape(np.array(state_stack), [1, n_size, n_size, 4])


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    main()