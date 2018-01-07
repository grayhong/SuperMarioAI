import gym
import sys

import ppaquette_gym_super_mario
import readchar

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

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
    self.action_space = action_space

    def act(self, observation, reward, done):
    return self.action_space.sample()


def main():

    MAX_BUFFER_SIZE = 100000
    MAX_EPISODES = 10000
    TRAIN_EPISODE = 100
    TARGET_UPDATE_EPS = 1000

    batch_size = 32
    discount = 0.99

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
        e = 1. / ((episode / 10) + 1)
        done = False
        step_count = 0
        state = env.reset()

        
        while not done:
            step_count += 1
            if np.random.rand() < e:
                action = env.action_space.sample()
            else:
                # Choose an action by greedily from the Q-network
                action = np.argmax(mainDQN.predict(state))

            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(action)

            if done:  # Penalty
                reward = -1

            replay_buffer.add(state, action, reward, next_state, done)

            if step_count % TRAIN_EPISODE == 0:
                states, actions, rewards, next_states, done = replay_buffer.sample(batch_size)
                Q_t = targetDQN.predict(next_states)
                Q_m = mainDQN.predict(states)
                Q_t = np.max(Q_t, axis=1)
                estimates = rewards + discount * Q_t
                Q_m[range(len(states)), actions]
                loss = mainDQN.train(states, Q_m)
                print("eps: {} step: {} loss: {}".format(eps, step_count, loss))

            if step_count % TARGET_UPDATE_EPS == 0:
                sess.run(copy_ops)

            state = next_state


if __name__ == '__main__':
    main()