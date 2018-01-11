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
    save_file_name = 'mario_weight_server.ckpt'
    
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
    dqn_var_list = targetDQN.var_list

    sess.run(tf.global_variables_initializer())

    copy_ops = get_copy_var_ops(dest_scope_name="target",
                                src_scope_name="main")
    sess.run(copy_ops)
    
    saver = tf.train.Saver(var_list = dqn_var_list)


    for eps in range(MAX_EPISODES):
        # decaying epsilon greedy
        e = 1. / ((eps / 10) + 1)
        done = False
        step_count = 0
        state = env.reset()
        state_queue = deque(maxlen=4)
        next_state_queue = deque(maxlen=4)

        state_queue.append(state)
        next_state_queue.append(state)

        
        while not done:
            step_count += 1

            # cumulate 4 frames
            if step_count < 4:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                state_queue.append(next_state)
                next_state_queue.append(next_state)
                continue

            # training starts
            if np.random.rand() < e:
                action = env.action_space.sample()
            else:
                # Choose an action by greedily from the Q-network
                action = np.argmax(mainDQN.predict(np.reshape(np.array(state_queue), [1, n_size, n_size, 4])))

            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(action)

            if done:  # Penalty
                reward = -1

            next_state_queue.append(next_state)

            replay_buffer.add(np.array(state_queue), action, reward, np.array(next_state_queue), done)

            if step_count % TRAIN_EPISODE == 0:
                states, actions, rewards, next_states, _ = replay_buffer.sample(batch_size)
                states, next_states = np.reshape(states, [batch_size, n_size, n_size, 4]), np.reshape(next_states, [batch_size, n_size, n_size, 4])
                
                Q_t = targetDQN.predict(next_states)
                Q_m = mainDQN.predict(states)
                Q_t = np.max(Q_t, axis=1)

                estimates = rewards + discount * Q_t
                Q_m[np.arange(batch_size), actions] = estimates

                loss = mainDQN.update(states, Q_m)
                print("eps: {} step: {} loss: {}".format(eps, step_count, loss))

            if step_count % TARGET_UPDATE_EPS == 0:
                sess.run(copy_ops)
                save_path = saver.save(sess, os.path.join(checkpoint_dir, save_file_name))
                print("Model saved in file: %s" % save_path)

            state_queue.append(next_state)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    main()
