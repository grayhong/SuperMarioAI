import gym
import sys

import ppaquette_gym_super_mario
import readchar

from wrappers import MarioActionSpaceWrapper, ProcessFrame84

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
  # 1. Create gym environment
  env = gym.make("ppaquette/SuperMarioBros-1-1-v0")
  # 2. Apply action space wrapper
  env = MarioActionSpaceWrapper(env)
  # 3. Apply observation space wrapper to reduce input size
  env = ProcessFrame84(env)

  agent = RandomAgent(env.action_space)

  episode_count = 100
  reward = 0
  done = False
  # for i in range(episode_count):
  #   ob = env.reset()
  #   while True:
  #     action = agent.act(ob, reward, done)
  #     ob, reward, done, _ = env.step(1)
  #     if done:
  #       break

  for i in range(episode_count):
    ob = env.reset()
    while True:
      key = readchar.readkey()
      # Choose an action from keyboard
      if key not in arrow_keys.keys():
          print("Game aborted!")
          break
      action = arrow_keys[key]
      state, reward, done, info = env.step(action)

      if done:
        print("Finished with reward", reward)
        break





if __name__ == '__main__':
  main()