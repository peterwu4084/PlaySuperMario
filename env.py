import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT, RIGHT_ONLY



def build_env(env_name, action_complexity=3):
    # env_name: 'SuperMarioBros-{world}-{stage}-v{version}'
    #            worlds (1-8), stages (1-4), version (0-3)
    # action_complexity: 0-3 (the larger, the more complex)
    env = gym_super_mario_bros.make(env_name)
    if action_complexity == 3:
        return env
    else:
        action_space = (RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT)[action_complexity]
        return JoypadSpace(env, action_space)