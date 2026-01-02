import gymnasium as gym
from tetris_gymnasium.mappings.rewards import RewardsMapping
import numpy as np


# https://gymnasium.farama.org/api/wrappers/#methods
class MyReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_max_height = None
        self.prev_holes = None
        self.prev_bumpiness = None
        # self.alife_p = env.unwrapped.rewards.alife

        # self.prev_state = None

    # https://stackoverflow.com/questions/73675262/openai-gym-problem-override-observationwrapper-reset-method
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # data from the last block placement
        self.prev_max_height = obs[10]
        self.prev_holes = obs[11]
        self.prev_bumpiness = obs[12]
        # ata from the last place step
        self.last_holes = obs[11]
        self.last_bumpiness = obs[12]

        self.prev_state = np.zeros(10)
        # obs = np.concatenate([obs, self.prev_state])

        return obs, info

    def step(self, action):
        obs, reward, term, trun, info = self.env.step(action)

        """ https://github.com/Max-We/Tetris-Gymnasium/blob/main/tetris_gymnasium/envs/tetris.py
        def score(self, rows_cleared) -> int:
            return (rows_cleared**2) * self.width """
        if reward > 1:
            reward = 0.01

        lines_cleared = info.get("lines_cleared", 0)
        if lines_cleared > 0:
            # reward /= 10
            reward += (lines_cleared**2) + 1
            # print("lines_creared: ", lines_cleared)

        """
        https://github.com/Max-We/Tetris-Gymnasium/blob/main/tetris_gymnasium/wrappers/observation.py#L114
        observation space is 1D vector "obs":
        - The height of each column: obs[0...9]
        - The maximum height : obs[10]
        - The number of holes : obs[11]
        - The bumpiness : obs[12]
        """
        max_height = obs[10]
        holes = obs[11]
        bumpiness = obs[12]

        # when the block is placed
        if reward > 0:
            # delta (have all been divided by 20 in the observation function)
            d_holes = holes - self.prev_holes
            d_bump = bumpiness - self.prev_bumpiness
            d_height = max_height - self.prev_max_height

            reward += - 0.05 * d_height * max_height -0.01 *max_height
            reward += -0.5 * d_holes - 0.05 * d_bump  # - 0.01 * d_height
            # self.prev_max_height = max_height
            # self.prev_bumpiness = bumpiness
            self.prev_holes = holes
            if d_holes == 0:
                reward += 0.8
            # print("0 delta hole")

            # print("delta hole: ", d_holes)
            self.env.prev_state = obs[:10]

            if self.env.obs_size in (26, 32):
                obs[-10:] = self.env.prev_state

        if self.env.obs_size not in (26, 32):
            obs[:10] = self.env.prev_state

        # # when the block is in the air
        # """ if reward == 0:
        #     d_holes = holes - self.last_holes
        #     d_bump = bumpiness - self.last_bumpiness

        #     reward += -0.02 * d_holes - 0.1 * d_bump

        #     self.last_holes = holes
        #     self.last_bumpiness = bumpiness """

        #if action == 6:  # swap
        #    reward -= 0.1

        if action == 5:  # drop
            reward += 0.01

        return obs, np.float32(reward), term, trun, info
