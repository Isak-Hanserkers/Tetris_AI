from envs import tetris_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env(lambda: tetris_env.make_env(render=False, obs_size=22), n_envs=8)

model = PPO.load("continuation/model/PPO_1_5_2_1_1", vec_env)
# (alife=0.05, game_over=-10)
# PPO{obs_size}_{T-steps}_{d_holes==0 -> +0.a}_{-0.b * d_holes} _{clear_line^c * 1 + 1}_{-0.d * d_height}_{-0.0e * d_bump}_{-0.0f * max_height}
model.learn(
    total_timesteps=50_000_000,
    tb_log_name="PPO22_2_5_2_1_3_1_2",
    reset_num_timesteps=False,
)
model.save("continuation/model/PPO_1_5_2_1_3_1_2")

""" 
uv run python -m con.con_train
tensorboard --logdir=con/log

 """


# def step(self, action):
#     obs, reward, term, trun, info = self.env.step(action)

#     if reward > 1:
#         reward = 0.01

#     lines_cleared = info.get("lines_cleared", 0)
#     if lines_cleared > 0:
#         reward += (lines_cleared**2) * 1 + 2

#     max_height = obs[10]
#     holes = obs[11]
#     bumpiness = obs[12]

#     # when the block is placed
#     if reward > 0:
#         # delta (have all been divided by 20 in the observation function)
#         d_holes = holes - self.prev_holes
#         d_bump = bumpiness - self.prev_bumpiness
#         d_height = max_height - self.prev_max_height

#         # reward += -0.001 * max_height - 0.005 * holes - 0.001 * bumpiness
#         reward += -0.5 * d_holes - 0.01 * d_height - 0.01 * d_bump
#         # self.prev_max_height = max_height
#         # self.prev_bumpiness = bumpiness
#         self.prev_holes = holes
#         if d_holes == 0:
#             reward += 0.1
#         # print("0 delta hole")

#         # print("delta hole: ", d_holes)
#         self.env.prev_state = obs[:10]

#         if self.env.obs_size in (26, 32):
#             obs[-10:] = self.env.prev_state

#     if self.env.obs_size not in (26, 32):
#         obs[:10] = self.env.prev_state

#     if action == 6:  # swap
#         reward -= 0.05

#     return obs, np.float32(reward), term, trun, info
