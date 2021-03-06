"""
This is the example config file
same_room
no parameter share
take a look at transfer_config (differences are there)
"""
import numpy as np

# More one-char representation will be added in order to support
# other objects.
# The following a=10 is an example although it does not work now
# as I have not included a '10' object yet.
a = 10

# This is the map array that represents the map
# You have to fill the array into a (m x n) matrix with all elements
# not None. A strange shape of the array may cause malfunction.
# Currently available object indices are # they can fill more than one element in the array.
# 0: nothing
# 1: wall
# 2: ladder
# 3: coin
# 4: spike
# 5: triangle -------source
# 6: square ------ source
# 7: coin -------- target
# 8: princess -------source
# 9: player # elements(possibly more than 1) filled will be selected randomly to place the player
# unsupported indices will work as 0: nothing
map_array = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 5, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 9, 9, 9, 9, 1, 9, 9, 9, 8, 1],
    [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [1, 0, 0, 2, 0, 0, 0, 2, 0, 7, 1],
    [1, 9, 9, 2, 9, 9, 9, 2, 9, 9, 1],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1],
    [1, 0, 2, 0, 1, 0, 2, 0, 0, 0, 1],
    [1, 0, 2, 0, 1, 0, 2, 0, 6, 0, 1],
    [1, 9, 9, 9, 1, 9, 9, 9, 9, 9, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]


# set to true -> win when touching the object
# 0, 1, 2, 3, 4, 9 are not possible
end_game = {
    6: True,
}

rewards = {
    "positive": 0,      # when collecting a coin
    "win": 1,          # endgame (win)
    "negative": -25,    # endgame (die)
    "tick": 0           # living
}


######### dqn only ##########
# ensure correct import
import os
import sys
__file_path = os.path.abspath(__file__)
__dqn_dir = '/'.join(str.split(__file_path, '/')[:-2]) + '/'
sys.path.append(__dqn_dir)
__cur_dir = '/'.join(str.split(__file_path, '/')[:-1]) + '/'

from dqn_utils import PiecewiseSchedule, NoOpWrapperMK

# load the random sampled obs
import pickle
# pkl_file = __cur_dir + 'eval_obs_array_random_batch.pkl'
pkl_file = '/home/beeperman/Project/ple-monsterkong/examples/dqn_get_eval_obs/eval_obs_array_batch_1221.pkl'
with open(pkl_file, 'rb') as f:
   eval_obs_array = pickle.loads(f.read())


def seed_func():
    return np.random.randint(0, 1000)

num_timesteps = 2.5e7
learning_freq = 4
# training iterations to go
num_iter = num_timesteps / learning_freq

# piecewise learning rate
lr_multiplier = 1.0
learning_rate = PiecewiseSchedule([
    (0, 2e-4 * lr_multiplier),
    (num_iter / 2, 1e-4 * lr_multiplier),
    (num_iter * 3 / 4,  5e-5 * lr_multiplier),
], outside_value=5e-5 * lr_multiplier)

learning_rate_term = PiecewiseSchedule([
    (0, 2e-2 * lr_multiplier),
    (num_iter / 2, 1e-2 * lr_multiplier),
    (num_iter * 3 / 4,  5e-3 * lr_multiplier),
], outside_value=5e-4 * lr_multiplier)

# piecewise exploration rate
exploration = PiecewiseSchedule([
    (0, 1.0),
    (num_iter / 8, 0.7),
    (num_iter * 3 / 16, 0.1),
    (num_iter * 7 / 32, 0.05),
], outside_value=0.05)

######### transfer only #########

source_dirs = [
    # an old map policy
    # '/home/lsy/logs/target6c_12_05_17_21:26:25/dqn',
    # '/home/lsy/PycharmProjects/ple-monstrerkong/examples/dqn_new/logs/target5_12_05_17_19:49:45',
    # '/home/lsy/target8c_12_10_17_15:25:06/dqn',
    # '/home/beeperman/Project/ple-monsterkong/examples/dqn_transfer_option/logs/same6',
    #'/home/lsy/same_room_12_12_17_20:54:53/dqn',
    # '/home/lsy/PycharmProjects/ple-monstrerkong/examples/dqn_new/logs/12_20same_room6_12_20_17_21:15:55/dqn',
    # '/home/lsy/source/dqn7',
    # '/home/lsy/source/dqn8',
    '/home/lsy/source/different/12_21target5_12_21_17_22:04:21/dqn',
    '/home/lsy/source/different/12_21target7c_12_23_17_15:28:31/dqn',
    '/home/lsy/source/different/12_21target8_12_23_17_15:29:51/dqn',
]

transfer_config = {
    'source_dirs': source_dirs,
    'online_q_omega': False,    # default false off policy with experience replay
    'q_omega_uniform_sample': False,    # default false
    'four_to_two': False,  # default false frame_history_len must be 4!
    'source_noop': True,  # default false (false means source policies HAS noop action)
    'no_share_para': True,  # default false set to true to stop sharing parameter between q network and q_omega/term
    'xi': 0.01,                # default none you may specify a constant. none means xi = 0.5 (q_omega_val - q_omega_second_max)
    'target_beta': False,    # default false (true means using target beta)
    'termination_stop': True,    # default false train cnn when training beta online
    'learning_rate_term': learning_rate_term,
}


dqn_config = {
    'seed': seed_func,  # will override game settings
    'num_timesteps': num_timesteps,
    'replay_buffer_size': 1000000,
    'batch_size': 32,
    'gamma': 0.99,
    'learning_starts': 1e5,
    'learning_freq': learning_freq,
    'frame_history_len': 2,
    'target_update_freq': 10000,
    'grad_norm_clipping': 10,
    'learning_rate': learning_rate,
    'exploration': exploration,
    # 'additional_wrapper': NoOpWrapperMK,
    'eval_obs_array': eval_obs_array,  # TODO: construct some eval_obs_array
    'room_q_interval': 5e4,  # q_vals will be evaluated every room_q_interval steps
    'epoch_size': 5e4,  # you decide any way
    'config_name': str.split(__file_path, '/')[-1].replace('.py', ''),  # the config file name
    'transfer_config': transfer_config,
}


map_config = {
    'map_array': map_array,
    'rewards': rewards,
    'end_game': end_game,
    'init_score': 0,
    'init_lives': 1,  # please don't change, not going to work
    # configs for dqn
    'dqn_config': dqn_config,
    # work automatically only for aigym wrapped version
    'fps': 30,
    'frame_skip': 1,
    'force_fps': True,  # set to true to make the game run as fast as possible
    'display_screen': False,
    'episode_length': 1200,
    'episode_end_sleep': 0.,  # sec
}
