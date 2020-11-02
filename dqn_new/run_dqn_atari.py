import sys
import os
import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time

__file_path = os.path.abspath(__file__)
__ple_dir = '/'.join(str.split(__file_path, '/')[:-3]) + '/'
__cur_dir = '/'.join(str.split(__file_path, '/')[:-1]) + '/'
sys.path.append(__ple_dir)

import dqn
from dqn_utils import *
from atari_wrappers import *


def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_learn(env,
                env_test,
                session,
                num_timesteps=2e7,
                learning_rate=None,
                exploration=None,
                dqn_config=None):
    '''
    fill the hyperparameters before running dqn
    :param env: ai gym env
    :param session: tensorflow session
    :param num_timesteps: int
    :param learning_rate: piecewise function
    :param exploration: piecewise function
    :param dqn_config: will override parameters above
    :return: none
    '''


    replay_buffer_size = 1000000
    batch_size = 32
    gamma = 0.99
    learning_starts = 50000
    learning_freq = 4
    frame_history_len = 4
    target_update_freq = 10000
    grad_norm_clipping = 10
    eval_obs_array = None
    room_q_interval = 1e5
    epoch_size = 5e3
    config_name = None


    if dqn_config:
        if dqn_config.has_key('num_timesteps'):
            num_timesteps = dqn_config['num_timesteps']
        if dqn_config.has_key('replay_buffer_size'):
            replay_buffer_size = dqn_config['replay_buffer_size']
        if dqn_config.has_key('batch_size'):
            batch_size = dqn_config['batch_size']
        if dqn_config.has_key('gamma'):
            gamma = dqn_config['gamma']
        if dqn_config.has_key('learning_starts'):
            learning_starts = dqn_config['learning_starts']
        if dqn_config.has_key('learning_freq'):
            learning_freq = dqn_config['learning_freq']
        if dqn_config.has_key('frame_history_len'):
            frame_history_len = dqn_config['frame_history_len']
        if dqn_config.has_key('target_update_freq'):
            target_update_freq = dqn_config['target_update_freq']
        if dqn_config.has_key('grad_norm_clipping'):
            grad_norm_clipping = dqn_config['grad_norm_clipping']
        if dqn_config.has_key('learning_rate'):
            learning_rate = dqn_config['learning_rate']
        if dqn_config.has_key('exploration'):
            exploration = dqn_config['exploration']
        if dqn_config.has_key('eval_obs_array'):
            eval_obs_array = dqn_config['eval_obs_array']
        if dqn_config.has_key('room_q_interval'):
            room_q_interval = dqn_config['room_q_interval']
        if dqn_config.has_key('epoch_size'):
            epoch_size = dqn_config['epoch_size']
        if dqn_config.has_key('config_name'):
            config_name = dqn_config['config_name']


    # log_dir = __cur_dir + 'logs/' + config_name + '_' + time + '/'
    cur_time = time.strftime("%m_%d_%y_%H:%M:%S", time.localtime(time.time()))
    log_dir = __cur_dir + 'logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if config_name != None:
        log_dir = log_dir + config_name + '_' + cur_time + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        dqn_network_dir = log_dir + 'dqn/'
        if not os.path.exists(dqn_network_dir):
            os.makedirs(dqn_network_dir)
        pkl_dir = log_dir + 'pkl/'
        if not os.path.exists(pkl_dir):
            os.makedirs(pkl_dir)
    else:
        log_dir = None
        print("config_name not specified! info may not be logged in this run.")


    # This is just a rough estimate
    num_iterations = float(num_timesteps) / learning_freq

    if learning_rate != None:
        lr_schedule = learning_rate
    else:
        lr_multiplier = 1.0
        lr_schedule = PiecewiseSchedule([
            (0,                   1e-4 * lr_multiplier),
            (num_iterations / 10, 1e-4 * lr_multiplier),
            (num_iterations / 2,  5e-5 * lr_multiplier),
        ],
            outside_value=5e-5 * lr_multiplier)

    if exploration != None:
        exploration_schedule = exploration
    else:
        exploration_schedule = PiecewiseSchedule(
            [
                (0, 1.0),
                (1e6, 0.1),
                (num_iterations / 2, 0.01),
            ], outside_value=0.01
        )

    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):        # notice that here t is the number of steps of the wrapped env,

        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps


    dqn.learn(
        env,
        env_test,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        learning_starts=learning_starts,
        learning_freq=learning_freq,
        frame_history_len=frame_history_len,
        target_update_freq=target_update_freq,
        grad_norm_clipping=grad_norm_clipping,
        eval_obs_array=eval_obs_array,
        room_q_interval=room_q_interval,
        epoch_size=epoch_size,
        log_dir=log_dir
    )
    env.close()
    env_test.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '~/tmp/transfer-gym/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def wrap_env(env, seed):  # non-atari
    env.seed(seed)

    expt_dir = '~/tmp/transfer-gym/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)


    return env

def main():
    # Get arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('map_config', type=str,
                        help='The map and config you want to run in MonsterKong.')
    args = parser.parse_args()

    import imp
    try:
        map_config_file = args.map_config
        map_config = imp.load_source('map_config', map_config_file).map_config
    except Exception as e:
        sys.exit(str(e) + '\n'
                 +'map_config import error. File not exist or map_config not specified')

    # Get MonsterKong game.
    from gym.envs.registration import register

    register(
        id='MonsterKong-v0',
        entry_point='ple.gym_env.monsterkong:MonsterKongEnv',
        kwargs={'map_config': map_config},
    )

    num_timesteps = 2e7  # max timesteps in the training

    env = gym.make('MonsterKong-v0')
    env = ProcessFrame(env)
    env_test = gym.make('MonsterKong-v0')
    env_test = ProcessFrame(env_test)

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    dqn_config = None
    if map_config.has_key('dqn_config'):
        dqn_config = map_config['dqn_config']

    if dqn_config:
        if dqn_config.has_key('seed'):
            if callable(dqn_config['seed']):
                seed = dqn_config['seed']()
            else:
                seed = dqn_config['seed']
        if dqn_config.has_key('additional_wrapper'):
            env = dqn_config['additional_wrapper'](env)
            env_test = dqn_config['additional_wrapper'](env_test)

    env = wrap_env(env, seed)
    env_test = wrap_env(env_test, seed)
    session = get_session()
    atari_learn(env, env_test, session, num_timesteps=num_timesteps, dqn_config=dqn_config)


if __name__ == "__main__":
    main()
