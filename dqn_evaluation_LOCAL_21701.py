# -*- coding: utf-8 -*-
import sys
import gym.spaces
import itertools
import os
import readchar
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
import time
import pickle

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


def learn(env,
          env_test,
          env_test1,
          q_func,
          optimizer_spec,
          optimizer_spec_omega,
          optimizer_spec_term,
          session,
          options,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          xi=0.01,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10,
          eval_obs_array=None,
          room_q_interval=1e5,
          epoch_size=5e4,
          log_dir=None,
          transfer_config=None,
          random_length=1000):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    env_test: gym.Env
        gym environment to test on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    optimizer_spec_omega: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    options: Option array
        source policies and primitive actions to be used.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    eval_obs_array: array with shape (num of kinds, num of similar states, shape of a frame)
        obs used to evaluate q values
    room_q_interval: int
        time steps between two q values evaluations
    epoch_size: int
        time steps in an epoch
    log_dir: string
        path to store log
    transfer_config: dict
        contain miscellaneous configurations for transfer learning.
    """

    four_to_two = False
    no_share_para = False

    debug_no_term_train = False

    if transfer_config:
        if transfer_config.has_key('online_q_omega'):
            online_q_omega = transfer_config['online_q_omega']
        if transfer_config.has_key('online_termination'):
            online_termination = transfer_config['online_termination']
        if transfer_config.has_key('q_omega_uniform_sample'):
            q_omega_uniform_sample = transfer_config['q_omega_uniform_sample']
        if transfer_config.has_key('four_to_two'):
            four_to_two = transfer_config['four_to_two']
        if transfer_config.has_key('no_share_para'):
            no_share_para = transfer_config['no_share_para']
        if transfer_config.has_key('debug_no_term_train'):
            debug_no_term_train = transfer_config['debug_no_term_train']

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete
    if four_to_two:
        assert frame_history_len == 4

    ###############
    # BUILD MODEL #
    ###############

    img_h, img_w, img_c = env.observation_space.shape
    input_history_len = 2 if four_to_two else frame_history_len
    input_shape = (img_h, img_w, input_history_len * img_c)
    num_actions = env.action_space.n
    # source + primitive
    num_options = len(options)

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph = tf.placeholder(tf.int32, [None])
    # placeholder for current option
    opt_t_ph = tf.placeholder(tf.int32, [None])
    # placeholder for current options whose actions are the same as action taken (k-hot)
    opa_t_ph = tf.placeholder(tf.float32, [None] + [num_options])
    # placeholder for current reward
    rew_t_ph = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float = tf.cast(obs_t_ph, tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    # Here, you should fill in your own code to compute the Bellman error. This requires
    # evaluating the current and next Q-values and constructing the corresponding error.
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network,
    # respectively. A convenient way to get these is to make use of TF's "scope" feature.
    # For example, you can create your Q-function network with the scope "q_func" like this:
    # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    # And then you can obtain the variables like this:
    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
    ######

    # YOUR CODE HERE
    q_current, q_omega_current, term_current = q_func(obs_t_float, num_actions, num_options, scope='q_func',
                                                      reuse=False, no_share_para=no_share_para)
    q_target, q_omega_target, term_target = q_func(obs_tp1_float, num_actions, num_options, scope='target_q_func',
                                                   reuse=False, no_share_para=no_share_para)
    q_next_current, q_omega_next_current, term_next_current = q_func(obs_tp1_float, num_actions, num_options,
                                                                     scope='q_func', reuse=True,
                                                                     no_share_para=no_share_para)
    # if debug_no_term_train:
    #     term_next_current = term_current = term_target = tf.constant(1.0, dtype=tf.float32, shape=(1, num_options))
    # add some tensor for logs
    u_current = (tf.cast(tf.constant(np.ones(num_options)),
                         tf.float32) - term_current) * q_omega_current + term_current * (tf.reduce_max(q_omega_current,
                                                                                                       axis=-1))[:,
                                                                                        None]
    advantage_current = q_omega_current - tf.reduce_max(q_omega_current, axis=-1)[:, None] + xi
    term_loss = term_current * advantage_current
    tf.summary.histogram('u_current', u_current)
    tf.summary.histogram('advantage_current', advantage_current)
    tf.summary.histogram('term_loss', term_loss)
    tf.summary.histogram('term_current', term_current)
    merged_summary_op = tf.summary.merge_all()
    if log_dir is not None:
        summary_writer = tf.summary.FileWriter(log_dir + 'tfb', session.graph)

    # q_value
    q_val_current = tf.reduce_sum(q_current * tf.one_hot(act_t_ph, num_actions), axis=-1)
    q_val_next_raw = tf.reduce_sum(q_target * tf.one_hot(tf.argmax(q_next_current, axis=-1), num_actions), axis=-1)
    q_val_next = q_val_next_raw * (1 - done_mask_ph)

    # q_value error
    total_error_q = tf.reduce_mean(tf.losses.mean_squared_error(rew_t_ph + gamma * q_val_next, q_val_current))

    # q_omega_value
    # q_omega_val_current = tf.reduce_sum(q_omega_current * tf.one_hot(opt_t_ph, num_options), axis=-1)
    term_val_next = tf.reduce_sum(term_next_current * tf.one_hot(opt_t_ph, num_options), axis=-1)
    q_omega_val_next = tf.reduce_sum(q_omega_next_current * tf.one_hot(opt_t_ph, num_options), axis=-1)
    max_q_omega_next = tf.reduce_max(q_omega_next_current, axis=-1)
    q_omega_val_next_targ = tf.reduce_sum(q_omega_target * tf.one_hot(opt_t_ph, num_options), axis=-1)
    # max_q_omega_next_targ = tf.reduce_max(q_omega_target, axis=-1)
    max_q_omega_next_targ = tf.reduce_sum(
        q_omega_target * tf.one_hot(tf.argmax(q_omega_next_current, axis=-1), num_options), axis=-1)
    # u_val_next_raw = (1 - term_val_next) * q_omega_val_next + term_val_next * max_q_omega_next
    # u_val_next = u_val_next_raw * (1 - done_mask_ph)

    u_next_raw = (1 - term_next_current) * q_omega_target + term_next_current * max_q_omega_next_targ[..., None]
    u_next = u_next_raw * (1 - done_mask_ph)[..., None]

    # q_omega_value error
    total_error_q_omega = tf.reduce_mean(tf.reduce_sum(
        # tf.one_hot(opa_t_ph, num_options) *
        # tf.one_hot(act_t_ph, num_actions) *
        opa_t_ph *
        tf.losses.mean_squared_error(rew_t_ph[..., None] + gamma * u_next, q_omega_current,
                                     reduction=tf.losses.Reduction.NONE),
        axis=-1
    ))

    # optimize termination
    # 这里为什么是stop_gradient
    # 换成另一种写法试一下
    advantage_go = q_omega_val_next - max_q_omega_next + xi
    advantage = tf.stop_gradient(advantage_go)
    # total_error_term = term_val_next * advantage * (1 - done_mask_ph)
    total_error_term = term_val_next * advantage  # 修改

    def term_grad(optimizer, objective, var_list, clip_val=10):
        gradients = optimizer.compute_gradients(objective, var_list=var_list)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val) * advantage_go, var)
        return optimizer.apply_gradients(gradients)

    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    # for log purposes
    q_max = tf.reduce_max(q_current, 1)
    u_max = tf.reduce_max(u_current, 1)
    omega_max = tf.reduce_max(q_omega_current, 1)
    # average_Q = tf.reduce_mean(q_max)
    ######


    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    option_activated = False
    option_running = None
    j = 0
    last_obs = env.reset()
    next_recent_obs = None
    idx = replay_buffer.store_frame(last_obs)

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        # At this point, "last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.
        # Specifically, last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # obs, reward, done, info = env.step(action)
        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # Note that you cannot use "last_obs" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes. The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)

        #####

        # YOUR CODE HERE


        # store the first random_length frames to calculate the average Q

        if next_recent_obs is not None:
            recent_obs = next_recent_obs
        else:
            recent_obs = replay_buffer.encode_recent_observation()

        # pick an option
        if not option_activated:
            if random.random() > 0 and model_initialized:
                q_omega_vals, Term_current, Q_Current, Advantage_Current = session.run(
                    [q_omega_current, term_current, q_current, advantage_current], {obs_t_ph: recent_obs[None, ..., -input_history_len:]})
                option_running = np.argmax(q_omega_vals)
            else:
                option_running = random.choice(range(num_options))
        option_activated = True

        # choose action

        print "########################################################"
        if j < 5:
            # print "noop action"
            action = 5
            j += 1
        else:
            q_omega_vals1, Term_current1, Q_Current1, Advantage_Current1 = session.run(
                [q_omega_current, term_current, q_current, advantage_current],
                {obs_t_ph: recent_obs[None, ..., -input_history_len:]})
            # key = readchar.readkey()
            # print "action", key
            # if key == 's':
            #   action = 1
            # elif key == 'a':
            #   action = 4
            # elif key == 'd':
            #   action = 2
            # elif key == 'w':
            #   action = 3
            # else:
            #   key = int(key)
            #   if key == 0:
            #       action = 0
            #   else:
            #       action = 5
            # action = np.argmax(Q_Current1)
            # time.sleep(0.2)
            action = options[option_running].act(recent_obs)
            print "q_omega", q_omega_vals1
            print "option_running", option_running
            print "best_policy", np.argmax(q_omega_vals1)
            print "beta", Term_current1
            # print "Q", Q_Current1
            print "best action", options[option_running].act(recent_obs)
            # print "advantage_function", Advantage_Current1
            # if np.argmax(Advantage_Current) == np.argmax(Term_current):
            #     print "True"
                # break
            j += 1

        # take a step in the environment
        new_obs, r, done, _ = env.step(action)
        if done:
            print "done"
            new_obs = env.reset()
            j = 0

        opa = np.array(
            [1 if i == option_running or options[i].act(recent_obs) == action else 0 for i in range(num_options)]
        )
        opa[option_running] = 2
        replay_buffer.store_effect(idx, action, r, done, opa)
        idx = replay_buffer.store_frame(new_obs)
        next_recent_obs = replay_buffer.encode_recent_observation()

        if not model_initialized:
            initialize_interdependent_variables(session, tf.global_variables(), {
                obs_t_ph: recent_obs[None, ..., -input_history_len:],
                obs_tp1_ph: next_recent_obs[None, ..., -input_history_len:],
            })
            session.run(update_target_fn)
            model_initialized = True
            saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state(
                "/home/lsy/PycharmProjects/ple-monstrerkong/examples/dqn_transfer_option/logs/12_27nobeta_12_27_17_19:37:26/dqn")

                # "/home/beeperman/Project/ple-monsterkong/examples/dqn_transfer_option/logs/example_old_map_1216c_12_21_17_17:55:35/dqn")
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(session, checkpoint.model_checkpoint_path)
                print("Successfully loaded: ", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")

        term_probs = session.run(term_next_current, {obs_tp1_ph: next_recent_obs[None, ..., -input_history_len:]})
        print("term_probs: " + str(term_probs[0][option_running]))
        random_value = random.random()
        print random_value
        if done or not random_value > term_probs[0][option_running]:  # will re-pick an option
            option_activated = False
            print "change"











