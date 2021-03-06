# -*- coding: utf-8 -*-
import sys
import gym.spaces
import itertools
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import tensorflow                as tf
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
          xi=None,
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

    Costs = []
    loss_term_log = []
    average_loss_term = []
    loss_omega_log = []
    average_loss_omega = []
    Average_costs = []
    test_rewards = []
    test_rewards1= []
    none_discount = []
    none_discount1 = []
    short_average = []
    short_average1 = []
    short_average_none = []
    short_average_none1 = []
    test_q_max_log = []
    test_u_log = []
    test_omega_log = []
    Episode_num = 0

    online_q_omega = False
    online_termination = True
    q_omega_uniform_sample = False
    four_to_two = False
    no_share_para = False
    target_beta = False
    termination_stop = False
    beta_no_bias = False

    debug_no_term_train = False

    obs_random_batch = None

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
        if transfer_config.has_key('target_beta'):
            target_beta = transfer_config['target_beta']
        if transfer_config.has_key('xi'):
            xi = transfer_config['xi']
        if transfer_config.has_key('termination_stop'):
            termination_stop = transfer_config['termination_stop']
        if transfer_config.has_key('beta_no_bias'):
            beta_no_bias = transfer_config['beta_no_bias']

        if transfer_config.has_key('debug_no_term_train'):
            debug_no_term_train = transfer_config['debug_no_term_train']

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete
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
    obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current option
    opt_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current options whose actions are the same as action taken (k-hot)
    opa_t_ph              = tf.placeholder(tf.float32,   [None] + [num_options])
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph          = tf.placeholder(tf.float32, [None])
    Episode_reward        = tf.placeholder(tf.float32)

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
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
    q_current,      q_omega_current,        term_current = q_func(obs_t_float, num_actions, num_options, scope='q_func', reuse=False, no_share_para=no_share_para, termination_stop=termination_stop, beta_no_bias=beta_no_bias)
    q_target,       q_omega_target,         term_target = q_func(obs_tp1_float, num_actions, num_options, scope='target_q_func', reuse=False, no_share_para=no_share_para, termination_stop=termination_stop, beta_no_bias=beta_no_bias)
    q_next_current, q_omega_next_current,   term_next_current = q_func(obs_tp1_float, num_actions, num_options, scope='q_func', reuse=True, no_share_para=no_share_para, termination_stop=termination_stop, beta_no_bias=beta_no_bias)
    if debug_no_term_train:
        term_next_current_old = term_next_current = term_current = term_target = tf.constant(1.0, dtype=tf.float32, shape=(1, num_options))



    # q_value
    q_val_current = tf.reduce_sum(q_current * tf.one_hot(act_t_ph, num_actions), axis=-1)
    q_val_next_raw = tf.reduce_sum(q_target * tf.one_hot(tf.argmax(q_next_current, axis=-1), num_actions), axis=-1)
    q_val_next = q_val_next_raw * (1 - done_mask_ph)

    # q_value error
    total_error_q = tf.reduce_mean(tf.losses.mean_squared_error(rew_t_ph + gamma * q_val_next, q_val_current))
    # q_omega_value
    term_val_next = tf.reduce_sum(term_next_current * tf.one_hot(opt_t_ph, num_options), axis=-1)
    q_omega_val_next = tf.reduce_sum(q_omega_next_current * tf.one_hot(opt_t_ph, num_options), axis=-1)
    max_q_omega_next = tf.reduce_max(q_omega_next_current, axis=-1)
    max_q_omega_next_targ = tf.reduce_sum(q_omega_target * tf.one_hot(tf.argmax(q_omega_next_current, axis=-1), num_options), axis=-1)

    if target_beta:  # change var
        term_next_current_old = term_next_current
        term_next_current = term_target

    u_next_raw = (1 - term_next_current) * q_omega_target + term_next_current * max_q_omega_next_targ[..., None]
    u_next = tf.stop_gradient(u_next_raw * (1 - done_mask_ph)[..., None])

    if target_beta:  # row back
        term_next_current = term_next_current_old

    # q_omega_value error
    total_error_q_omega = tf.reduce_mean(tf.reduce_sum(
        opa_t_ph *
        tf.losses.mean_squared_error(rew_t_ph[..., None] + gamma * u_next, q_omega_current, reduction=tf.losses.Reduction.NONE),
        axis=-1
    ))

    # optimize termination
    if xi == None:
        xi = 0.8 * (max_q_omega_next - tf.nn.top_k(q_omega_next_current, 2)[0][:, 1])
    advantage_go = q_omega_val_next - max_q_omega_next + xi
    advantage = tf.stop_gradient(advantage_go)
    # total_error_term = term_val_next * advantage * (1 - done_mask_ph)
    total_error_term = term_val_next * advantage  #修改
    tf.summary.scalar('total_error_term', total_error_term)
    # def term_grad(optimizer, objective, var_list, clip_val=10):
    #     gradients = optimizer.compute_gradients(objective, var_list=var_list)
    #     for i, (grad, var) in enumerate(gradients):
    #         if grad is not None:
    #             gradients[i] = (tf.clip_by_norm(grad, clip_val) * advantage_go, var)
    #     return optimizer.apply_gradients(gradients)

    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # print len(q_func_vars)
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')
    # print len(target_q_func_vars)


    # for log purposes
    # add some tensor for logs
    u_current = (tf.cast(tf.constant(np.ones(num_options)),
                         tf.float32) - term_current) * q_omega_current + term_current * (tf.reduce_max(q_omega_current,
                                                                                                       axis=-1))[:, None]
    max_q_omega = tf.reduce_max(q_omega_current, axis=-1)
    values1 = tf.nn.top_k(q_omega_current, 2)[0]
    advantage_current = q_omega_current - max_q_omega[:, None] + 0.8 * (max_q_omega - values1[:, 1])[:, None]
    term_loss = term_current * advantage_current
    tf.summary.histogram('u_current', u_current)
    tf.summary.histogram('advantage_current', advantage_current)
    tf.summary.histogram('term_loss', term_loss)
    tf.summary.histogram('term_current', term_current, family='term_current')
    [tf.summary.histogram('term_current_%d' % (i), tf.reduce_mean(term_current[:, i]), family='term_current') for i in range(num_options)]
    [tf.summary.scalar('term_current_%d' % (i), tf.reduce_mean(term_current[:, i]), family='term_vals') for i in range(num_options)]
    # summaries_first = [tf.summary.histogram('u_current', u_current),
    #                    tf.summary.histogram('advantage_current', advantage_current),
    #                    tf.summary.histogram('term_loss', term_loss),
    #                    tf.summary.histogram('term_current', term_current, family='term_current'),
    #                    [tf.summary.histogram('term_current_%d' % (i), tf.reduce_mean(term_current[:, i]), family='term_current') for i in range(num_options)],
    #                    [tf.summary.scalar('term_current_%d' % (i), tf.reduce_mean(term_current[:, i]), family='term_vals') for i in range(num_options)]]
    # summaries_diff = [tf.summary.histogram('u_current11', u_current),
    #                    tf.summary.histogram('advantage_current1', advantage_current),
    #                    tf.summary.histogram('term_loss1', term_loss),
    #                    tf.summary.histogram('term_current1', term_current, family='term_current1'),
    #                    [tf.summary.histogram('term_current1_%d' % (i), tf.reduce_mean(term_current[:, i]),
    #                                          family='term_current1') for i in range(num_options)],
    #                    [tf.summary.scalar('term_current1_%d' % (i), tf.reduce_mean(term_current[:, i]),
    #                                       family='term_vals1') for i in range(num_options)]]
    # merged_summary_diff=tf.summary.merge(summaries_diff)
    # merged_summary_op = tf.summary.merge(summaries_first)
    merged_summary_op = tf.summary.merge_all()
    # summary_second = [tf.summary.scalar("my_second_graph_loss", Episode_reward)]
    # merged_summary_op1 = tf.summary.merge(summary_second)
    summary_writer = tf.summary.FileWriter(log_dir + 'tfb', session.graph)

    q_max = tf.reduce_max(q_current, 1)
    u_max = tf.reduce_max(u_current, 1)
    omega_max = tf.reduce_max(q_omega_current, 1)
    average_Q = tf.reduce_mean(q_max)
    average_U = tf.reduce_mean(u_max)
    average_omega = tf.reduce_mean(omega_max)
    ######

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    learning_rate_omega = tf.placeholder(tf.float32, (), name="learning_rate_omega")
    learning_rate_term = tf.placeholder(tf.float32, (), name="learning_rate_term")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    optimizer_omega = optimizer_spec_omega.constructor(learning_rate=learning_rate_omega, **optimizer_spec_omega.kwargs)
    optimizer_term = optimizer_spec_term.constructor(learning_rate=learning_rate_term, **optimizer_spec_term.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error_q, var_list=q_func_vars, clip_val=grad_norm_clipping)
    train_fn_omega = minimize_and_clip(optimizer_omega, total_error_q_omega, var_list=q_func_vars, clip_val=grad_norm_clipping)

    if debug_no_term_train:
        train_fn_term = tf.no_op()
    else:
        train_fn_term = minimize_and_clip(optimizer_term, total_error_term, var_list=q_func_vars, clip_val=grad_norm_clipping)
        # train_fn_term = term_grad(optimizer_term, term_val_next, var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    # for log purposes
    replay_buffer_test = ReplayBuffer(10000, frame_history_len)
    replay_buffer_test1 = ReplayBuffer(10000, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    q_model_initialized = False
    option_activated = False
    option_running = None
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    next_recent_obs = None
    idx = replay_buffer.store_frame(last_obs)
    LOG_EVERY_N_STEPS = 10000
    running_time = [0, time.time()]

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
        epsilon = exploration.value(t / learning_freq)
        if next_recent_obs is not None:
            recent_obs = next_recent_obs
        else:
            recent_obs = replay_buffer.encode_recent_observation()

        # pick an option
        if not option_activated:
            if random.random() > epsilon and model_initialized:
                q_omega_vals = session.run(q_omega_current, {obs_t_ph: recent_obs[None, ..., -input_history_len:]})
                option_running = np.argmax(q_omega_vals)
            else:
                option_running = random.choice(range(num_options))
        option_activated = True

        # choose action
        action = options[option_running].act(recent_obs)

        # take a step in the environment
        new_obs, r, done, _ = env.step(action)
        if done:
            new_obs = env.reset()

        # TODO: change replay_buffer to support option storage
        opa = np.array(
            [1 if i == option_running or options[i].act(recent_obs) == action else 0 for i in range(num_options)]
        )
        # opa[option_running] = 2
        replay_buffer.store_effect(idx, action, r, done, opa)
        idx = replay_buffer.store_frame(new_obs)
        next_recent_obs = replay_buffer.encode_recent_observation()

        if not model_initialized:
            initialize_interdependent_variables(session, tf.global_variables(), {
                obs_t_ph: recent_obs[None, ...,  -input_history_len:],
                obs_tp1_ph: next_recent_obs[None, ..., -input_history_len:],
            })
            session.run(update_target_fn)
            model_initialized = True
            saver = tf.train.Saver()
            # checkpoint = tf.train.get_checkpoint_state(
            #     "/home/lsy/PycharmProjects/ple-monstrerkong/examples/dqn_transfer_option/logs/12_16off1c_12_16_17_20:38:47/dqn")
            # if checkpoint and checkpoint.model_checkpoint_path:
            #     saver.restore(session, checkpoint.model_checkpoint_path)
            #     print("Successfully loaded: ", checkpoint.model_checkpoint_path)
            #     Load = True
            # else:
            #     print("Could not find old network weights")

        # online update q_omega & termination
        if t > learning_starts and not debug_no_term_train and not done:
            loss_term, _ = session.run([total_error_term, train_fn_term], feed_dict={
                obs_t_ph: recent_obs[None, ..., -input_history_len:],
                opt_t_ph: [option_running],
                # opa_t_ph: [option_running],
                # rew_t_ph: [r],
                obs_tp1_ph: next_recent_obs[None, ..., -input_history_len:],
                done_mask_ph: [1.0 if done == True else 0.0],
                learning_rate_omega: optimizer_spec_omega.lr_schedule.value(t / learning_freq),
                learning_rate_term: optimizer_spec_term.lr_schedule.value(t / learning_freq)
            })
            loss_term_log.append(loss_term)
        if online_q_omega:
            for i in range(num_options):
                if opa[i] > 0:
                    _, loss_omega = session.run([train_fn_omega, total_error_q_omega], feed_dict={
                        obs_t_ph: recent_obs[None, ..., -input_history_len:],
                        opa_t_ph: [i],
                        rew_t_ph: [r],
                        obs_tp1_ph: next_recent_obs[None, ..., -input_history_len:],
                        done_mask_ph: [1.0 if done == True else 0.0],
                        learning_rate_omega: optimizer_spec_omega.lr_schedule.value(t / learning_freq),
                    })
                    loss_omega_log.append(loss_omega)


        term_probs = session.run(term_next_current, {obs_tp1_ph: next_recent_obs[None, ...,  -input_history_len:]})
        # print "term_probs", term_probs
        if done or not random.random() > term_probs[0][option_running]:  # will re-pick an option
            option_activated = False


        #####

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).
            # 3.b: initialize the model if it has not been initialized yet; to do
            # that, call
            #    initialize_interdependent_variables(session, tf.global_variables(), {
            #        obs_t_ph: obs_t_batch,
            #        obs_tp1_ph: obs_tp1_batch,
            #    })
            # where obs_t_batch and obs_tp1_batch are the batches of observations at
            # the current and next time step. The boolean variable q_model_initialized
            # indicates whether or not the model has been initialized.
            # Remember that you have to update the target network too (see 3.d)!
            # 3.c: train the model. To do this, you'll need to use the train_fn and
            # total_error ops that were created earlier: total_error is what you
            # created to compute the total Bellman error in a batch, and train_fn
            # will actually perform a gradient step and update the network parameters
            # to reduce total_error. When calling session.run on these you'll need to
            # populate the following placeholders:
            # obs_t_ph
            # act_t_ph
            # rew_t_ph
            # obs_tp1_ph
            # done_mask_ph
            # (this is needed for computing total_error)
            # learning_rate -- you can get this from optimizer_spec.lr_schedule.value(t)
            # (this is needed by the optimizer to choose the learning rate)
            # 3.d: periodically update the target network by calling
            # session.run(update_target_fn)
            # you should update every target_update_freq steps, and you may find the
            # variable num_param_updates useful for this (it was initialized to 0)
            #####
            
            # YOUR CODE HERE

            # step a
            obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask_batch, opa_batch = replay_buffer.sample(batch_size)

            # step b
            if not q_model_initialized:
                session.run(update_target_fn)
                q_model_initialized = True

            # step c --also log cost(loss) TODO: off-policy update q_omega & termination
            # run_list = [train_fn, total_error_q]
            run_list = []
            feed_dict = {
                obs_t_ph: obs_t_batch[..., -input_history_len:],
                act_t_ph: act_t_batch,
                rew_t_ph: rew_t_batch,
                obs_tp1_ph: obs_tp1_batch[..., -input_history_len:],
                done_mask_ph: done_mask_batch,
                learning_rate: optimizer_spec.lr_schedule.value(t / learning_freq)
            }
            if not online_q_omega and not q_omega_uniform_sample:
                run_list.append(train_fn_omega)
                run_list.append(total_error_q_omega)
                feed_dict[opa_t_ph] = opa_batch

                feed_dict[learning_rate_omega] = optimizer_spec_omega.lr_schedule.value(t / learning_freq)
            # if q_omega_uniform_sample:
            #     _, cost = session.run(run_list, feed_dict=feed_dict)
            # else:
            _, loss_omega = session.run(run_list, feed_dict=feed_dict)
            # Costs.append(cost)
            if not online_q_omega and q_omega_uniform_sample:
                obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask_batch, opa_batch = replay_buffer.sample(batch_size, opa_uniform=True)
                opa_t_batch = opa_batch
                _, loss_omega = session.run([train_fn_omega, total_error_q_omega], feed_dict={
                    obs_t_ph: obs_t_batch[..., -input_history_len:],
                    opa_t_ph: opa_t_batch,
                    rew_t_ph: rew_t_batch,
                    obs_tp1_ph: obs_tp1_batch[..., -input_history_len:],
                    done_mask_ph: done_mask_batch,
                    learning_rate_omega: optimizer_spec_omega.lr_schedule.value(t / learning_freq)
                })

            loss_omega_log.append(loss_omega)
            # step d
            if t % target_update_freq == 0 and q_model_initialized:
                num_param_updates += 1
                session.run(update_target_fn)

            #####

        ### 4. Log progress
        if (t > 0) and (t % epoch_size == 0) and (t > learning_starts):
            average_loss_omega.append(np.mean(np.array(loss_omega_log)[int(-epoch_size):]))
            average_loss_term.append(np.mean(np.array(loss_term_log)[int(-epoch_size):]))
            plt.figure(8)
            plt.plot(average_loss_omega)
            plt.grid()
            plt.savefig(log_dir + 'average_loss_omega.png')
            plt.figure(9)
            plt.plot(average_loss_term)
            plt.grid()
            plt.savefig(log_dir + 'average_loss_term.png')
        if t % 1e3 == 0 and t > 0:
            if eval_obs_array:
                if obs_random_batch is None:
                    # print len(eval_obs_array)
                    if len(eval_obs_array) == 1:
                        if len(eval_obs_array[0]) < 6:
                            obs_random_batch = np.array(eval_obs_array)[0, 0,... , -input_history_len:]
                        else:
                            obs_random_batch = np.array(eval_obs_array)[0, 0:100, 0,..., -input_history_len:]
                    else:
                            obs_random_batch = eval_obs_array[0][0]
                            diff_batch = eval_obs_array[1][0]
                summary_str = session.run(merged_summary_op, feed_dict={obs_t_ph: obs_random_batch})
                summary_writer.add_summary(summary_str, t)
                # summary_str_diff = session.run(merged_summary_diff, feed_dict={obs_t_ph: diff_batch})
                # summary_writer.add_summary(summary_str_diff, t)
            # else:
            #     print "no eval_obs_array"
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0 and q_model_initialized:
            last_t, last_time = running_time
            new_t , new_time  = t / LOG_EVERY_N_STEPS, time.time()
            running_time = [new_t, new_time]
            print "###########################################"
            print("Timestep %d" % (t,))
            print("Training time per %d timesteps %.2fs" %
                  (LOG_EVERY_N_STEPS, (new_time - last_time) / (new_t - last_t)))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t / learning_freq))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t / learning_freq))
            # evaluate q functions for different rooms (log_dir should exist!)
        if  t % epoch_size == 0 and log_dir != None and t > learning_starts:
            saver.save(session, log_dir + 'dqn/', global_step=t)
            if eval_obs_array:
                Q_current, U_max, Omega_max = session.run([average_Q, average_U, average_omega], feed_dict={obs_t_ph: obs_random_batch})
                print("evaluated q values for eval_obs_array")
                test_q_max_log.append(Q_current)
                test_u_log.append(U_max)
                test_omega_log.append(Omega_max)
                plt.figure(10)
                plt.plot(test_u_log)
                plt.grid()
                plt.savefig(log_dir + 'test_u_max.png')
                plt.figure(11)
                plt.plot(test_omega_log)
                plt.grid()
                plt.savefig(log_dir + 'test_omega_max.png')
            else:
                print("no eval_obs_array! q values are not evaluated! check your config")
        sys.stdout.flush()

    # tests & log cost
        # add a step: test 10000 step after 50000 step
        if t % epoch_size == 0 and log_dir != None and q_model_initialized:

            # test Q_omega and beta
            option_activated_test = False
            option_running_test = None
            last_obs_test1 = env_test1.reset()
            replay_buffer_test1.store_frame(last_obs_test1)
            h = 0
            episode_num1 = 1
            test_step1 = 0
            done_test1 = False
            next_recent_obs_test = None
            while test_step1 < 5000 or done_test1 == False:
                if next_recent_obs_test is not None:
                    recent_obs_test = next_recent_obs_test
                else:
                    recent_obs_test = replay_buffer_test1.encode_recent_observation()
                # pick an option
                if not option_activated_test:
                    if random.random() > 0.05:
                        q_omega_test = session.run(q_omega_current,
                                                   {obs_t_ph: recent_obs_test[None, ..., -input_history_len:]})
                        option_running_test = np.argmax(q_omega_test)
                    else:
                        option_running_test = random.choice(range(num_options))
                option_activated_test = True

                # choose action
                action_test1 = options[option_running_test].act(recent_obs_test)

                new_obs_test, r_test1, done_test1, _ = env_test1.step(action_test1)
                h += 1
                test_step1 += 1
                if done_test1:
                    # option_activated_test = False
                    reward = r_test1 * (gamma ** h)
                    h = 0
                    # summary_str1 = session.run(merged_summary_op1, feed_dict={Episode_reward: reward})
                    # summary_writer.add_summary(summary_str1, global_step=Episode_num)
                    Episode_num += 1
                    if test_step1 < 5000:
                        new_obs_test = env_test1.reset()
                        episode_num1 += 1

                replay_buffer_test1.store_frame(new_obs_test)
                next_recent_obs_test = replay_buffer_test1.encode_recent_observation()
                term_probs_test = session.run(term_next_current,
                                         {obs_tp1_ph: next_recent_obs_test[None, ..., -input_history_len:]})
                if done_test1 or not random.random() > term_probs_test[0][option_running_test]:  # will re-pick an option
                    option_activated_test = False

            episode_rewards_test1 = get_wrapper_by_name(env_test1, "Monitor").get_episode_rewards()
            episode_lengths_test1 = get_wrapper_by_name(env_test1, "Monitor").get_episode_lengths()
            evaluation_metric1 = np.array(episode_rewards_test1) * (gamma ** np.array(episode_lengths_test1))
            test_reward1 = np.mean(evaluation_metric1)
            none_discount_average1 = np.mean(episode_rewards_test1)
            none_discount1.append(none_discount_average1)
            short_average1.append(np.mean(evaluation_metric1[-episode_num1:]))
            short_average_none1.append(np.mean(episode_rewards_test1[-episode_num1:]))
            test_rewards1.append(test_reward1)
            plt.figure(12)
            plt.plot(test_rewards1)
            plt.grid()
            plt.savefig(log_dir + 'test_rewards1.png')
            plt.figure(13)
            plt.plot(none_discount1)
            plt.grid()
            plt.savefig(log_dir + 'none_discount1.png')
            plt.figure(14)
            plt.plot(short_average1)
            plt.grid()
            plt.savefig(log_dir + 'short_average1.png')
            plt.figure(15)
            plt.plot(short_average_none1)
            plt.grid()
            plt.savefig(log_dir + 'short_average_none1.png')
            plt.figure(1)
            plt.plot(evaluation_metric1)
            plt.grid()
            plt.savefig(log_dir + 'episode_reward.png')

            with open(log_dir + '/pkl/test_rewards1.pkl', 'wb') as output:
                pickle.dump(test_rewards1, output)
            with open(log_dir + '/pkl/none_discount1.pkl', 'wb') as output:
                pickle.dump(none_discount1, output)
            with open(log_dir + '/pkl/short_average1.pkl', 'wb') as output:
                pickle.dump(short_average1, output)
            with open(log_dir + '/pkl/short_average_none1.pkl', 'wb') as output:
                pickle.dump(short_average_none1, output)
            with open(log_dir + '/pkl/episode_reward.pkl', 'wb') as output:
                pickle.dump(evaluation_metric1, output)
            with open(log_dir + '/pkl/test_omega_log.pkl', 'wb') as output:
                pickle.dump(test_omega_log, output)
