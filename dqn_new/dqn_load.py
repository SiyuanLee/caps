import sys
import gym.spaces
import itertools
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import readchar
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
import time
import pickle

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


def learn(env,
          env_test,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10,
          eval_obs_array=None,
          room_q_interval=1e5,
          epoch_size=5e4,
          log_dir=None,
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
    session: tf.Session
        tensorflow session to use.
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
        If not None gradients' norms are clipped to this value.
    """

    average_Qs = []
    Costs = []
    Average_costs = []
    test_rewards = []

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph = tf.placeholder(tf.int32, [None])
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
    q_current = q_func(obs_t_float, num_actions, scope='q_func', reuse=False)
    q_target = q_func(obs_tp1_float, num_actions, scope='target_q_func', reuse=False)
    q_next_current = q_func(obs_tp1_float, num_actions, scope='q_func', reuse=True)
    # q_target_current = q_func(obs_t_float, num_actions, scope='target_q_func', reuse=True)

    q_val_current = tf.reduce_sum(q_current * tf.one_hot(act_t_ph, num_actions), axis=-1)
    q_val_next_raw = tf.reduce_sum(q_target * tf.one_hot(tf.argmax(q_next_current, axis=-1), num_actions), axis=-1)
    q_val_next = q_val_next_raw * (1 - done_mask_ph)

    total_error = tf.reduce_sum(tf.losses.mean_squared_error(rew_t_ph + gamma * q_val_next, q_val_current))

    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    # # for log purposes
    # q_max = tf.reduce_max(q_current, 1)
    # average_Q = tf.reduce_mean(q_max)
    ######

    # construct optimization op (with gradient clipping)
    # learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    # optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    # train_fn = minimize_and_clip(optimizer, total_error,
    #                              var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    # # for log purposes
    # random_buffer = RandomBuffer(random_length, frame_history_len)
    # replay_buffer_test = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
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
        # if t < random_length:
        #     random_buffer.store_frame(last_obs)
        # if t == random_length:
        #     obs_random, _, _, _, _ = random_buffer.sample(random_length - 1)

        idx = replay_buffer.store_frame(last_obs)
        # epsilon = exploration.value(t)
        if random.random() > 0.0 and model_initialized:
            # action = session.run(tf.argmax(q_current, axis=-1),
            #                 feed_dict={obs_t_ph: replay_buffer.encode_recent_observation()[None, :]})[0]
            input_batch = replay_buffer.encode_recent_observation()
            q_vals = session.run(q_current, {obs_t_ph: input_batch[None, :]})
            action = np.argmax(q_vals)
            # evaluation after training
            print "q_vals", q_vals
            print "max_q", np.max(q_vals)
            print "greedy_polciy", np.argmax(q_vals)
            print "action", action
            # key = readchar.readkey()

            # print key
            # if key == 's':
            #     action = 1
            # elif key == 'a':
            #     action = 4
            # elif key == 'd':
            #     action = 2
            # elif key == 'w':
            #     action = 3
            # else:
            #     key = int(key)
            #     if key == 0:
            #         action = 0
            #     else:
            #         action = 5
        else:
            # action = random.choice(range(num_actions))
            action = 5
            print "noop action:", action
        last_obs, r, done, _ = env.step(action)
        replay_buffer.store_effect(idx, action, r, done)

        if done:
            last_obs = env.reset()

        #####

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (replay_buffer.can_sample(batch_size)):
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
            # the current and next time step. The boolean variable model_initialized
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
            obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask_batch = replay_buffer.sample(batch_size)

            # step b
            if not model_initialized:
                initialize_interdependent_variables(session, tf.global_variables(), {
                    obs_t_ph: obs_t_batch,
                    obs_tp1_ph: obs_tp1_batch,
                })
                session.run(update_target_fn)
                model_initialized = True

                # create saver or load saved networks (different runs?
                saver = tf.train.Saver()
                checkpoint = tf.train.get_checkpoint_state("/home/lsy/source/same/ez1_20same_s6_01_20_18_11:28:17/dqn")
                if checkpoint and checkpoint.model_checkpoint_path:
                    saver.restore(session, checkpoint.model_checkpoint_path)
                    print("Successfully loaded: ", checkpoint.model_checkpoint_path)
                else:
                    print("Could not find old network weights")

            # step c --also log cost(loss)
            # _, cost = session.run([train_fn, total_error], feed_dict={
            #     obs_t_ph: obs_t_batch,
            #     act_t_ph: act_t_batch,
            #     rew_t_ph: rew_t_batch,
            #     obs_tp1_ph: obs_tp1_batch,
            #     done_mask_ph: done_mask_batch,
            #     learning_rate: optimizer_spec.lr_schedule.value(t)
            # })
            # Costs.append(cost)

            # step d
            # if t % target_update_freq == 0 and model_initialized:
            #     num_param_updates += 1
            #     session.run(update_target_fn)

                #####

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
            # saver.save(session, 'saved_networks/dqn', global_step=t)
            last_t, last_time = running_time
            new_t, new_time = t / LOG_EVERY_N_STEPS, time.time()
            running_time = [new_t, new_time]
            print "###########################################"
            print("Timestep %d" % (t,))
            print("Training time per %d timesteps %.2fs" %
                  (LOG_EVERY_N_STEPS, (new_time - last_time) / (new_t - last_t)))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            # evaluate q functions for different rooms (log_dir should exist!)
            # if t % room_q_interval == 0 and log_dir != None:
            #     saver.save(session, log_dir + 'dqn', global_step=t)
            #     if eval_obs_array:
            #         q_vals_eval_obs_array = []
            #         for eval_obs in eval_obs_array:
            #             q_vals_eval_obs = []
            #             for ob in eval_obs:
            #                 q_vals_eval_obs.append(session.run(q_current, {obs_t_ph: ob})[0])
            #             q_vals_eval_obs_array.append(q_vals_eval_obs)
            #         with open(log_dir + str(t) + "_q.pkl", 'wb') as f:
            #             pickle.dump(q_vals_eval_obs_array, f)
            #         print("evaluated q values for eval_obs_array")
            #     else:
            #         print("no eval_obs_array! q values are not evaluated! check your config")
            sys.stdout.flush()

            # tests & log cost
            # add a step: test 10000 step after 50000 step
            # if (t - learning_starts) % epoch_size == 0 and log_dir != None:
            #     last_obs_test = env_test.reset()
            #     test_step = 0
            #     done_test = False
            #     while test_step < 10000 or done_test == False:
            #         # for test_step in range(10000):
            #         replay_buffer_test.store_frame(last_obs_test)
            #         if random.random() > 0.05:
            #             input_batch_test = replay_buffer_test.encode_recent_observation()
            #             q_vals_test = session.run(q_current, {obs_t_ph: input_batch_test[None, :]})
            #             action_test = np.argmax(q_vals_test)
            #         else:
            #             action_test = random.choice(range(num_actions))
            #         last_obs_test, r_test, done_test, _ = env_test.step(action_test)
            #         if done_test:
            #             if test_step < 10000:
            #                 last_obs_test = env_test.reset()
            #         test_step += 1
            #     episode_rewards_test = get_wrapper_by_name(env_test, "Monitor").get_episode_rewards()
            #     episode_lengths_test = get_wrapper_by_name(env_test, "Monitor").get_episode_lengths()
            #     evaluation_metric = np.array(episode_rewards_test) * (gamma ** np.array(episode_lengths_test))
            #     test_reward = np.mean(evaluation_metric)
            #     test_rewards.append(test_reward)
            #     plt.figure(3)
            #     plt.plot(test_rewards)
            #     plt.savefig(log_dir + 'test_rewards.png')

                # add loss to tensorboard
                # obs_random, _, _, _, _ = random_buffer.sample(random_length - 1)
                # average_q = session.run(average_Q, feed_dict={
                #     obs_t_ph: obs_random
                # })
                # average_Qs.append(average_q)
                # plt.figure(2)
                # plt.plot(average_Qs, color="blue")
                # plt.savefig(log_dir + 'average_q.png')
                # # plt.savefig('evaluate/average_q.png')
                #
                # # if t % epoch_size == 0 and log_dir != None:
                # plt.figure(1)
                # Average_costs.append(np.mean(Costs))
                # plt.plot(Average_costs, color="red")
                # # plt.plot(cost_smooth, color="red")
                # plt.savefig(log_dir + 'cost.png')