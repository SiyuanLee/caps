# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from dqn_utils import NoOpWrapperMK

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

def get_session():
    """
    Use a new graph for each source policy
    """
    source_graph = tf.Graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(graph=source_graph, config=tf_config)
    return session

def wrapper_offset(dqn_config):
    offset = 0
    if dqn_config is not None:
        if dqn_config.has_key('additional_wrapper'): # assume noop
            offset += 1
        if dqn_config.has_key('transfer_config'):
            transfer_config = dqn_config['transfer_config']
            if transfer_config is not None:
                if transfer_config.has_key('source_noop') and transfer_config['source_noop']:
                    offset -= 1
    return offset

class Option(object):
    def act(self, obs):
        pass

class PrimitiveOption(Option):
    def __init__(self, action):
        self.action = action

    def act(self, obs):
        return self.action

class Source(Option):
    """
    load saved source policies
    act method can be called to act
    """

    def __init__(self, dqn_config, env, checkpoint):
        self.sess = get_session()

        img_h, img_w, img_c = env.observation_space.shape
        num_actions = env.action_space.n
        num_actions += wrapper_offset(dqn_config)
        frame_history_len = dqn_config['frame_history_len']  # 可能需要修改
        input_shape = (img_h, img_w, frame_history_len * img_c)
        with self.sess.graph.as_default():
            self.obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
            obs_t_float = tf.cast(self.obs_t_ph, tf.float32) / 255.0
            self.q_current = atari_model(obs_t_float, num_actions, scope='q_func', reuse=False)
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
            self.saver = tf.train.Saver(var_list=q_func_vars)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)


    def act(self, obs):
        q_vals = self.sess.run(self.q_current, {self.obs_t_ph: obs[None, :]})
        action = np.argmax(q_vals)
        return action