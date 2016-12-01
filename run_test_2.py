#! /usr/bin/env python
import sys
import csv
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import gzip
import pickle
import random
from stat_collector import StatisticsCollector
from tensorflow.contrib import learn
import pdb
# Parameters
# ==================================================
DATA_LOC = sys.argv[1]

"""Line 133ish: fix restore variables"""
"""122ish: fix path to model"""


# Model Hyperparameters
#tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("filters_per_layer", '128,256,512,1024', "Number of filters per layer (default: 8,8,12,16)")
tf.flags.DEFINE_string("cnn", "mod", "which cnn to use (default: 'reg')")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("dropout_factor", 1.0, "Probability of weights to keep for dropout (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", .0001, "Gradient descent learning rate (default: .0005)")
#tf.flags.DEFINE_float("fc_layers", 1, "number of fully connected layers at output (1 or 2) (default: 1)")
#tf.flags.DEFINE_string("activation_func", 'relu', "activation function (can be: tanh or relu) (default: relu)")
tf.flags.DEFINE_float("l2_constraint", None, "Constraint on l2 norms of weight vectors (default: None)")
tf.flags.DEFINE_float("dev_size", 0.1, "size of the dev batch in percent vs entire train set (default: 0.20)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 4, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 400, "Evaluate model on dev set after this many steps (default: 400)")
tf.flags.DEFINE_integer("checkpoint_every", 800, "Save model after this many steps (default: 200)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
flags_list = ["{}={}".format(attr.upper(), value) for attr,value in sorted(FLAGS.__flags.items())]
save_flags = ["{}={}".format(attr.upper(), value) for attr,value in sorted(FLAGS.__flags.items())]
for i in flags_list:
    print(i)
print("")

# choose which cnn to use
# ==================================================
cnns = {'mod':'eeg_cnn_mod', 'triple':'audio_cnn_triple'}
AudioCNN = getattr(__import__(cnns[FLAGS.cnn], fromlist=['EegCNN']), 'EegCNN')

# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
data_loc = DATA_LOC
spect_dict = data_helpers.read_from_pickles(data_loc)
# zero-mean spect-dict
# print("Zero-meaning data...")
# spect_dict_mean = np.mean(list(spect_dict.values()),0)
# spect_dict = {k: v-spect_dict_mean for k,v in spect_dict.items()}
# print("Normalizing data...")
# spect_dict_std = np.std(list(spect_dict.values()),0)
# spect_dict = {k: v/spect_dict_std for k,v in spect_dict.items()}
# get cliques from dataset textfile
# cliques = data_helpers.txt_to_cliques(train_loc)
# prune cliques to make sure we're not referencing songs that weren't downloaded
# pruned_cliques = data_helpers.prune_cliques(cliques,spect_dict)

test_cliques = data_helpers.cliques_to_test(spect_dict)
x_test = [i for i in test_cliques.keys()]

print("Dataset Size: {:d}".format(len(x_test)))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement = FLAGS.allow_soft_placement,
      log_device_placement = FLAGS.log_device_placement,
      )

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = AudioCNN(
            spect_dim=random.choice(list(spect_dict.values())).shape,
            num_classes=2,
            filters_per_layer = [int(i) for i in FLAGS.filters_per_layer.split(',')],
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        if FLAGS.l2_constraint: #add l2 constraint as described in (Kim, Y. 2014)
             grads_and_vars = [(tf.clip_by_norm(grad, FLAGS.l2_constraint), var) for grad, var in grads_and_vars]
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        print(save_flags)
        important_flags = [i for i in save_flags if 'FILTERS_PER_LAYER' in i or 'DROPOUT' in i or 'LEARNING_RATE' in i or 'L2_REG_LAMBDA' in i or 'BATCH_SIZE' in i]
        out_dir = os.path.abspath(os.path.join(os.path.curdir)) #, "runs") #, ','.join(important_flags)))
        print("Writing to {}\n".format(out_dir))

        # Train/Dev Summary Dirs
        test_summary_dir = os.path.join(out_dir, "summaries", "dev")
        # test_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, checkpoint_prefix+"/model-8800")
        #tf.initialize_all_variables()

        def test_step(x_test, writer=None):
            '''
            Predicts on full test set.
            --------------------------------
            Since full test set likely won't fit into memory, this function
            splits the test set into minibatches and writes them to csv as we go.
            '''
            dev_stats = StatisticsCollector()
            dev_batches = data_helpers.batch_iter(list(zip(x_test, [(0,0)for i in x_test])),
                                      FLAGS.batch_size, 1, shuffle=False)
            """
                                                                Using a dummy value for y_test
                                                                because it's not touched when predicting"""
            for dev_batch in dev_batches:
                if len(dev_batch) > 0:
                    x_dev_batch, y_dev_batch = zip(*dev_batch)

                    print("Feeding data")
                    feed_dict = {
                      cnn.input_eeg: tuple(spect_dict[i] for i in x_dev_batch),
                      cnn.input_y: y_dev_batch,
                      cnn.dropout_keep_prob: 1.0
                    }
                    step, predictions = sess.run(
                        [global_step, cnn.predictions],
                        feed_dict)
                    dev_stats.collect(accuracy, loss)

                    filenames = [name for name in x_dev_batch]
                    with open('predictions.csv', 'ab') as csvfile:
                        print("writing")
                        testwriter = csv.writer(csvfile, delimiter=',')
                        testwriter.writerows([(filenames[i], predictions[i]) for i in range(len(filenames))])

            time_str = datetime.datetime.now().isoformat()
            print("Predicted {} files. Time: {}".format(len(dev_batches), time_str))
        # send entire test set to dev_step each eval and split into minibatches there
        test_step(x_test)
