import tensorflow as tf
import tensorflow.contrib.layers.python.layers as tf_new
import numpy as np
import pdb

class EegCNN(object):
    """
    A CNN for classifying EEG's as either preictal or non preictal. 
    Pipelines based on FCN-4 from Choi et. al. 2016 https://arxiv.org/abs/1606.00298 
    Uses convolutional, max-pooling and batch-norm layers followed by a binary softmax layer.
    """
    def __init__(
      self, spect_dim, num_classes,
      filters_per_layer, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_eeg = tf.placeholder(tf.float32, [None, *spect_dim], name="input_eeg")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # create wrappers for basic convnet functions
        def create_variable(self, name, shape, initializer = tf.random_normal_initializer()):
            '''
            Function wrapper for creating a layer with random initializer
            '''
            return tf.get_variable(name, shape, initializer=initializer)

        def conv(self, x, kx, ky, in_depth, num_filters, sx=1, sy=1, name=None, reuse=None, batch_norm=True):
            '''
            Function that defines a convolutional layer
            -------------------------------------------
            x           : input tensor
            kx,ky       : filter (kernel) dimensions
            sx,sy       : stride dimensions
            in_depth    : depth of input tensor
            num_filters : number of conv filters
            '''
            with tf.variable_scope(name, reuse=reuse) as scope:
                kernel = create_variable(self, "weights", [kx, ky, in_depth, num_filters], tf.contrib.layers.xavier_initializer_conv2d())
                bias = create_variable(self, "bias", [num_filters])
                conv = tf.nn.relu(tf.nn.bias_add(
                       tf.nn.conv2d(x, kernel, strides=[1, sx, sy, 1], padding='SAME'), bias),
                                    name=scope.name)
                if batch_norm:
                    # batch normalization
                    conv = tf_new.batch_norm(conv,scale=False)
            return conv

        def pool(self, x, kx, ky, sx=None, sy=None, name=None):
            '''
            Function that defines a pooling layer
            If no specified stride: stride = kernel size
            -------------------------------------------
            x           : input tensor
            kx,ky       : kernel dimensions
            sx,sy       : stride dimensions
            '''            
            if not sx or sy: sx,sy = kx,ky
            pool = tf.nn.max_pool(x, ksize=[1, kx, ky, 1], strides=[1, sx, sy, 1], padding='SAME')
            return pool

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.name_scope("eeg"), tf.device('/gpu:0'):
            # convolutional architecture for eeg
            conv1a = conv(self, x=self.input_eeg, kx=3, ky=3, in_depth=16, num_filters=filters_per_layer[0]/4, name='conv1a')
            conv1a = conv(self, x=conv1a, kx=3, ky=3, in_depth=filters_per_layer[0]/4, num_filters=filters_per_layer[0]/2, name='conv1b')
            conv1a = conv(self, x=conv1a, kx=3, ky=3, in_depth=filters_per_layer[0]/2, num_filters=filters_per_layer[0], name='conv1c')
            conv1a = pool(self, conv1a, kx=2, ky=4, name='pool1a')
            # conv2a
            conv2a = conv(self, x=conv1a, kx=1, ky=1, in_depth=filters_per_layer[0], num_filters=filters_per_layer[0]/2, name='conv2a')
            conv2a = conv(self, x=conv2a, kx=3, ky=1, in_depth=filters_per_layer[0]/2, num_filters=filters_per_layer[1]/2, name='conv2b')
            conv2a = conv(self, x=conv2a, kx=1, ky=3, in_depth=filters_per_layer[1]/2, num_filters=filters_per_layer[1]/2, name='conv2c')
            conv2a = pool(self, conv2a, kx=3, ky=5, name='pool2a')
            # conv3a
            conv3a = conv(self, x=conv2a, kx=1, ky=1, in_depth=filters_per_layer[1]/2, num_filters=filters_per_layer[1]/2, name='conv3a')
            conv3a = conv(self, x=conv3a, kx=3, ky=1, in_depth=filters_per_layer[1]/2, num_filters=filters_per_layer[2]/2, name='conv3b')
            conv3a = conv(self, x=conv3a, kx=1, ky=3, in_depth=filters_per_layer[2]/2, num_filters=filters_per_layer[2]/2, name='conv3c')
            conv3a = pool(self, conv3a, kx=3, ky=8, name='pool3a')
            # conv4a
            conv4a = conv(self, x=conv3a, kx=1, ky=1, in_depth=filters_per_layer[2]/2, num_filters=filters_per_layer[2]/2, name='conv4a')
            conv4a = conv(self, x=conv4a, kx=3, ky=1, in_depth=filters_per_layer[2]/2, num_filters=filters_per_layer[3]/2, name='conv4b')
            conv4a = conv(self, x=conv4a, kx=1, ky=3, in_depth=filters_per_layer[3]/2, num_filters=filters_per_layer[3]/2, name='conv4c')
            conv4a = pool(self, conv4a, kx=5, ky=8, name='pool4a')
            # conv5a
            conv5a = conv(self, x=conv4a, kx=1, ky=1, in_depth=filters_per_layer[3]/2, num_filters=filters_per_layer[3], name='conv5a')
            conv5a = pool(self, conv5a, kx=2, ky=2, name='pool5a')
            conv5a = conv(self, x=conv5a, kx=1, ky=1, in_depth=filters_per_layer[3], num_filters=filters_per_layer[3], name='conv5b')
            conv5a = pool(self, conv5a, kx=2, ky=2, name='pool5b')
            conv5a = conv(self, x=conv5a, kx=1, ky=1, in_depth=filters_per_layer[3], num_filters=filters_per_layer[3], name='conv5c')
            conv5a = pool(self, conv5a, kx=3, ky=4, name='pool5c')

            self.song1_out = tf.reshape(conv5a, [-1, filters_per_layer[3]])

        # Add dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.drop = tf.nn.dropout(self.song1_out, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"), tf.device('/gpu:2'):
            W = tf.get_variable(
                "W",
                shape=[filters_per_layer[3], num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), "b")
            self.scores = tf.nn.xw_plus_b(self.drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate L2 loss
        vars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars])

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"), tf.device('/gpu:3'):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"), tf.device('/gpu:3'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
              
