import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import math
import numpy as np

MOVING_AVERAGE_DECAY = 0.9999

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
      These placeholders are used as inputs by the rest of the model building
      code and will be fed from the downloaded data in the .run() loop, below.
      Args:
        batch_size: The batch size will be baked into both placeholders.
      Returns:
      
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           c3d_model.NUM_FRAMES_PER_CLIP,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    return images_placeholder, labels_placeholder

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads,0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def tower_loss(name_scope, logit, labels):
    cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels)
                  )
    tf.summary.scalar(name_scope + 'cross_entropy',cross_entropy_mean)
    #tf.scalar_summary(name_scope + 'cross entropy',cross_entropy_mean)
    weight_decay_loss = tf.add_n(tf.get_collection('losses', name_scope))
    #tf.scalar_summary(name_scope + 'weight decay loss', weight_decay_loss)
    tf.summary.scalar(name_scope + 'weight_decay_loss', weight_decay_loss)
    tf.add_to_collection('losses', cross_entropy_mean)
    losses = tf.get_collection('losses', name_scope)

  # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    tf.summary.scalar(name_scope + 'total_loss', total_loss)

  # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.99)###################################################
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)

        return total_loss

def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def get_op(images_placeholder, labels_placeholder,FLAGS,gpu_num):
    for gpu_index in range(0, gpu_num):
        global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )
#         images_placeholder, labels_placeholder = placeholder_inputs(
#                     FLAGS.batch_size * gpu_num
#                     )
        tower_grads1 = []
        tower_grads2 = []
        logits = []
        opt1 = tf.train.AdamOptimizer(1e-4) #.minimize(d_loss, var_list=d_vars)
        opt2 = tf.train.AdamOptimizer(2e-4)
        with tf.device('/gpu:%d' % gpu_index):
            with tf.name_scope('%s_%d' % ('dextro-research', gpu_index)) as scope:
                with tf.variable_scope('var_name') as var_scope:
                    weights = {
                        'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
                        'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                        'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                        'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
                        'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                        'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
                        'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                        'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
                        'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
                        'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                        'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005)
                        }
                    biases = {
                      'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
                      'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
                      'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
                      'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
                      'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
                      'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
                      'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
                      'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
                      'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
                      'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
                      'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
                      }
                varlist1 = weights.values()
                varlist2 = biases.values()
                logit = c3d_model.inference_c3d(
                          images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:],
                          0.5,
                          FLAGS.batch_size,
                          weights,
                          biases
                          )
                loss = tower_loss(
                          scope,
                          logit,
                          labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                          )
                grads1 = opt1.compute_gradients(loss, varlist1)
                grads2 = opt2.compute_gradients(loss, varlist2)
                tower_grads1.append(grads1)
                tower_grads2.append(grads2)
                logits.append(logit)

    logits = tf.concat(logits,0)
    accuracy = tower_acc(logits, labels_placeholder)
    tf.summary.scalar('accuracy', accuracy)
    grads1 = average_gradients(tower_grads=tower_grads1)
    grads2 = average_gradients(tower_grads=tower_grads2)           
    apply_gradient_op1 = opt1.apply_gradients(grads_and_vars=grads1)
    apply_gradient_op2 = opt2.apply_gradients(grads_and_vars=grads2, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)
    null_op = tf.no_op()
    return train_op,accuracy,weights,biases

def get_logits(images_placeholder, labels_placeholder,FLAGS,gpu_num):
    with tf.variable_scope('var_name') as var_scope:
        weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
            }
        biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
            }
    logits = []
    for gpu_index in range(0, gpu_num):
        with tf.device('/gpu:%d' % gpu_index):
            logit = c3d_model.inference_c3d(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:], 1, FLAGS.batch_size, weights, biases)
            logits.append(logit)
    logits = tf.concat(logits,0)
    return logits