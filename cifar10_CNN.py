#Author: Qilin You USC ID: 3647897461
#Date: Apr 3 2017
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import time
import math
import tensorflow as tf
import numpy as np


from six.moves import urllib
from datetime import datetime
from scipy.cluster.vq import kmeans2,whiten,kmeans
from numpy import linalg as LA
from array import array
#Cite subfile
import cifar10_input

FLAGS = tf.app.flags.FLAGS
# Define model parameters.
tf.app.flags.DEFINE_integer('batch_size', 256,
                            """Number of images per batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/qilin/cifar10_data',
                           """Path to data directory.""")

tf.app.flags.DEFINE_string('train_dir', '/tmp/qilin/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 200000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('K_size', 512,
                            """How many sample used for K-means intialization.""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/qilin/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('trainflag', 0,
                            """Indicate train or test inference.""")
tf.app.flags.DEFINE_integer('testflag', 1,
                            """Indicate train or test inference.""")
# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999      # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

# URL to download cifar-10 dataset
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

#get variable 
def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float32
  var = _variable_on_cpu(name,shape, #tf.contrib.layers.xavier_initializer(dtype=dtype))
    tf.random_normal_initializer(stddev=stddev,dtype=dtype))
    #tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def batch_norm(x, n_out):
    with tf.variable_scope('bn'):
      beta = _variable_on_cpu('beta', [n_out], tf.constant_initializer(0.0))
      gamma = _variable_on_cpu('gamma', [n_out], tf.constant_initializer(1.0))

      batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
      normed = tf.nn.batch_normalization(x,batch_mean , batch_var, beta, gamma, 1e-3)
    return normed


def distorted_inputs():
  
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  return images, labels


def inputs(eval_data, samplesize):
  
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=samplesize)
  return images, labels


def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss'), tf.add_n([cross_entropy_mean])

def _add_loss_summaries(total_loss):
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  for l in losses + [total_loss]:
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def accuracy(logits, labels, name):
  fname=name+'_accuracy'
  correct_pred=tf.nn.in_top_k(logits, labels, 1)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  tf.summary.scalar(fname, accuracy)
  return accuracy

def train(total_loss, global_step):
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)

  tf.summary.scalar('learning_rate', lr)
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    # Compute gradients.
    opt = tf.train.GradientDescentOptimizer(lr)
    #opt = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9)
    #opt = tf.train.AdamOptimizer(lr)
    #opt = tf.train.AdagradOptimizer(lr)
    grads = opt.compute_gradients(loss=total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

   # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def kmeansIN(images,weights,biases):
 
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel=weights[scope.name]
    biase=biases[scope.name]
    conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='VALID')
    pre_activation = tf.nn.bias_add(conv, biase)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel=weights[scope.name]
    biase=biases[scope.name]
    conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='VALID')
    pre_activation = tf.nn.bias_add(conv, biase)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
  


  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [images.get_shape()[0].value, -1])
    weight=weights[scope.name]
    biase=biases[scope.name]
    local3 = tf.nn.relu(tf.matmul(reshape, weight) + biase, name=scope.name)

  # local4
  with tf.variable_scope('local4') as scope:
    weight=weights[scope.name]
    biase=biases[scope.name]
    local4 = tf.nn.relu(tf.matmul(local3, weight) + biase, name=scope.name)
    
  # linear layer(WX + b),
  with tf.variable_scope('softmax_linear') as scope:
    weight=weights[scope.name]
    biase=biases[scope.name]
    softmax_linear = tf.add(tf.matmul(local4, weight), biase, name=scope.name)
    

  return softmax_linear

def LeNet_5_model(images,mflag):
  print('LeNet-5 Model initializing...')
  # LeNet-5
  #Steps:
  # 1. Convolution 5x5 with padding=0: 3x32x32->6x28x28
  # 2. Max-pooling stride 2: 6x28x28->6x14x14
  # 3. Convolution 5x5 without padding: 6x14x14->16x10x10
  # 4. Max-pooling stride 2: 16x10x10->16x5x5
  # 5. Fully connected: 120 cells
  # 6. Fully connected: 84 cells
  # 7. Soft: 10 output
  #imagedrop=[0.2,0]
  #layerdrop=[0.5,0]

  #images_d=tf.layers.dropout(images,imagedrop[mflag])
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[5, 5, 3, 6],
                                         stddev=5e-2,wd=None)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [6], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  #pool1_d=tf.layers.dropout(pool1,layerdrop[mflag])

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[5, 5, 6, 16],
                                         stddev=5e-2,wd=None)
    conv = tf.nn.conv2d(pool1 , kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')
  #pool2_d=tf.layers.dropout(pool2,layerdrop[mflag])

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [images.get_shape()[0].value, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 120],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [120], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[120, 84],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [84], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

  # linear layer(WX + b),
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [84, NUM_CLASSES],
                                          stddev=1/84.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

  return softmax_linear

def Improved_model(images,mflag):
  print('Improved Model initializing...')
  # Improved
  #Steps:
  # 1. Convolution 3x3 with padding=1, stride=1: 3x32x32->96x32x32
  # 2. Convolution 3x3 with padding=1, stride=1: 96x32x32->96x32x32
  # 3. Convolution 3x3 with padding=1, stride=2: 96x32x32->96x16x16
  # 4. Convolution 3x3 with padding=1, stride=1: 96x16x16->192x16x16
  # 5. Convolution 3x3 with padding=1, stride=1: 192x16x16->192x16x16
  # 6. Convolution 3x3 with padding=1, stride=2: 192x16x16->192x8x8
  # 7. Convolution 3x3 with padding=0, stride=1: 192x8x8->192x6x6
  # 8. Convolution 1x1 with padding=0, stride=1: 
  # 9. Convolution 1x1 with padding=0, stride=1: 
  imagedrop=[0.2,0]
  layerdrop=[0.5,0]

  images_d=tf.layers.dropout(images,imagedrop[mflag])
  # Display the training images in the visualizer.
  #tf.summary.image('images_dropout', images_d, max_outputs=3)

  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 3, 96],
                                         stddev=5e-2,wd=0.001)
    conv = tf.nn.conv2d(images_d, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm=batch_norm(pre_activation, 96)
    conv1 = tf.nn.relu(norm, name=scope.name)

  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 96, 96],
                                         stddev=5e-2,wd=0.001)
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm=batch_norm(pre_activation, 96)
    conv2 = tf.nn.relu(norm, name=scope.name)
  
  # replace pooling
  with tf.variable_scope('conv3') as scope: 
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 96, 96],
                                         stddev=5e-2,wd=0.001)
    conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm=batch_norm(pre_activation, 96)
    conv3 = tf.nn.relu(norm, name=scope.name)
  
  #dropout
  conv3_d=tf.layers.dropout(conv3, layerdrop[mflag])

  # norm1
  #norm1 = tf.nn.lrn(conv3_d, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 96, 192],
                                         stddev=5e-2,wd=0.001)
    conv = tf.nn.conv2d(conv3_d, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm=batch_norm(pre_activation, 192)
    conv4 = tf.nn.relu(norm, name=scope.name)

  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 192, 192],
                                         stddev=5e-2,wd=0.001)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm=batch_norm(pre_activation, 192)
    conv5 = tf.nn.relu(norm, name=scope.name)
  

  # replace pooling
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 192, 192],
                                         stddev=5e-2,wd=0.001)
    conv = tf.nn.conv2d(conv5, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm=batch_norm(pre_activation, 192)
    conv6 = tf.nn.relu(norm, name=scope.name)

  #dropout
  conv6_d=conv3_d=tf.layers.dropout(conv6, layerdrop[mflag])

  # norm2
  #norm2 = tf.nn.lrn(conv6_d, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')

  with tf.variable_scope('conv7') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 192, 192],
                                         stddev=5e-2,wd=0.001)
    conv = tf.nn.conv2d(conv6_d, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm=batch_norm(pre_activation, 192)
    conv7 = tf.nn.relu(norm, name=scope.name)

  with tf.variable_scope('conv8') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[1, 1, 192, 192],
                                         stddev=5e-2,wd=0.001)
    conv = tf.nn.conv2d(conv7, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm=batch_norm(pre_activation, 192)
    conv8 = tf.nn.relu(norm, name=scope.name)

  with tf.variable_scope('conv9') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[conv8.get_shape()[1].value,
                                         conv8.get_shape()[2].value, 192, 96],
                                         stddev=5e-2,wd=0.001)
    conv = tf.nn.conv2d(conv8, kernel, [1, 1, 1, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    norm=batch_norm(pre_activation, 96)
    conv9 = tf.nn.relu(norm, name=scope.name)
    


  with tf.variable_scope('softmax_linear') as scope:
    reshape=tf.reshape(conv9,[conv9.get_shape()[0].value,-1])
    weights = _variable_with_weight_decay('weights',shape=[96, 10],
                                         stddev=5e-2,wd=0.001)
    biases = _variable_on_cpu('biases', [10], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)
    #softmax=tf.layers.average_pooling2d(conv10,pool_size=conv10.get_shape()[1].value,strides=1)
    #softmax_linear=tf.reshape(softmax, [softmax.get_shape()[0].value, -1])

  
  return softmax_linear


def KmeansInitial():
  print('K-means Model initializing...')
  weights = {
      'conv1':_variable_with_weight_decay('weight1',shape=[5, 5, 3, 6],stddev=5e-2,wd=None),
      'conv2':_variable_with_weight_decay('weight2',shape=[5, 5, 6, 16],stddev=5e-2,wd=None),
      'local3':_variable_with_weight_decay('weight3', shape=[400, 120],stddev=5e-2, wd=0.004),
      'local4':_variable_with_weight_decay('weight4', shape=[120, 84],stddev=5e-2, wd=0.004),
      'softmax_linear':_variable_with_weight_decay('weight5', [84, NUM_CLASSES],stddev=1/84.0, wd=None)
  }

  biases = {
      'conv1':_variable_on_cpu('biase1', [6], tf.constant_initializer(0.0)),
      'conv2':_variable_on_cpu('biase2', [16], tf.constant_initializer(0.1)),
      'local3':_variable_on_cpu('biase3', [120], tf.constant_initializer(0.1)),
      'local4':_variable_on_cpu('biase4', [84], tf.constant_initializer(0.1)),
      'softmax_linear':_variable_on_cpu('biase5', [NUM_CLASSES],tf.constant_initializer(0.0))
  }
  init=tf.global_variables_initializer()
  with tf.Session() as sess:
    trainimages, trainlabels = inputs(False,FLAGS.K_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      sess.run(init)
      trimages,trainlabels=sess.run([trainimages,trainlabels])
      trlabels=np.zeros((trainlabels.shape[0],10))
      for i in range(0,trainlabels.shape[0]):
        trlabels[i,trainlabels[i]]=1
      #conv1
      buff = array("d")
      for index in range(0,trimages.shape[0]):
        image=trimages[index]
        sys.stdout.write('\r>>Conv1 weight Porcessing:%.2f%%..'%(index/FLAGS.K_size*100))
        for h in range(0,image.shape[0]-4):
          for w in range(0,image.shape[1]-4):
            patch=image[h:h+5,w:w+5,:]
            patch=np.reshape(patch,(1,-1))
            u=np.mean(patch)
            mpatch=patch-u#np.c_[u,patch] #augumentation
            std=np.std(mpatch)
            if std>0.5:
              det=LA.norm(mpatch,2)
              npatch=mpatch/det
              for i in range(0,75):
                buff.append(npatch[0,i])
      
      patches=np.frombuffer(buff,dtype=np.float).reshape(-1,75)
      sys.stdout.write('\r>>Conv1 weight Porcessing: 100%')
      print(patches.shape)
      print('Kmeans classifying...')
      #ncentroids=np.zeros((6,75))
      centroids,_=kmeans(patches,6)
      for index in range(0,6):
        det=LA.norm(centroids[index,:],2)
        centroids[index,:]=centroids[index,:]/det

      for i in range(6):
        slice1=np.reshape((centroids[i,:]),(5,5,3))
        slice1=np.expand_dims(slice1, axis=3)
        if(i==0):
          weight1=slice1
        else:
          weight1=np.append(weight1, slice1, axis=3)
  
      op1= tf.assign(weights['conv1'],weight1)
    
      
      #conv2
      c2tmp = tf.nn.conv2d(trimages, weight1, [1, 1, 1, 1], padding='VALID')
      c2pre = tf.nn.bias_add(c2tmp, biases['conv1'])
      conv1 = tf.nn.relu(c2pre)
      pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      images = sess.run(pool1)
      buff = array("d")
      for index in range(0,images.shape[0]):
        image=images[index]
        sys.stdout.write('\r>>Conv2 weight Porcessing:%.2f%%..'%(index/FLAGS.K_size*100))
        for h in range(0,image.shape[0]-4):
          for w in range(0,image.shape[1]-4):
            patch=image[h:h+5,w:w+5,:]
            patch=np.reshape(patch,(-1))
            mpatch=patch-np.mean(patch)
            std=np.std(mpatch)
            if std > 1.6:
              det=LA.norm(mpatch,2)
              npatch=mpatch/det
              for i in range(0,150):
                buff.append(npatch[i])
                
      patches=np.frombuffer(buff,dtype=np.float).reshape(-1,150)
      sys.stdout.write('\r>>Conv2 weight Porcessing: 100%')
      print(patches.shape)
      print('>>Kmeans classifying...')
      centroids,_=kmeans(patches,16)
      for index in range(0,16):
        det=LA.norm(centroids[index,:],2)
        centroids[index,:]/=det

      for i in range(16):
        slice1=np.reshape((centroids[i,:]),(5,5,6))
        slice1=np.expand_dims(slice1, axis=3)
        if(i==0):
          weight2=slice1
        else:
          weight2=np.append(weight2, slice1, axis=3)
    
      op2 = tf.assign(weights['conv2'],weight2)
      
     
      #local3
      
      l3tmp = tf.nn.conv2d(pool1 , weight2, [1, 1, 1, 1], padding='VALID')
      l3pre = tf.nn.bias_add(l3tmp, biases['conv2'])
      conv2 = tf.nn.relu(l3pre)
      pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
      local3 = tf.reshape(pool2, [FLAGS.K_size, -1])
      
      cdata=sess.run(local3)
      buff = array("d")
      for index in range(0,cdata.shape[0]):
        sys.stdout.write('\r>>Conv3 weight Porcessing:%.2f%%..'%(index/FLAGS.K_size*100))
        patch=cdata[index]
        mpatch=patch-np.mean(patch)
        std=np.std(mpatch)
        if std > 0.3:
          det=LA.norm(mpatch,2)
          npatch=mpatch/det
          for i in range(0,400):
            buff.append(npatch[i])
      patches=np.frombuffer(buff,dtype=np.float).reshape(-1,400)
      sys.stdout.write('\r>>local3 weight Porcessing: 100%')
      print(patches.shape)
      print('>>Kmeans classifying...')
      '''
      centroids,_=kmeans(patches,120)
      for index in range(0,120):
        det=LA.norm(centroids[index,:],2)
        centroids[index,:]/=(det*400)

      weight3=np.zeros((400,120))
      for d1 in range(0,400):
        for d2 in range(0,120):
          weight3[d1][d2]=centroids[d2][d1]
	  '''
      weight3=sess.run(weights['local3'])
      op3 = tf.assign(weights['local3'],weight3)

      #local4
      local4=tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.cast(local3,tf.float64), tf.cast(weight3,tf.float64)),
      	tf.cast(biases['local3'],tf.float64)))
     
      cdata2=sess.run(local4)
      buff = array("d")
      for index in range(0,cdata2.shape[0]):
        sys.stdout.write('\r>>local4 weight Porcessing:%.2f%%..'%(index/FLAGS.K_size*100))
        patch=cdata2[index]
        mpatch=patch-np.mean(patch)
        std=np.std(mpatch)
        if std > 0.05:
          det=LA.norm(mpatch,2)
          npatch=mpatch/det
          for i in range(0,120):
            buff.append(npatch[i])
      patches=np.frombuffer(buff,dtype=np.float).reshape(-1,120)
      sys.stdout.write('\r>>local4 weight Porcessing: 100%')
      print(patches.shape)
      print('>>Kmeans classifying...')
      '''
      centroids,_=kmeans(patches,84)
      for index in range(0,84):
        det=LA.norm(centroids[index,:],2)
        centroids[index,:]/=(det*120)

      weight4=np.zeros((120,84))
      for d1 in range(0,120):
        for d2 in range(0,84):
          weight4[d1][d2]=centroids[d2][d1]
       '''
      weight4=sess.run(weights['local4'])
      op4 = tf.assign(weights['local4'],weight4)
	
      
      #softmax_linear
      stmp=tf.matmul(tf.cast(local4,tf.float64), tf.cast(weight4,tf.float64))
      softmax=tf.nn.relu(tf.nn.bias_add(stmp,tf.cast(biases['local4'],tf.float64)))
      cdata3=sess.run(softmax)
      weight5=np.dot(LA.pinv(cdata3),trlabels)
      op5 = tf.assign(weights['softmax_linear'],weight5)     

      with tf.control_dependencies([op1,op2,op3,op4,op5]):
        ops = tf.no_op(name='kmeansinitilizer')
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)

  return weights,biases,ops

def Evaldataread():
  
  with tf.Session() as sess:
    testimages, testlabels = inputs(True,FLAGS.Evalsize)
    trainimages, trainlabels = inputs(False,FLAGS.Evalsize)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      testi,testl,traini,trainl=sess.run([testimages, testlabels,trainimages, trainlabels])
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)

  return tf.constant(testi),tf.constant(testl),tf.constant(traini),tf.constant(trainl)

def cnntrain():
  #print('Loading Evalation Data')
  #testimages,testlabels,trainimages,trainlabels= Evaldataread()
  #inference=LeNet_5_model
  #inference = Improved_model
  inference = kmeansIN
  with tf.variable_scope('') as scope:
    weights,biases,ops=KmeansInitial()
    global_step = tf.contrib.framework.get_or_create_global_step()
    images, labels = distorted_inputs()
    #logits = inference(images,FLAGS.trainflag)
    logits = inference(images,weights,biases)
    total_loss,raw_loss = loss(logits, labels)
    train_op = train(total_loss, global_step)

    batch_acc = accuracy(logits, labels, 'Batch')
    # Validation
    testimages,testlabels=inputs(True,250)
    trainimages,trainlabels=inputs(False,250)
    scope.reuse_variables()
    #testlogists = inference(testimages,FLAGS.testflag)
    #trainlogists = inference(trainimages,FLAGS.testflag)
    testlogists = inference(testimages,weights,biases)
    trainlogists = inference(trainimages,weights,biases)
    test_acc = accuracy(testlogists, testlabels, 'Test')
    train_acc = accuracy(trainlogists, trainlabels, 'Train')
    
  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()
  init = tf.global_variables_initializer()  

  with tf.Session() as sess:
    sess.run(init)
    #kmeans operations
    sess.run(ops)

    summary_op = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter(FLAGS.train_dir,sess.graph)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
    stepcount=0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        start_time=time.time()
        while stepcount < FLAGS.max_steps and not coord.should_stop():
          losssum,lossraw,no_op=sess.run([total_loss,raw_loss,train_op])
          if math.isnan(losssum):
            print('Loss is NaN!')
            exit()
          if stepcount % FLAGS.log_frequency ==0:
            current_time = time.time()
            duration = current_time - start_time
            start_time=time.time()
            examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
            b_acc=sess.run(batch_acc)
            format_str = ('%s: step %d, total_loss = %.2f,raw_loss = %.2f, batch_accuracy = %.2f (%.1f examples/sec)')
            print (format_str % (datetime.now(), stepcount, losssum, lossraw, b_acc,
                                 examples_per_sec))
          
          if stepcount % 100 == 0:
            test_precision,train_precision=sess.run([test_acc,train_acc])
            print ('Test_accuracy: %.2f%%, Train_accuracy: %.2f%% ' % (test_precision*100,train_precision*100))
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Test_Precision', simple_value=test_precision)
            summary.value.add(tag='Train_Precision', simple_value=train_precision)
            summary_writer.add_summary(summary, stepcount)
          #if stepcount % 10000 ==0:
            #saver.save(sess,FLAGS.train_dir+'/variables.chkp')
            # Counts the number of correct predictions.
            true_count = 0
            t_step = 0
            while t_step < 40 and not coord.should_stop():
              test_predictions= sess.run(test_acc)
              true_count += test_predictions
              t_step += 1
            test_precision = true_count / 40
            #counts the training accuracy
            true_count = 0
            t_step = 0
            while t_step < 40 and not coord.should_stop():
              train_predictions= sess.run(train_acc)
              true_count += train_predictions
              t_step += 1
            train_precision = true_count / 40
          stepcount+=1
        

    except Exception as e: 
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)




def main(argv=None):  # pylint: disable=unused-argument
  # Access to experiment data
  maybe_download_and_extract()
  # Build training outcome set
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  # Build Evaluation out set
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)

  #train model
  cnntrain()


if __name__ == '__main__':
  tf.app.run()
