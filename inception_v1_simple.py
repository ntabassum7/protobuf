from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def InceptionV1(inputs,
               num_classes=1000,
               is_training=True,
               reuse=None,
               scope='InceptionV1'):


   with tf.variable_scope(scope, "Model", reuse=reuse):
      with slim.arg_scope(default_arg_scope(is_training)):

         end_points = {}

         end_point = 'Conv2d_1a_7x7'
         net = slim.conv2d(inputs, 64, [7,7], stride=2, scope=end_point)
         end_points[end_point] = net
         

         end_point = 'MaxPool_2a_3x3'
         net = slim.max_pool2d(Conv2d_1a_7x7, [3,3], stride=2, scope=end_point)
         end_points[end_point] = net
         

         end_point = 'Conv2d_2b_1x1'
         net = slim.conv2d(MaxPool_2a_3x3, 64, [1,1], stride=1, scope=end_point)
         end_points[end_point] = net
         

         end_point = 'Conv2d_2c_3x3'
         net = slim.conv2d(Conv2d_2b_1x1, 192, [3,3], stride=1, scope=end_point)
         end_points[end_point] = net
         

         end_point = 'MaxPool_3a_3x3'
         net = slim.max_pool2d(Conv2d_2c_3x3, [3,3], stride=2, scope=end_point)
         end_points[end_point] = net
         

Parameters for Layer 15:
Name: Mixed_3b/Branch_0/Conv2d_0a_1x1
         end_point = 'Mixed_3b'
         with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
Parameters for Layer 16:
Name: Mixed_3b/Branch_0/Conv2d_0a_1x1_bn
               end_point = 'Mixed_3b'
               with tf.variable_scope(end_point):
                  with tf.variable_scope('Branch_0'):
Parameters for Layer 17:
Name: Mixed_3b/Branch_0/Conv2d_0a_1x1_scale
                     end_point = 'Mixed_3b'
                     with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
Parameters for Layer 18:
Name: Mixed_3b/Branch_0/Conv2d_0a_1x1_relu
                           end_point = 'Mixed_3b'
                           with tf.variable_scope(end_point):
                              with tf.variable_scope('Branch_0'):
Parameters for Layer 19:
Name: Mixed_3b/Branch_1/Conv2d_0a_1x1
                                 end_point = 'Mixed_3b'
                                 with tf.variable_scope(end_point):
                                    with tf.variable_scope('Branch_1'):
Parameters for Layer 20:
Name: Mixed_3b/Branch_1/Conv2d_0a_1x1_bn
                                       end_point = 'Mixed_3b'
                                       with tf.variable_scope(end_point):
                                          with tf.variable_scope('Branch_1'):
Parameters for Layer 21:
Name: Mixed_3b/Branch_1/Conv2d_0a_1x1_scale
                                             end_point = 'Mixed_3b'
                                             with tf.variable_scope(end_point):
                                                with tf.variable_scope('Branch_1'):
Parameters for Layer 22:
Name: Mixed_3b/Branch_1/Conv2d_0a_1x1_relu
                                                   end_point = 'Mixed_3b'
                                                   with tf.variable_scope(end_point):
                                                      with tf.variable_scope('Branch_1'):
Parameters for Layer 23:
Name: Mixed_3b/Branch_1/Conv2d_0b_3x3
                                                         end_point = 'Mixed_3b'
                                                         with tf.variable_scope(end_point):
                                                            with tf.variable_scope('Branch_1'):
Parameters for Layer 24:
Name: Mixed_3b/Branch_1/Conv2d_0b_3x3_bn
                                                               end_point = 'Mixed_3b'
                                                               with tf.variable_scope(end_point):
                                                                  with tf.variable_scope('Branch_1'):
Parameters for Layer 25:
Name: Mixed_3b/Branch_1/Conv2d_0b_3x3_scale
                                                                     end_point = 'Mixed_3b'
                                                                     with tf.variable_scope(end_point):
                                                                        with tf.variable_scope('Branch_1'):
Parameters for Layer 26:
Name: Mixed_3b/Branch_1/Conv2d_0b_3x3_relu
                                                                           end_point = 'Mixed_3b'
                                                                           with tf.variable_scope(end_point):
                                                                              with tf.variable_scope('Branch_1'):
Parameters for Layer 27:
Name: Mixed_3b/Branch_2/Conv2d_0a_1x1
                                                                                 end_point = 'Mixed_3b'
                                                                                 with tf.variable_scope(end_point):
                                                                                    with tf.variable_scope('Branch_2'):
Parameters for Layer 28:
Name: Mixed_3b/Branch_2/Conv2d_0a_1x1_bn
                                                                                       end_point = 'Mixed_3b'
                                                                                       with tf.variable_scope(end_point):
                                                                                          with tf.variable_scope('Branch_2'):
Parameters for Layer 29:
Name: Mixed_3b/Branch_2/Conv2d_0a_1x1_scale
                                                                                             end_point = 'Mixed_3b'
                                                                                             with tf.variable_scope(end_point):
                                                                                                with tf.variable_scope('Branch_2'):
Parameters for Layer 30:
Name: Mixed_3b/Branch_2/Conv2d_0a_1x1_relu
                                                                                                   end_point = 'Mixed_3b'
                                                                                                   with tf.variable_scope(end_point):
                                                                                                      with tf.variable_scope('Branch_2'):
Parameters for Layer 31:
Name: Mixed_3b/Branch_2/Conv2d_0b_3x3
                                                                                                         end_point = 'Mixed_3b'
                                                                                                         with tf.variable_scope(end_point):
                                                                                                            with tf.variable_scope('Branch_2'):
Parameters for Layer 32:
Name: Mixed_3b/Branch_2/Conv2d_0b_3x3_bn
                                                                                                               end_point = 'Mixed_3b'
                                                                                                               with tf.variable_scope(end_point):
                                                                                                                  with tf.variable_scope('Branch_2'):
Parameters for Layer 33:
Name: Mixed_3b/Branch_2/Conv2d_0b_3x3_scale
                                                                                                                     end_point = 'Mixed_3b'
                                                                                                                     with tf.variable_scope(end_point):
                                                                                                                        with tf.variable_scope('Branch_2'):
Parameters for Layer 34:
Name: Mixed_3b/Branch_2/Conv2d_0b_3x3_relu
                                                                                                                           end_point = 'Mixed_3b'
                                                                                                                           with tf.variable_scope(end_point):
                                                                                                                              with tf.variable_scope('Branch_2'):
Parameters for Layer 35:
Name: Mixed_3b/Branch_3/MaxPool_0a_3x3
                                                                                                                                 end_point = 'Mixed_3b'
                                                                                                                                 with tf.variable_scope(end_point):
                                                                                                                                    with tf.variable_scope('Branch_3'):
Parameters for Layer 36:
Name: Mixed_3b/Branch_3/Conv2d_0b_1x1
                                                                                                                                       end_point = 'Mixed_3b'
                                                                                                                                       with tf.variable_scope(end_point):
                                                                                                                                          with tf.variable_scope('Branch_3'):
Parameters for Layer 37:
Name: Mixed_3b/Branch_3/Conv2d_0b_1x1_bn
                                                                                                                                             end_point = 'Mixed_3b'
                                                                                                                                             with tf.variable_scope(end_point):
                                                                                                                                                with tf.variable_scope('Branch_3'):
Parameters for Layer 38:
Name: Mixed_3b/Branch_3/Conv2d_0b_1x1_scale
                                                                                                                                                   end_point = 'Mixed_3b'
                                                                                                                                                   with tf.variable_scope(end_point):
                                                                                                                                                      with tf.variable_scope('Branch_3'):
Parameters for Layer 39:
Name: Mixed_3b/Branch_3/Conv2d_0b_1x1_relu
                                                                                                                                                         end_point = 'Mixed_3b'
                                                                                                                                                         with tf.variable_scope(end_point):
                                                                                                                                                            with tf.variable_scope('Branch_3'):
Parameters for Layer 40:
Name: Mixed_3b
                                                                                                                                                               end_point = 'Mixed_3b'
                                                                                                                                                               with tf.variable_scope(end_point):
