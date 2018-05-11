import numpy as np
import tensorflow as tf
from .layer import *
import json
import pdb
with open('info.json') as f:
    ParSet = json.load(f)
pdb.set_trace()

def Net(x):
	# f1
	conv1_1 = conv3D(x, kernel_size=3, in_channels=ParSet['CHANNEL_NUM'], out_channels=32)
	conv1_1_bn = tf.contrib.layers.batch_norm(conv1_1, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv1_1_relu = tf.nn.relu(conv1_1_bn)

	conv1_2 = conv3D(conv1_1_relu, kernel_size=3, in_channels=32, out_channels=32)
	conv1_2_bn = tf.contrib.layers.batch_norm(conv1_2, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)

	conv1_2_relu = tf.nn.relu(conv1_2_bn)
	pool1 = tf.layers.max_pooling3d(inputs=conv1_2_relu, pool_size=2, strides=2, padding='same')
	# f2
	conv2_1 = conv3D(pool1, kernel_size=3, in_channels=32, out_channels=64)
	conv2_1_bn = tf.contrib.layers.batch_norm(conv2_1, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv2_1_relu = tf.nn.relu(conv2_1_bn)

	conv2_2 = conv3D(conv2_1_relu, kernel_size=3, in_channels=64, out_channels=64)
	conv2_2_bn = tf.contrib.layers.batch_norm(conv2_2, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv2_2_relu = tf.nn.relu(conv2_2_bn)

	conv2_3 = conv3D((conv2_2_relu), kernel_size=3, in_channels=64, out_channels=64)
	conv2_3_bn = tf.contrib.layers.batch_norm(conv2_3, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv2_3_relu = tf.nn.relu(conv2_3_bn)
	pool2 = tf.layers.max_pooling3d(inputs=conv2_3_relu, pool_size=2, strides=2, padding='same')

	# f3
	conv3_1 = conv3D(pool2, kernel_size=5, in_channels=64, out_channels=128)
	conv3_1_bn = tf.contrib.layers.batch_norm(conv3_1, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv3_1_relu = tf.nn.relu(conv3_1_bn)

	conv3_2 = conv3D(conv3_1_relu, kernel_size=5, in_channels=128, out_channels=128)
	conv3_2_bn = tf.contrib.layers.batch_norm(conv3_2, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv3_2_relu = tf.nn.relu(conv3_2_bn)

	conv3_3 = conv3D((conv3_2_relu), kernel_size=5, in_channels=128, out_channels=128)
	conv3_3_bn = tf.contrib.layers.batch_norm(conv3_3, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv3_3_relu = tf.nn.relu(conv3_3_bn)
	pool3 = tf.layers.max_pooling3d(inputs=conv3_3_relu, pool_size=2, strides=2, padding='same')

	# f4
	conv4_1 = conv3D(pool3, kernel_size=5, in_channels=128, out_channels=128)
	conv4_1_bn = tf.contrib.layers.batch_norm(conv4_1, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv4_1_relu = tf.nn.relu(conv4_1_bn)

	conv4_2 = conv3D(conv4_1_relu, kernel_size=5, in_channels=128, out_channels=128)
	conv4_2_bn = tf.contrib.layers.batch_norm(conv4_2, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv4_2_relu = tf.nn.relu(conv4_2_bn)

	conv4_3 = conv3D((conv4_2_relu), kernel_size=5, in_channels=128, out_channels=128)
	conv4_3_bn = tf.contrib.layers.batch_norm(conv4_3, decay=0.9, updates_collections=None, epsilon=1e-5,
	                                          scale=True, is_training=True)
	conv4_3_relu = tf.nn.relu(conv4_3_bn)
	# auxiliary prediction 3
	conv_aux3 = conv3D(conv4_3_relu, kernel_size=1, in_channels=128, out_channels=2)
	# up1
	deconv3_1 = deconv_bn_relu(conv4_3_relu, kernel_size=5, in_channels=128, out_channels=128)
	concat_3 = concat_3D(deconv3_1, conv3_3_relu)
	deconv3_2 = conv3d_bn_relu(concat_3, kernel_size=5, in_channels=256, out_channels=128)
	deconv3_3 = conv3d_bn_relu(deconv3_2, kernel_size=5, in_channels=128, out_channels=128)
	# auxiliary prediction 2
	conv_aux2 = conv3D(deconv3_3, kernel_size=1, in_channels=128, out_channels=2)
	# up2
	deconv2_1 = deconv_bn_relu(deconv3_3, kernel_size=3, in_channels=128, out_channels=64)
	concat_2 = concat_3D(deconv2_1, conv2_3_relu)
	deconv2_2 = conv3d_bn_relu(concat_2, kernel_size=3, in_channels=128, out_channels=64)
	deconv2_3 = conv3d_bn_relu(deconv2_2, kernel_size=3, in_channels=64, out_channels=32)
	# auxiliary prediction 1
	conv_aux1 = conv3D(deconv2_3, kernel_size=1, in_channels=32, out_channels=2)
	# up1
	deconv1_1 = deconv_bn_relu(deconv2_3, kernel_size=3, in_channels=32, out_channels=32)
	concat_1 = concat_3D(deconv1_1, conv1_2_relu)
	deconv1_2 = conv3d_bn_relu(concat_1, kernel_size=3, in_channels=64, out_channels=64)
	deconv1_3 = conv3d_bn_relu(deconv1_2, kernel_size=3, in_channels=64, out_channels=32)
	y_conv = conv3D(deconv1_3, kernel_size=1, in_channels=32, out_channels=2)

	return conv_aux3, conv_aux2, conv_aux1, y_conv