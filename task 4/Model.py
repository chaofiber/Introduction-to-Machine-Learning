from __future__ import division
from __future__ import print_function

import numpy as np
import h5py
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.keras.applications.resnet50 import ResNet50
from keras import backend as K


# import xplot
def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'): # ---这两个loss部分似乎不对，但我不知道这么改
    # anchor, positive, negative = inputs
    inputs = K.l2_normalize(inputs, axis=1)
    dist = 'sqeuclidean'
    anchor, positive, negative = inputs[0],inputs[1],inputs[2]
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 10 + loss)
    elif margin == 'softplus':
        loss = K.log(10 + K.exp(loss))
    return K.mean(loss)


def encoder(x,outdim):

		with tf.variable_scope('alex', reuse=tf.AUTO_REUSE):

			# kwargs = dict(padding='same',strides=2, activation=tf.nn.relu)

			c1 = tf.layers.conv2d(x,filters=96,kernel_size=(7, 7),strides=(2, 2),padding='valid',activation=tf.nn.relu)
			c2 = tf.layers.batch_normalization(c1)
			max_pool_2d = tf.keras.layers.MaxPool2D(pool_size=(3, 3),strides=(2, 2),padding='valid')
			max_pool_2d(c2)
			
			c3 = tf.layers.conv2d(c2,filters=256,kernel_size=(5, 5),strides=(1, 1),padding='same',activation=tf.nn.relu)
			c4 = tf.layers.batch_normalization(c3)

			max_pool_2d(c4)
			
			c5 = tf.layers.flatten(c4)
			out = tf.layers.dense(c5, outdim, None)

		return out	


def encoder_resnet(x,outdim):

	# with tf.variable_scope('resnet',reuse=tf.AUTO_REUSE):
	base_model = ResNet50(input_shape=(224,224,3),include_top=False,weights='imagenet')
	base_model.trainable=False
	last = base_model(x)
	c = tf.layers.flatten(last)
	# c1 = tf.layers.dense(c,1024,None)
	# c1 = tf.keras.layers.Dropout(c1,0.5)
	c = tf.layers.dense(c,512,None)
	out = tf.layers.dense(c,outdim,None)

	return out


class Model:

	def __init__(self, opt):

		self.batch_size = opt.batch_size
		self.height = 224
		self.width = 224
		self.channels = 3
		self.outdim = 256
		self.lr = opt.lr
		self.beta1 = 0.9
		self.Isloadcheckpoint = False
		self.path = '.'
		self.graph = tf.Graph()
		self.step = 0
		self.network = opt.network

		with self.graph.as_default():

			tf.set_random_seed(0);

			with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
				self.create_network()

			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True

			self.sess = tf.Session(config = config)
			self.sess.run(tf.global_variables_initializer())

			self.saver = tf.train.Saver()

			if self.Isloadcheckpoint:
				checkpoint = tf.train.get_checkpoint_state(self.path+'saved_networks/')

				if checkpoint and checkpoint.model_checkpoint_path:
					print("load_path", checkpoint.model_checkpoint_path)
					self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
					print("successfully loaded", checkpoint.model_checkpoint_path)
				else:
					print("could not find network")
					os._exit()

			if tf.gfile.Exists(self.path + 'summary_train/'):
				tf.gfile.DeleteRecursively(self.path + 'summary_train/')
			if tf.gfile.Exists(self.path + 'summary_test/'):
				tf.gfile.DeleteRecursively(self.path + 'summary_test/')

			self.train_writer = tf.summary.FileWriter(self.path + 'summary_train/', graph=self.sess.graph)
			self.test_writer = tf.summary.FileWriter(self.path + 'summary_test/', graph=self.sess.graph)




	def create_network(self):
		self.Image =  tf.placeholder(tf.uint8, shape=[self.batch_size,3,self.height, self.width, self.channels])
		print(self.Image)

		self.Image = tf.cast(self.Image, tf.float32)/255.0

		Image_anchor = self.Image[:,0,:,:,:]
		Image_positove = self.Image[:,1,:,:,:]
		Image_negative = self.Image[:,2,:,:,:]

		# Image_anchor = tf.reshape(self.Image[:,0,:,:,:],[self.batch_size,self.height,self.width,self.channels])
		# Image_positove = tf.reshape(self.Image[:,1:,:,:],[self.batch_size,self.height,self.width,self.channels])
		# Image_negative = tf.reshape(self.Image[:,2,:,:,:],[self.batch_size,self.height,self.width,self.channels])

		# anchor_embedding = encoder(Image_anchor,self.outdim)
		# positive_embedding = encoder(Image_positove,self.outdim)
		# negative_embedding = encoder(Image_negative,self.outdim)
		if self.network=='ResNet50':
			anchor_embedding = encoder_resnet(Image_anchor,self.outdim)
			positive_embedding = encoder_resnet(Image_positove,self.outdim)
			negative_embedding = encoder_resnet(Image_negative,self.outdim)
		else:
			anchor_embedding = encoder(Image_anchor,self.outdim)
			positive_embedding = encoder(Image_positove,self.outdim)
			negative_embedding = encoder(Image_negative,self.outdim)


		# normed_positive_embedding = tf.math.l2_normalize(positive_embedding)
		# normed_negative_embedding = tf.math.l2_normalize(negative_embedding)
		# normed_anchor_embedding = tf.math.l2_normalize(anchor_embedding)

		# self.positive_distance = (tf.reduce_sum(tf.square(normed_anchor_embedding-normed_positive_embedding),axis=-1,keepdims=True))
		# self.negative_distance = (tf.reduce_sum(tf.square(normed_anchor_embedding-normed_negative_embedding),axis=-1,keepdims=True))


		# self.loss = tf.reduce_mean(tf.maximum(0.0,1+self.positive_distance- self.negative_distance) )

		self.loss = triplet_loss([anchor_embedding,positive_embedding,negative_embedding])

		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.loss)

		tf.summary.scalar('loss',self.loss)
		self.summary_op = tf.summary.merge_all()



	def update(self,Image):

		loss,positive_distance,negative_distance = self.sess.run([self.loss, self.positive_distance,self.negative_distance],
			feed_dict = {
			    self.Image : Image,
			})

		self.step += 1
		# xplot('loss',loss)


		if self.step % 1000 == 0:
			self.saver.save(self.sess, self.path + 'saved_networks/',global_step = self.step)

		if self.step % 100 == 0:
			summary = self.sess.run(self.summary_op, feed_dict={self.Image:Image})
			self.train_writer.add_summary(summary, self.step)
			self.train_writer.flush()

		# xplot.tick()
		return loss,positive_distance,negative_distance

	def test(self,Image):

		loss,positive_distance,negative_distance = self.sess.run([self.loss, self.positive_distance,self.negative_distance],
			feed_dict = {
			    self.Image : Image,
			})

		return loss,positive_distance,negative_distance
