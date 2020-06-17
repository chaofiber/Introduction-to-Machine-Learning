from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# import xplot


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

class Model:

	def __init__(self, opt):

		self.batch_size = opt.batch_size
		self.height = 28
		self.width = 28
		self.channels = 3
		self.outdim = 32
		self.lr = opt.lr
		self.beta1 = 0.9
		self.Isloadcheckpoint = False
		self.path = '.'
		self.graph = tf.Graph()
		self.step = 0

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
		self.Image =  tf.placeholder(tf.float32, shape=[self.batch_size,3,self.height, self.width, self.channels])
		print(self.Image)

		Image_anchor = self.Image[:,0,:,:,:]
		Image_positove = self.Image[:,1,:,:,:]
		Image_negative = self.Image[:,2,:,:,:]

		# Image_anchor = tf.reshape(self.Image[:,0,:,:,:],[self.batch_size,self.height,self.width,self.channels])
		# Image_positove = tf.reshape(self.Image[:,1:,:,:],[self.batch_size,self.height,self.width,self.channels])
		# Image_negative = tf.reshape(self.Image[:,2,:,:,:],[self.batch_size,self.height,self.width,self.channels])

		anchor_embedding = encoder(Image_anchor,self.outdim)
		positive_embedding = encoder(Image_positove,self.outdim)
		negative_embedding = encoder(Image_negative,self.outdim)


		self.loss = tf.reduce_mean( tf.reduce_sum(tf.square(anchor_embedding-positive_embedding)-tf.square(anchor_embedding - negative_embedding),axis=1))

		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.loss)

		tf.summary.scalar('loss',self.loss)
		self.summary_op = tf.summary.merge_all()



	def update(self,Image):

		loss = self.sess.run([self.loss],
			feed_dict = {
			    self.Image : Image
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

	def test(self,Image):

		loss = self.sess.run([self.loss],
			feed_dict = {
			    self.Image : Image
			})

		return loss
