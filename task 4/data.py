import numpy as np
import random
from PIL import Image
import os

class DataSet():
	
	def __init__(self, train_list, test_list):
		self.train_list = train_list
		self.test_list = test_list

	def get_data(self):
		
		train = []
		test = []
		
		# get train data, sample format: 3*(28*28*3)
		# Anchor;Positive;Negative, each image: 28*28*3
		count = 0
		for item in self.train_list:
			sample = []
			for idx in range(3):
				pic_path = "food/"+ item[idx]+'.jpg'
				pic = Image.open(pic_path)
				pic = pic.resize((28,28))
				pic_array = np.asarray(pic)/255.0
				sample.append(pic_array);
			train.append(sample)
			count += 1
			print("collecting sample: ", count)

		for item in self.test_list:
			sample = []
			for idx in range(3):
				pic_path = "food/"+item[idx]+'.jpg'
				pic = Image.open(pic_path)
				pic = pic.resize((28,28))
				pic_array = np.asarray(pic)/255.0
				sample.append(pic_array)
			test.append(sample)
			count += 1
			print("collecting sample: ", count)

		print("sample collecting finish")
		np.save('train',train)
		np.save('test',test);
		return train, test

	def split_train_val(self,data,train_ratio):

		random.shuffle(data)
		train = data[0:int(train_ratio*len(data))]
		validation = data[int(train_ratio*len(data))];

		return train,validation

	def compress_data_to_numpy(self):
		data = []
		count = 0
		for i in range(10000):
			filename = "food/"+ str(i).zfill(5) + ".jpg"
			image = Image.open(filename)
			image = image.resize((224,224))
			image_array = np.asarray(image)
			data.append(image_array)
			count += 1
			print("collecting sample: ", count)
		np.save('buffer',data)
		return data

	def get_batch(self,buffer,mini_batch):

		batch = []

		for item in mini_batch:
			sample = []
			sample.append(buffer[int(item[0])])
			sample.append(buffer[int(item[1])])
			sample.append(buffer[int(item[2])])
			batch.append(sample)

		return batch


