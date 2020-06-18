# --- Dataset部分实现
import numpy as np
import random
from PIL import Image
import os
class DataSet():
    def __init__(self, train_list, test_list):
        self.train_list = train_list
        self.test_list = test_list
        train = []
        test = []
        train_list = self.train_list[:int(len(self.train_list))]
        test_list = self.test_list[:int(len(self.test_list))]
        self.buffer = np.load('buffer.npy')
        # get train data, sample format: 3*(28*28*3)
        # Anchor;Positive; Negative, each image: 28*28*3
        # count = 0
        # for item in self.train_list:
        # for item in train_list:
        #     sample = []
        #     for idx in range(3):
        #         pic_path = "food/" + item[idx] + '.jpg'
        #         pic = Image.open(pic_path)
        #         pic = pic.resize((28, 28))
        #         pic_array = np.asarray(pic) / 255.0
        #         sample.append(pic_array)
        #     train.append(sample)
        #     count += 1
        #     # print("collecting sample: ", count)
        #
        # # for item in self.test_list:
        # for item in test_list:
        #     sample = []
        #     for idx in range(3):
        #         pic_path = "food/" + item[idx] + '.jpg'
        #         pic = Image.open(pic_path)
        #         pic = pic.resize((28, 28))
        #         pic_array = np.asarray(pic) / 255.0
        #         sample.append(pic_array)
        #     test.append(sample)
        #     count += 1
        #     # print("collecting sample: ", count)
        # self.train = np.array(train)
        # self.test = np.array(test)
        # self.num_train_data, self.num_test_data = self.train.shape[0], self.test.shape[0]
        # self.buffer = np.load('buffer.npy')
        print("sample collecting finish")

    # def get_batch(self, batch):
    #     # index = np.random.randint(0, self.num_train_data, batch_size)
    #     return self.buffer[batch, :]
    def get_batch(self, mini_batch):
        batch = []
        for item in mini_batch:
            sample = []
            sample.append(self.buffer[int(item[0])])
            sample.append(self.buffer[int(item[1])])
            sample.append(self.buffer[int(item[2])])
            batch.append(sample)

        return np.array(batch)
    def get_batch_test(self,mini_batch):
        batch = []
        for item in mini_batch:
            batch.append(self.buffer[int(item)])
        return np.array([batch])

    def split_train_val(self,train_list,train_ratio):


        random.shuffle(train_list)
        train = train_list[0:int(train_ratio*len(train_list))]
        validation = train_list[int(train_ratio*len(train_list)):];


        train = []
        val = []

        bound = int(5000 * train_ratio)

        for item in train_list:
            if int(item[0])> bound and (int(item[1])>bound) and (int(item[2])>bound):
                val.append(item)

            if int(item[0])<=bound and (int(item[1])<=bound) and (int(item[2])<=bound):
                train.append(item)

        print(len(train))
        print(len(val))
        print(len(train_list))

        return train,val

    def compress_data_to_numpy(self):
        data = []
        count = 0
        for i in range(10000):
            filename = "food/" + str(i).zfill(5) + ".jpg"
            image = Image.open(filename)
            image = image.resize((28, 28))
            image_array = np.asarray(image) / 255.0
            data.append(image_array)
            count += 1
            print("collecting sample: ", count)
        np.save('buffer', data)
        return data