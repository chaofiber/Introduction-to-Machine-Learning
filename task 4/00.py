# --- 一个简易网络实现
import tensorflow as tf
import numpy as np
from alex import triplet_loss,alex_model
from temp import DataSet
import random
import keras
X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])

class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.bn2 = tf.keras.layers.BatchNormalization(axis=1)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=256)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.bn1(x)
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.bn2(x)
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        output = self.dense2(x)                      # [batch_size, 10]
        # output = tf.nn.softmax(x)
        return output

# 以下代码结构与前节类似
num_epochs = 10
batch_size = 16
learning_rate = 0.005
model = CNN()
train_path = './train_triplets.txt'
test_path = './test_triplets.txt'
data_saved = True
train_list = []
test_list = []
with open(train_path) as f:
    for line in f:
        line = line.rstrip('\n')
        inner_list = [elt for elt in line.split(' ')]
        train_list.append(inner_list)

with open(test_path) as f:
    for line in f:
        line = line.rstrip('\n')
        inner_list = [elt for elt in line.split(' ')]
        test_list.append(inner_list)

data_loader = DataSet(train_list, test_list)

# buffer = np.load('buffer.npy')

# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
# num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
# print('number of batches: ',num_batches)
# index = [i for i in range(data_loader.num_train_data)]
# mini_batches = [ index[k:k+batch_size]
#                  for k in range(0,data_loader.num_train_data-batch_size,batch_size)]
# for batch_index in range(num_batches):
train_list,val_list = data_loader.split_train_val(train_list,0.9)
max_acc = 0
for epoch in range(num_epochs):
    sample_path = './sample_{}.txt'.format(epoch)
    temp = random.sample(val_list, int(len(val_list) * 0.01))
    test_names = [temp[k] for k in range(0, len(temp))]
    corr_num = 0
    test_total_loss = 0
    temp = random.sample(train_list, int(len(train_list)))

    mini_batches = [temp[k:k + batch_size] for k in range(0, len(temp) - batch_size, batch_size)]

    for iteration, mini_batch in enumerate(mini_batches):
        X = data_loader.get_batch(mini_batch)  # batchsize *3 * 28 * 28* 3
        # X = data_loader.get_batch(batch)
        # X = temp_val
        with tf.GradientTape() as tape:
            anchor,pos,neg = X[:,0],X[:,1],X[:,2]
            anchor_emb,positive_emb,negative_emb = model(anchor),model(pos),model(neg)
            loss = triplet_loss([anchor_emb,positive_emb,negative_emb])
        if iteration % 20 == 0:
            print("epoch %d : batch %d: loss %f" % (epoch, iteration, loss.numpy()))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

    for test_iter, mini_batch in enumerate(test_names):
        X = data_loader.get_batch_test(mini_batch)  # 3 * 28 * 28* 3
        anchor,pos,neg = X[:,0],X[:,1],X[:,2]
        anchor_emb,positive_emb,negative_emb = model(anchor),model(pos),model(neg)
        positive_distance = np.mean(np.square(anchor_emb - positive_emb))
        negative_distance = np.mean(np.square(anchor_emb - negative_emb))

        loss = triplet_loss([anchor_emb,positive_emb,negative_emb])
        test_total_loss += loss
        # print('loss value for {}: {}'.format(mini_batch,loss))
        if positive_distance < negative_distance:
            corr_num += 1
    test_avg_loss = test_total_loss/len(test_names)
    accuracy = corr_num/len(test_names)
    print('accuracy ',accuracy,' test avg loss: ',test_avg_loss)
    if accuracy>max_acc:
        f = open(sample_path,'w+')
        for test_iter, mini_batch in enumerate(test_list):
            X = data_loader.get_batch_test(mini_batch)  # 3 * 28 * 28* 3
            anchor,pos,neg = X[:,0],X[:,1],X[:,2]
            anchor_emb,positive_emb,negative_emb = model(anchor),model(pos),model(neg)
            positive_distance = np.mean(np.square(anchor_emb - positive_emb))
            negative_distance = np.mean(np.square(anchor_emb - negative_emb))

            loss = triplet_loss([anchor_emb,positive_emb,negative_emb])
            test_total_loss += loss
            # print('loss value for {}: {}'.format(mini_batch,loss))
            if positive_distance < negative_distance:
                f.write('1\n')
            else:
                f.write('0\n')
        f.close()

