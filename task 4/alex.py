import numpy as np
from keras import backend as K
# from keras import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input
# from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import normalize
# from vggface import VggFace
# import cfg
# from data import LFWReader, ARFaceReader, PCDReader, MixedReader, PEALReader
# from data import TripletGenerator
import tensorflow as tf
from tensorflow import keras
from keras import *
from keras.layers import *



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


def triplet_loss_np(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs[0],inputs[1],inputs[2]
    positive_distance = np.square(anchor - positive)
    negative_distance = np.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = np.sqrt(np.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = np.sqrt(np.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = np.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = np.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = np.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = np.log(1 + np.exp(loss))
    return np.mean(loss)



def GetModel():
    embedding_model = VggFace(is_origin=True)
    input_shape = (3, cfg.image_size, cfg.image_size)
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]

    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))
    return embedding_model, triplet_model

class alex_model(tf.keras.Model):
    # K.set_image_dim_ordering('tf')
    # AlexNet
    def __init__(self):
        super().__init__()
        self.model = keras.Sequential()
        # anchor = Input((28, 28, 3))
        # positive = Input((28, 28, 3))
        # negative = Input((28, 28, 3))
        # 第一段

        self.model.add(Conv2D(filters=96, kernel_size=(7, 7),strides=(2, 2), padding='valid',
                         input_shape=(28, 28, 3),activation='relu'))
        # K.int_shape()
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2),
                               padding='valid'))
        # 第二段
        self.model.add(Conv2D(filters=256, kernel_size=(5, 5),strides=(1, 1), padding='same',
                         activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='valid'))
        # 第四段
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
    def call(self,inputs):
        anchor = inputs[:,0]
        positive = inputs[:,1]
        negative = inputs[:,2]

        anc_embedding = self.model(anchor)
        pos_embedding = self.model(positive)
        neg_embedding = self.model(negative)
        # merge_model = Model([anchor,positive,negative],[output1,output2,output3])
        output = tf.concat([anc_embedding,pos_embedding,neg_embedding],1)

        return output



def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = triplet_loss(outputs, model(inputs))
        dW, db = t.gradient(current_loss, [model.W, model.b])
        model.W.assign_sub(learning_rate * dW)
        model.b.assign_sub(learning_rate * db)

def grad(model, inputs):
    with tf.GradientTape() as tape:
        # loss_value = triplet_loss(model, inputs, training=True)
        loss_value = triplet_loss(model(inputs))
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

