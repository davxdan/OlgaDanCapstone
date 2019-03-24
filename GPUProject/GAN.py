# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:32:22 2019

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
#image global
img_rows = 28
img_cols = 28
channel = 1
#Data
x_train = input_data.read_data_sets("mnist", one_hot=True).train.images.reshape(-1, img_rows, img_cols, 1).astype(np.float32)
#DL global
train_steps=10
batch_size=256
D = Sequential() #Descriminator
depth = 64
dropout = 0.4
# In: 28 x 28 x 1, depth = 1
# Out: 14 x 14 x 1, depth=64
input_shape = (img_rows, img_cols, channel)
D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
D.add(LeakyReLU(alpha=0.2))
D.add(Dropout(dropout))
D.add(Flatten())# Out: 1-dim probability
D.add(Dense(1))
D.add(Activation('sigmoid'))
D.summary()
G = Sequential() #Generator
dropout = 0.4
depth = 64+64+64+64
dim = 7
# In: 100
# Out: dim x dim x depth
G.add(Dense(dim*dim*depth, input_dim=100))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))
G.add(Reshape((dim, dim, depth)))
G.add(Dropout(dropout))
# In: dim x dim x depth
# Out: 2*dim x 2*dim x depth/2
G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))
G.add(UpSampling2D())
G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))
G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
G.add(BatchNormalization(momentum=0.9))
G.add(Activation('relu'))
# Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
G.add(Conv2DTranspose(1, 5, padding='same'))
G.add(Activation('sigmoid'))
G.add(D) #Add discribmiator to Generator
G.summary() 
#Compile networks
G.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001, decay=3e-8), metrics=['accuracy'])
D.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0002, decay=6e-8), metrics=['accuracy'])
for i in range(train_steps):
    # get random images
    images_train = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]
    images_fake = G.predict(np.random.uniform(-1.0, 1.0, size=[batch_size, 100]))
    import pdb; pdb.set_trace()
    plt.imshow(np.reshape(images_fake, [16, 16]), cmap='gray')
    plt.show()    
    x = np.concatenate((images_train, images_fake))
    y = np.ones([2*batch_size, 1])
    y[batch_size:,:] = 0
    
    d_loss = D.train_on_batch(x, y)
    a_loss = G.train_on_batch( np.random.uniform(-1.0, 1.0, size=[batch_size, 100]), np.ones([batch_size, 1]))
    print('bwa')
import pdb; pdb.set_trace()