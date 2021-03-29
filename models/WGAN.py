# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 18:49:05 2021

@author: bruce
"""
import numpy as np
from numpy import load
from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input,Dense,Reshape,Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.constraints import Constraint
from matplotlib import pyplot
 
# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value=0.01):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}
 
# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)
 
# define the standalone critic model
def define_critic(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# weight constraint
	const = ClipConstraint(0.01)
	# define model
	#in_src_image = Input(shape=image_shape)
	in_src = Input(shape=image_shape)
	# downsample to 128x128
	c=Conv2D(64, (4,4), strides=(4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(in_src)
	c=BatchNormalization()(c)
	c=LeakyReLU(alpha=0.2)(c)
	# downsample to 32x32
	c=Conv2D(64, (4,4), strides=(4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(in_src)
	c=BatchNormalization()(c)
	c=LeakyReLU(alpha=0.2)(c)
	# downsample to 8x8
	c=Conv2D(64, (4,4), strides=(4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(in_src)
	c=BatchNormalization()(c)
	c=LeakyReLU(alpha=0.2)(c)
	# scoring, linear activation
	c=Flatten()(c)
	c=Dense(1)(c)
	model = Model(in_src, c)
	# compile model
	opt = RMSprop(lr=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model
#%%
# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g

# define the standalone generator model
def define_generator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(image_shape[2], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model
#%%

# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
	# make weights in the critic not trainable
	for layer in critic.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the critic
	model.add(critic)
	# compile model
	opt = RMSprop(lr=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model


def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2, X3, X4 = data['arr_0'], data['arr_1'],data['arr_2'], data['arr_3']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    X3 = (X3 - 127.5) / 127.5
    X4 = (X4 - 127.5) / 127.5
    return [X1, X2, X3, X4]


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    idx = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[idx], trainB[idx]
    # generate 'real' class labels (1)
    y = -ones((n_samples, 1))
    return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = ones((len(X), 1))
    return X, y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = np.squeeze((X_realA + 1) / 2.0)
    X_realB = np.squeeze((X_realB + 1) / 2.0)
    X_fakeB = np.squeeze((X_fakeB + 1) / 2.0)
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_%04d.png' % (step)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%04d.h5' % (step)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
 
# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('plot_line_plot_loss.png')
	pyplot.close()
 
# train the generator and critic
def train(g_model, c_model, gan_model, dataset, n_epochs=2, n_batch=1, n_critic=2):
	# calculate the number of batches per training epoch
	trainA, trainB = dataset
    # calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	# lists for keeping track of loss
	c1_hist, c2_hist, g_hist = list(), list(), list()
	# manually enumerate epochs
	for i in range(n_steps):
		# update the critic more than the generator
		c1_tmp, c2_tmp = list(), list()
		for _ in range(n_critic):
			# get randomly selected 'real' samples
			[X_real, Y_real], y_real_lable = generate_real_samples(dataset, n_batch)
			# update critic model weights
			c_loss1 = c_model.train_on_batch(Y_real, y_real_lable)
			c1_tmp.append(c_loss1)
			# generate 'fake' examples
			Y_fake, y_fake_lable = generate_fake_samples(g_model, X_real)
			# update critic model weights
			c_loss2 = c_model.train_on_batch(Y_fake, y_fake_lable)
			c2_tmp.append(c_loss2)
		# store critic loss
		c1_hist.append(mean(c1_tmp))
		c2_hist.append(mean(c2_tmp))
		# create inverted labels for the fake samples
		y_gan = -ones((n_batch, 1))
		# update the generator via the critic's error
		g_loss = gan_model.train_on_batch(Y_real, y_gan)
		g_hist.append(g_loss)
		# summarize loss on this batch
		print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
		# evaluate the model performance every 'epoch'
		if (i+1) % 2*bat_per_epo == 0:
			summarize_performance((i+1)/(2*bat_per_epo), g_model, dataset)
	# line plots of loss
	plot_history(c1_hist, c2_hist, g_hist)
#%%

dataset = load_real_samples('data_split.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
X_train=(dataset[0])
y_train=(dataset[1])
X_test=(dataset[2])
y_test=(dataset[3])
traindataset=[X_train,y_train]
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the modelss
g_model = define_generator(image_shape)
# create the critic
critic = define_critic(image_shape)
# define the composite model
gan_model = define_gan(g_model, critic)
# load image data
# train model
train(g_model, critic, gan_model, traindataset, n_epochs=100)
