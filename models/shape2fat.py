# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:43:51 2020

@author: bruce
"""
# -*- coding: utf-8 -*-
"""
Pixel wise cGAN to predict the body fat with body shape
"""
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot
#import ot
#from sklearn.model_selection import train_test_split
import csv
import numpy as np
# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # target image input
    in_target_image_s = Input(shape=image_shape)
    in_target_image_v = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image_s,in_target_image_v])
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image_s,in_target_image_v], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

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
    ds1 = decoder_block(b, e7, 512)
    ds2 = decoder_block(ds1, e6, 512)
    ds3 = decoder_block(ds2, e5, 512)
    ds4 = decoder_block(ds3, e4, 512, dropout=False)
    ds5 = decoder_block(ds4, e3, 256, dropout=False)
    ds6 = decoder_block(ds5, e2, 128, dropout=False)
    ds7 = decoder_block(ds6, e1, 64, dropout=False)
    gs = Conv2DTranspose(image_shape[2], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(ds7)
    out_image_sat = Activation('tanh')(gs)
    dv1 = decoder_block(b, e7, 512)
    dv2 = decoder_block(dv1, e6, 512)
    dv3 = decoder_block(dv2, e5, 512)
    dv4 = decoder_block(dv3, e4, 512, dropout=False)
    dv5 = decoder_block(dv4, e3, 256, dropout=False)
    dv6 = decoder_block(dv5, e2, 128, dropout=False)
    dv7 = decoder_block(dv6, e1, 64, dropout=False)
    # output
    gv = Conv2DTranspose(image_shape[2], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(dv7)

    out_image_vat = Activation('tanh')(gv)
    # define model
    model = Model(in_image, [out_image_sat,out_image_vat])
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    [gen_out_sat, gen_out_vat] = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out_sat,gen_out_vat])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out_sat,gen_out_vat])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,50,100])
    return model

# load and prepare training images
def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2, X3, X4, X5, X6 = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    X3 = (X3 - 127.5) / 127.5
    X4 = (X4 - 127.5) / 127.5
    X5 = (X5 - 127.5) / 127.5
    X6 = (X6 - 127.5) / 127.5
    return [X1, X2, X3, X4, X5, X6]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    im_sp, im_s,im_v = dataset
    # choose random instances
    ix = randint(0, im_sp.shape[0], n_samples)
    # retrieve selected images
    X1, X2 ,X3= im_sp[ix], im_s[ix],im_v[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2, X3], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    gen_sat,gen_vat = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(gen_sat), patch_shape, patch_shape, 1))
    return gen_sat,gen_vat, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    # save the generator model
    filename1 = 'model_%05d.h5' % (step)
    g_model.save(filename1)
    print('>Saved: %s' % (filename1))

# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=10, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    train_bs, _, _   = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(train_bs) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [im_src, real_sat,real_vat], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        gen_sat, gen_vat,y_fake = generate_fake_samples(g_model, im_src, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([im_src, real_sat,real_vat], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([im_src, gen_sat,gen_vat], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(im_src, [y_real, gen_sat,gen_vat])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        print('%d' % (bat_per_epo))
        # summarize model performance
        #per_epo=1
        if (i+1) % (bat_per_epo) == 0:
              #summarize_performance((i+1)/(bat_per_epo * 10) , g_model, dataset)
              summarize_performance((i+1)/(bat_per_epo), g_model, dataset)



#%%
# load image data
filename='data_split.npz'
dataset = load_real_samples(filename)
print('Loaded', dataset[0].shape, dataset[1].shape,dataset[2].shape)
bs_train=(dataset[0])
sat_train=(dataset[1])
vat_train=(dataset[2])
bs_test=(dataset[3])
sat_test=(dataset[4])
vat_test=(dataset[5])
traindataset=[bs_train,sat_train,vat_train]
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the modelss
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# # train model
train(d_model, g_model, gan_model,traindataset,n_epochs=200)
