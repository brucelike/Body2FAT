# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 16:43:22 2020

@author: brucelike
"""

# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
from numpy import load
from numpy.random import randint
from sklearn.model_selection import train_test_split

# load all images in a directory into memory
def load_images(path, size=(512,512)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size,color_mode="grayscale" )
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
#		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		src_list.append(pixels)
#		tar_list.append(map_img)
	return [asarray(src_list)]


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


# dataset path
path1 = '../fat/bd_map/'
path2 = '../fat/sat_map/'
path3 = '../fat/vat_map/'
# load dataset
[bs_map] = load_images(path1)
[sat_map] = load_images(path2)
[vat_map] = load_images(path3)
print('Loaded: ', bs_map.shape, sat_map.shape,vat_map.shape)
X_train, X_test, y_train, y_test,y_train1, y_test1 = train_test_split(bs_map, sat_map,vat_map,
    test_size=0.2, random_state=1000)
# save as compressed numpy array
filename = 'data_split.npz'
savez_compressed(filename, X_train, y_train,y_train1, X_test, y_test, y_test1)
print('Saved dataset: ', filename)



