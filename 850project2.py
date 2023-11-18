import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras import layers


from tensorflow.keras.utils import image_dataset_from_directory


img_height, img_width, channel = 100,100,'rgb'
shape = (100,100,3)
train_path = './Project 2 Data/Data/Train'
val_path = './Project 2 Data/Data/Validation'
test_path ='./Project 2 Data/Data/Test'

print("importing Data")
train_ds = image_dataset_from_directory(
    train_path,
    label_mode="categorical",
    image_size=(100,100),
    shuffle = True,
    color_mode="rgb",
    seed=200,
    )
