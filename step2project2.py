from tensorflow.keras.utils import load_img,img_to_array
from tensorflow.keras.models import load_model
import numpy as np

from warnings import filterwarnings
from tensorflow import io
from tensorflow import image
from matplotlib import pyplot as plt


def image_loader(path):
    
    shape = [100,100]
    load = load_img(path, target_size = shape)
    input_arr = img_to_array(load)
    input_arr = np.array([input_arr])
    
    
    return input_arr

