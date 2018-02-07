from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import listdir
from os.path import join
import cv2
import numpy as np
import pandas as pd
from PIL import Image

def get_images(directory, flatten=True):

    img_dict = {}

    for f in listdir(directory):
        img_dict[f] = np.array(Image.open(join(directory, f)).convert('RGB')).ravel()

    img_df = pd.DataFrame(list(img_dict.items()), columns=['Image', 'Vector'])

    return img_df
