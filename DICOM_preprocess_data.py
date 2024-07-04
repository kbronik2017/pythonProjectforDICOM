# --------------------------------------------------
#
#     Copyright (C) 2024


import nibabel as nib
import numpy as np
import os
import shutil
# import cv2
import configparser

# import pandas as pd
import os
# import torch
# from torchvision.io import read_image
# from sklearn.model_selection import train_test_split
# import pydicom
# from pydicom.data import get_testdata_file
# import matplotlib.pyplot as plt

CEND = '\33[0m'
CBOLD = '\33[1m'
CITALIC = '\33[3m'
CURL = '\33[4m'
CBLINK = '\33[5m'
CBLINK2 = '\33[6m'
CSELECTED = '\33[7m'

CBLACK = '\33[30m'
CRED = '\33[31m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'
CBLUE = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE = '\33[36m'
CWHITE = '\33[37m'

CBLACKBG = '\33[40m'
CREDBG = '\33[41m'
CGREENBG = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG = '\33[46m'
CWHITEBG = '\33[47m'

CGREY = '\33[90m'
CRED2 = '\33[91m'
CGREEN2 = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2 = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2 = '\33[96m'
CWHITE2 = '\33[97m'

CGREYBG = '\33[100m'
CREDBG2 = '\33[101m'
CGREENBG2 = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2 = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2 = '\33[106m'
CWHITEBG2 = '\33[107m'

THIS_PATH = os.path.split(os.path.realpath(__file__))[0]

data_gen_args = dict(rescale=1. / 256)
mask_gen_args = dict()


def normalize(img, mask=None):
    img_data = img.get_data()
    if mask is not None and not isinstance(mask, str):
        mask_data = mask.get_data()
    elif mask == 'nomask':
        mask_data = img_data == img_data
    else:
        mask_data = img_data > img_data.mean()
    logical_mask = mask_data == 1  # force the mask to be logical type
    mean = img_data[logical_mask].mean()
    std = img_data[logical_mask].std()
    normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
    return normalized


def flatten_this(l):
    return flatten_this(l[0]) + (flatten_this(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]


def get_set_input_images(settings):
    # TRAIN_DIR1 = settings['training_folder1'] + '/data/'

    def print_info(filepath, data):
        filenames = [f for f in os.listdir(filepath)]
        input_files = [f for f in filenames if
                       any(filetype in f.lower() for filetype in ['.dcm'])]
        print(">> Found {} {} in {}".format(len(input_files), data, filepath))
        return len(input_files), input_files


    Dicom_DIR = settings['all_image_folder']
    image_count, imgs = print_info(Dicom_DIR, 'dicom images')

    return image_count, imgs



def preprocess_run(settings):

    return get_set_input_images(settings)








