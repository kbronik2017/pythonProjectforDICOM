# --------------------------------------------------
#
#     Copyright (C) 2024

CEND      = '\33[0m'
CBOLD     = '\33[1m'
CITALIC   = '\33[3m'
CURL      = '\33[4m'
CBLINK    = '\33[5m'
CBLINK2   = '\33[6m'
CSELECTED = '\33[7m'

CBLACK  = '\33[30m'
CRED    = '\33[31m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE  = '\33[36m'
CWHITE  = '\33[37m'

CBLACKBG  = '\33[40m'
CREDBG    = '\33[41m'
CGREENBG  = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG   = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG  = '\33[46m'
CWHITEBG  = '\33[47m'

CGREY    = '\33[90m'
CRED2    = '\33[91m'
CGREEN2  = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2   = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2  = '\33[96m'
CWHITE2  = '\33[97m'

CGREYBG    = '\33[100m'
CREDBG2    = '\33[101m'
CGREENBG2  = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2   = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2  = '\33[106m'
CWHITEBG2  = '\33[107m'

import numpy as np
import matplotlib.pyplot as plt
import glob
import pydicom
import pylab as pl
import sys
import matplotlib.path as mplPath

import numpy as np
# import pandas as pd
import os

import pydicom

import matplotlib.pyplot as plt
import os
import platform
host_os = platform.system()
if host_os == 'Darwin':
    import click
else:
    pass
import shutil

import sys
import configparser

import numpy as np
# from PIL import Image
# from torchvision import transforms
# import matplotlib.transforms as mtransforms
from DICOM_get_settings import dicom_settings, all_settings_show
from manual_dicom_data_augmentation import da_generator, val_generator
from DICOM_preprocess_data import preprocess_run
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset as Dataset
from torch.utils.data import DataLoader



fl_PATH = os.path.split(os.path.realpath(__file__))[0]
def overall_config():

    dm_config = configparser.ConfigParser()
    dm_config.read(os.path.join(fl_PATH, 'config', 'configuration.cfg'))

    settings = dicom_settings(dm_config)
    # set paths taking into account the host OS
    host_os = platform.system()

    if settings['debug']:
        all_settings_show(settings)

    return settings



class experimental_dataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data[idx]
        item = self.transform(item)
        return item






class Dicom_data_vis(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('Scroll to Navigate through the DICOM Image Slices')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('Slice Number: %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def do_plot(ax, Z, transform):
    im = ax.imshow(Z, interpolation='none',
                   origin='lower',
                   extent=[-2, 4, -3, 2], clip_on=True)

    trans_data = transform + ax.transData
    im.set_transform(trans_data)

    # display intended extent of the image
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
            transform=trans_data)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)

def transform_tensor(imgs, settings, folder):
    tnsr = []
    trans = transforms.Compose([transforms.ToTensor()])
    for img in imgs:
        # img_path = os.path.join(settings['all_image_folder'])
        img_path = os.path.join(settings[folder])
        print("Transform DICOM To Tensor")
        print(img_path + "/" +img)
        ds = pydicom.dcmread(img_path + "/" +img)
        pix = ds.pixel_array
        transformed_img = trans(pix)
        tnsr.append(transformed_img)
    return tnsr
# prepare image and figure
# def crop_my_image(image: PIL.Image.Image) -> PIL.Image.Image:
#     """Crop the images so only a specific region of interest is shown to my PyTorch model"""
#     left, right, width, height = 20, 80, 40, 60
#     return transforms.functional.crop(image, left=left, top=top, width=width, height=height)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    settings = overall_config()

    print('\x1b[6;30;41m' + "                         " + '\x1b[0m')
    print('\x1b[6;30;41m' + " Reading DICOM data ...! " + '\x1b[0m')
    print('\x1b[6;30;41m' + "                         " + '\x1b[0m')
    nur, imgs = preprocess_run(settings)

    # for f in imgs:
    #     print(f)

    tnsimg = transform_tensor(imgs, settings, 'all_image_folder')
    print(CRED + "Dimension of the transformed DICOM tensor:" + CEND, np.array(tnsimg).shape)
    transform = transforms.Compose([
        # transforms.Lambda(crop_my_image),
        transforms.ColorJitter(),
        transforms.RandomInvert(),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    print('Original data: \n', tnsimg)

    print('\x1b[6;30;41m' + "                         " + '\x1b[0m')
    print('\x1b[6;30;41m' + " Data augmentation ...!  " + '\x1b[0m')
    print('\x1b[6;30;41m' + "                         " + '\x1b[0m')

    dataset = experimental_dataset(tnsimg, transform)
    print('Transformed-augmented data: \n')
    for item in dataset:
        print(item)
        # # Data augmentation and normalization for training
        # # Just normalization for validation
        # data_transforms = {
        #     'train': transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]),
        #     'val': transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]),
        # }
    # epoch_size = settings['epochs']
    # batch_size = settings['batch_size']
    # for i in range(epoch_size):
    #     print('----------------------------------------------')
    #     print('the epoch', i, 'data: \n')
    #     for item in DataLoader(dataset, batch_size, shuffle=False):
    #        print(item)
    print('\x1b[6;30;41m' + "Models with the neural network layers can be added here alongside PyTorch training loop "
                            "!  " + '\x1b[0m')
