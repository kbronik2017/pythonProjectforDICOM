# --------------------------------------------------
#
#     Copyright (C) 2024

import os
import math
import shutil


import cv2, os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import pydicom

# application_window = tk.Tk()
#
# # Build a list of tuples for each file type the file dialog should display
# my_filetypes = [('all files', '.*'), ('text files', '.txt')]
#
# # Ask the user to select a folder.
# answer = filedialog.askdirectory(parent=application_window,
#                                  initialdir=os.getcwd(),
#                                  title="Please select a folder:")

# Ask the user to select a single file name.
# answer = filedialog.askopenfilename(parent=application_window,
#                                     initialdir=os.getcwd(),
#                                     title="Please select a file:",
#                                     filetypes=my_filetypes)
# application_window.destroy()
# print(answer)


def convertor(base_path):
    # base_path = "/home/kevinb/Videos/RETINA_PCVAE-OXFORDX_Diffusion/diffusion_example/image/"
    # new_path = "/home/kevinb/Videos/RETINA_PCVAE-OXFORDX_Diffusion/TRAN_VALID/IMAGES/data/"

    tmp = os.path.normpath(os.path.join(base_path, 'temp'))

    try:
        # os.rmdir(os.path.join(current_folder,  'tmp'))
        if os.path.exists(tmp) is True:
            shutil.rmtree(tmp)
            os.mkdir(tmp)
            print("tmp folder is created for training!")
        else:
            os.mkdir(tmp)
    except:
        print("I can not create data folder")

    new_path = base_path + '/temp/'
    print('new_path -->',  new_path)
    base_path = base_path
    print('base_path -->', base_path)
    for infile in os.listdir(base_path):
        # print("file : " + infile)
        img_path = os.path.join(base_path)
        print("Transform DICOM To Tensor")
        print(img_path + "/" + infile)
        ds = pydicom.dcmread(img_path + "/" + infile)
        read = ds.pixel_array
        # read = cv2.imread(base_path + infile)
        outfile = infile.split('.')[0] + '.jpg'
        cv2.imwrite(new_path + outfile, read, [int(cv2.IMWRITE_JPEG_QUALITY), 200])

