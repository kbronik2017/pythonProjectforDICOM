# --------------------------------------------------
#
#     Copyright (C) 2024


import os
import signal
import threading
import time
import shutil
import numpy as np
# from sources.build_CrossSec_nets import cross_network

import gc


from numpy import inf


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


def transform(Xb, yb):

    # Flip a given percentage of the images at random:
    bs = Xb.shape[0]
    indices = np.random.choice(bs, bs // 2, replace=False)
    x_da = Xb[indices]

    # apply rotation to the input batch
    rotate_90 = x_da[:, ::-1, :].transpose(0, 1, 2)
    rotate_180 = rotate_90[:, :: -1, :].transpose(0, 1, 2)
    rotate_270 = rotate_180[:, :: -1, :].transpose(0, 1, 2)
    # apply flipped versions of rotated patches
    rotate_0_flipped = x_da[:, :, ::-1]
    rotate_90_flipped = rotate_90[:, :, ::-1]
    rotate_180_flipped = rotate_180[:, :, ::-1]
    rotate_270_flipped = rotate_270[:, :, ::-1]

    augmented_x = np.stack([x_da, rotate_90, rotate_180, rotate_270,
                            rotate_0_flipped,
                            rotate_90_flipped,
                            rotate_180_flipped,
                            rotate_270_flipped],
                            axis=1)

    # select random indices from computed transformations
    r_indices = np.random.randint(0, 3, size=augmented_x.shape[0])

    Xb[indices] = np.stack([augmented_x[i,
                                        r_indices[i], :, :]
                            for i in range(augmented_x.shape[0])])

    return Xb, yb

#################### make the data generator threadsafe ####################

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def da_generator(x_train, y_train, batch_size):


    num_samples = int(x_train.shape[0] / batch_size) * batch_size
    while True:
        for b in range(0, num_samples, batch_size):
            x_ = x_train[b:b+batch_size]
            y_ = y_train[b:b+batch_size]
            x_, y_ = transform(x_, y_)
            yield x_, y_

@threadsafe_generator
def val_generator(x_train, y_train, batch_size):

    num_samples = int(x_train.shape[0] / batch_size) * batch_size
    while True:
        for b in range(0, num_samples, batch_size):
            x_ = x_train[b:b+batch_size]
            y_ = y_train[b:b+batch_size]
            x_, y_ = transform(x_, y_)
            yield x_, y_



# da_generator(
#         x_train_, y_train_
#


