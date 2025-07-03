import os
import pandas as pd
from datetime import datetime
from pprint import pprint
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from model import *
from metrics import *
from data_generator import *
import math

data_folder = 'GZspeed'
shape = np.loadtxt(os.path.join(data_folder, 'tensor_shape.txt')).astype(int)
tr_idxs = np.loadtxt(os.path.join(data_folder, 'train_indices.txt')).astype(int)
tr_vals = np.loadtxt(os.path.join(data_folder, 'train_values.txt'))
te_idxs = np.loadtxt(os.path.join(data_folder, 'test_indices.txt')).astype(int)
te_vals = np.loadtxt(os.path.join(data_folder, 'test_values.txt'))
val_idxs = np.loadtxt(os.path.join(data_folder, 'val_indices.txt')).astype(int)
val_vals = np.loadtxt(os.path.join(data_folder, 'val_values.txt'))

tau = 1
lr = 1e-4
rank = 20
nc = rank
epochs = 1000
batch_size = 1024
chunk_num = 1
seed = 3
verbose = 1

set_session(device_count={"GPU": 0}, seed=seed)
model = hoctc_sp(shape, rank, nc, tau)
optim = k.optimizers.Adam(learning_rate=lr)
model.compile(optim, loss=None, metrics=["mae", mape_keras, rmse_keras])

steps_per_epoch = math.ceil(len(tr_vals) / batch_size)
train_gen = data_generator(tr_idxs, tr_vals, batch_size, chunk_num)



hists = model.fit(
    # x=[tr_vals, transform(tr_idxs)],
    # y=tr_vals,
    train_gen,
    steps_per_epoch=steps_per_epoch,
    verbose=verbose,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([val_vals, transform(val_idxs)], val_vals),
    callbacks=[k.callbacks.EarlyStopping(
        monitor="val_mae",
        patience=20,
        restore_best_weights=True),
    ],
)

tr_info = get_metrics(model, [tr_vals, transform(tr_idxs)], tr_vals)
te_info = get_metrics(model, [te_vals, transform(te_idxs)], te_vals)
pprint({'train': tr_info, 'test': te_info})

