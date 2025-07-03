import numpy as np
import random
from multiprocessing import Pool
import os
import math


def data_generator(idxs, vals, batch_size, chunk_num):
    data = np.column_stack((idxs, vals.reshape(-1, 1)))
    chunk_size = len(data) // chunk_num
    while True:
        np.random.shuffle(data)
        chunk_order = np.arange(chunk_num)
        all_sorted_data = []
        for i in chunk_order:
            chunk_start = i * chunk_size
            chunk_end = (i + 1) * chunk_size if i != chunk_num - 1 else len(data)
            chunk = data[chunk_start:chunk_end]
            sort_orders = [
                (chunk[:, 0], chunk[:, 1], chunk[:, 2]),
                (chunk[:, 0], chunk[:, 2], chunk[:, 1]),
                (chunk[:, 1], chunk[:, 2], chunk[:, 0]),
                (chunk[:, 1], chunk[:, 0], chunk[:, 2]),
                (chunk[:, 2], chunk[:, 1], chunk[:, 0]),
                (chunk[:, 2], chunk[:, 0], chunk[:, 1])
            ]
            sorted_order = random.choice(sort_orders)
            sorted_chunk = chunk[np.lexsort(sorted_order)]
            all_sorted_data.extend(sorted_chunk)
        for start in range(0, len(all_sorted_data), batch_size):
            end = start + batch_size
            batch = np.array(all_sorted_data[start:end])
            batch_idxs = batch[:, :3].astype(int)
            batch_vals = batch[:, 3]
            yield [batch_vals, [batch_idxs[:, i] for i in range(3)]], batch_vals
