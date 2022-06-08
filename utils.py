import os
import os.path as osp
import sys
import time
from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
from torch import nn
import jiwer

import matplotlib.pylab as plt

def calc_wer(target, pred, ignore_indexes=[0]):
    target_chars = drop_duplicated(list(filter(lambda x: x not in ignore_indexes, map(str, list(target)))))
    pred_chars = drop_duplicated(list(filter(lambda x: x not in ignore_indexes, map(str, list(pred)))))
    target_str = ' '.join(target_chars)
    pred_str = ' '.join(pred_chars)
    error = jiwer.wer(target_str, pred_str)
    return error

def drop_duplicated(chars):
    ret_chars = [chars[0]]
    for prev, curr in zip(chars[:-1], chars[1:]):
        if prev != curr:
            ret_chars.append(curr)
    return ret_chars

def build_criterion(critic_params={}):
    criterion = {
        "ce": nn.CrossEntropyLoss(ignore_index=-1),
        "ctc": torch.nn.CTCLoss(**critic_params.get('ctc', {})),
    }
    return criterion

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    return train_list, val_list


def plot_image(image):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(image, aspect="auto", origin="lower",
                   interpolation='none')

    fig.canvas.draw()
    plt.close()

    return fig