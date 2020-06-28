import os
import sys
import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import utils
from src import bilstm
import src.dataset as dset
import src.pytorch_utils as ptu
import src.chu_liu_edmonds as chu

import warnings
warnings.filterwarnings('ignore')

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
versions_dir = 'models'


def loss_decision_func(model, device, batch, prints=False):
    words, tags, lens, y = batch
    out = model.forward(words.to(device), tags.to(device), lens, device, prints=prints)
    print('out', out.shape) if prints else None
    mask = (y > model.y_pad).int()
    print('mask', mask.shape) if prints else None
    print('y', y.shape) if prints else None

    flat_out = out[mask == 1.]
    flat_y = y[mask == 1.]

    return flat_y, flat_out, mask, out, y

def predict(model, dataset, batch_size, device, decision_func=None):
    model = model.to(device)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    model = model.eval()
    y_pred = np.array([])
    y_true = np.array([])

    with torch.no_grad():
        for batch in loader:
            flat_y, flat_out, mask, out, y = loss_decision_func(model, device, batch)

            y_pred = np.append(y_pred, decision_func(out.detach().cpu(), flat_out.detach().cpu().numpy(), mask, model.y_pad))
            y_true = np.append(y_true, flat_y.detach().cpu().numpy())
    return y_pred, y_true

def main():
    train_dataset_glove = dset.DataSet('data/train.labeled', use_glove=True)
#     test_dataset_glove = dset.DataSet('data/test.labeled', train_dataset=train_dataset_glove, use_glove=True)
    comp_dataset_glove = dset.DataSet('data/comp.unlabeled', train_dataset=train_dataset_glove, tagged=False, use_glove=True)
    model_m2 = torch.load('model_m2_final.pth')

    train_dataset_no_glove = dset.DataSet('data/train.labeled')
#     test_dataset_no_glove = dset.DataSet('data/test.labeled', train_dataset=train_dataset_no_glove)
    comp_dataset_no_glove = dset.DataSet('data/comp.unlabeled', train_dataset=train_dataset_no_glove, tagged=False)
    model_m1 = torch.load('model_m1_final.pth')

#     test_pred_m1, _ = predict(model_m1,
#                               dataset=test_dataset_no_glove.dataset(train=False),
#                               batch_size=32,
#                               device=device,
#                               decision_func=chu.test_chu_liu_edmonds)

    comp_pred_m1, _ = predict(model_m1,
                              dataset=comp_dataset_no_glove.dataset(train=False),
                              batch_size=32,
                              device=device,
                              decision_func=chu.test_chu_liu_edmonds)

#     test_pred_m2, _ = predict(model_m2,
#                               dataset=test_dataset_glove.dataset(train=False),
#                               batch_size=32,
#                               device=device,
#                               decision_func=chu.test_chu_liu_edmonds)

    comp_pred_m2, _ = predict(model_m2,
                              dataset=comp_dataset_glove.dataset(train=False),
                              batch_size=32,
                              device=device,
                              decision_func=chu.test_chu_liu_edmonds)

#     test_dataset_no_glove.insert_predictions(preds=test_pred_m1, name='test_m1')
#     print('m1_test_UAS:', test_dataset_no_glove.get_UAS())

#     test_dataset_glove.insert_predictions(preds=test_pred_m2, name='test_m2')
#     print('m2_test_UAS:', test_dataset_glove.get_UAS())

    comp_dataset_no_glove.insert_predictions(preds=comp_pred_m1, name='comp_m1')
    comp_dataset_glove.insert_predictions(preds=comp_pred_m2, name='comp_m2')

if __name__ == "__main__":
    main()
