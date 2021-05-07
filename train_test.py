#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/27
# @Author  : Eric
# @File    : train_test.py
# @Software: PyCharm

import warnings
warnings.filterwarnings("ignore")
import math
import torch
import torch.nn.functional as F
import numpy as np


def train(model, TrainData):
    # model.train()
    # initialize attention
    diff_targets = torch.zeros(TrainData.dataset.__len__(), 1)
    predictions = torch.zeros(diff_targets.size(0), 1)

    all_attention_bin = torch.zeros(TrainData.dataset.__len__(), (args.n_hms*args.n_bins))
    all_attention_hm = torch.zeros(TrainData.dataset.__len__(), args.n_hms)

    num_batches = int(math.ceil(TrainData.dataset.__len__()/float(args.batch_size)))
    all_gene_ids = [None]*TrainData.dataset.__len__()
    per_epoch_loss = 0

    # add by eric 2020.12.16
    hidden_all = np.zeros((0, 16))  # record all the final layer hidden state along with each input symbol

    print('Training')
    for idx, Sample in enumerate(TrainData):

        start, end = (idx*args.batch_size), min((idx*args.batch_size)+args.batch_size, TrainData.dataset.__len__())

        inputs_1 = Sample['input']
        batch_diff_targets = Sample['label'].unsqueeze(1).float()
        print('input shape:', inputs_1.shape)

        optimizer.zero_grad()
        batch_predictions, beta_attention, output, h = model(inputs_1.type(dtype))
        # print('batch_predictions:', batch_predictions)
        # print('beta_attention:', beta_attention)
        # print('batch_diff_targets:', batch_diff_targets)
        # print('output size:', output.shape)
        # print('h size:', h.shape)

        # batch_predictions, batch_beta, batch_alpha = model(inputs_1.type(dtype))  # added by eric 2020.12.16

        # hidden_all = np.concatenate((hidden_all, output.view(args.n_hms, args.bin_rnn_size).cpu().data.numpy()), axis=0)  # added by eric 2020.12.16

        loss = F.binary_cross_entropy_with_logits(batch_predictions.cpu(), batch_diff_targets, reduction='mean')

        per_epoch_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # all_attention_bin[start:end] = batch_alpha.data
        # all_attention_hm[start:end] = batch_beta.data

        diff_targets[start:end, 0] = batch_diff_targets[:, 0]
        all_gene_ids[start:end] = Sample['geneID']
        batch_predictions = torch.sigmoid(batch_predictions)  # change
        predictions[start:end] = batch_predictions.data.cpu()
        # print('train predictions:', predictions)

    per_epoch_loss = per_epoch_loss/num_batches
    return predictions, diff_targets, all_attention_bin, all_attention_hm, per_epoch_loss, all_gene_ids


def test(model, ValidData, split_name):
    model.eval()

    diff_targets = torch.zeros(ValidData.dataset.__len__(), 1)
    predictions = torch.zeros(diff_targets.size(0), 1)

    all_attention_bin = torch.zeros(ValidData.dataset.__len__(), (args.n_hms*args.n_bins))
    all_attention_hm = torch.zeros(ValidData.dataset.__len__(), args.n_hms)

    num_batches = int(math.ceil(ValidData.dataset.__len__()/float(args.batch_size)))
    all_gene_ids = [None]*ValidData.dataset.__len__()
    per_epoch_loss = 0
    print(split_name)
    for idx, Sample in enumerate(ValidData):

        start, end = (idx*args.batch_size), min((idx*args.batch_size)+args.batch_size, ValidData.dataset.__len__())
        optimizer.zero_grad()

        inputs_1 = Sample['input']
        batch_diff_targets = Sample['label'].unsqueeze(1).float()

        # batch_predictions, batch_beta, batch_alpha = model(inputs_1.type(dtype))
        batch_predictions, beta_attention, _, _ = model(inputs_1.type(dtype))
        # print('batch_predictions:', batch_predictions)
        # print('beta_attention:', beta_attention)
        # print('batch_diff_targets:', batch_diff_targets)

        loss = F.binary_cross_entropy_with_logits(batch_predictions.cpu(), batch_diff_targets, reduction='mean')
        # all_attention_bin[start: end] = batch_alpha.data
        # all_attention_hm[start: end] = batch_beta.data

        diff_targets[start:end, 0] = batch_diff_targets[:, 0]
        all_gene_ids[start:end] = Sample['geneID']
        # batch_predictions = torch.sigmoid(batch_predictions)  #  added by eric 2020.12.17
        predictions[start:end] = batch_predictions.data.cpu()
        # print('test predictions:', predictions)

        per_epoch_loss += loss.item()
    per_epoch_loss = per_epoch_loss/num_batches
    return predictions, diff_targets, all_attention_bin, all_attention_hm, per_epoch_loss, all_gene_ids
