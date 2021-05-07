#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/17
# @Author  : Eric
# @File    : fsa_generated.py
# @Software: PyCharm

import warnings
warnings.filterwarnings("ignore")
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch import cuda
import os
import csv
import numpy as np
import models_revised as Models
import evaluate
import data
import train_test
import models_revised as Models
# import seaborn as sns
# import matplotlib.pyplot as plt
from pdb import set_trace as stop

# python train.py --cell_type=Cell1 --model_name=attchrome --epochs=120 --lr=0.0001 --data_root=data/ --save_root=Results/

# parser = argparse.ArgumentParser(description='DeepDiff')
parser = argparse.ArgumentParser(description='AttentiveChrome')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--model_type', type=str, default='attchrome', help='DeepDiff variation')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout) if n_layers LSTM > 1')
parser.add_argument('--cell_type', type=str, default='Toy', help='cell type')
parser.add_argument('--save_root', type=str, default='./Results/', help='where to save')
parser.add_argument('--data_root', type=str, default='../data/', help='data location')  # change './data/' to '../data/'
parser.add_argument('--gpuid', type=int, default=0, help='CUDA gpu')
parser.add_argument('--gpu', type=int, default=0, help='CUDA gpu')
parser.add_argument('--n_hms', type=int, default=5, help='number of histone modifications')
parser.add_argument('--n_bins', type=int, default=100, help='number of bins')
parser.add_argument('--bin_rnn_size', type=int, default=32, help='bin rnn size')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--unidirectional', action='store_true', help='bidirectional/unidirectional LSTM')
parser.add_argument('--save_attention_maps', action='store_true', help='set to save validation beta attention maps')
parser.add_argument('--attentionfilename', type=str, default='beta_attention.txt', help='where to save attention maps')
parser.add_argument('--test_on_saved_model', action='store_true', help='only test on saved model')
args = parser.parse_args()

torch.manual_seed(1)
start_time = time.time()

model_name = ''
model_name += args.cell_type + '_'
model_name += args.model_type

args.bidirectional = not args.unidirectional

print('the model name: ', model_name)
args.data_root += ''
args.save_root += ''
args.dataset = args.cell_type
args.data_root = os.path.join(args.data_root)
print('loading data from: ', args.data_root)
args.save_root = os.path.join(args.save_root, args.dataset)
print('saving results into: ', args.save_root)
model_dir = os.path.join(args.save_root, model_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

attentionmapfile = model_dir + '/' + args.attentionfilename
print('==>processing data')
Train, Valid, Test = data.load_data(args)

print('==>building model')
model = Models.att_chrome(args)

if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(1)
    dtype = torch.cuda.FloatTensor
    # cuda.set_device(args.gpuid)
    model.type(dtype)
    print('Using GPU '+str(args.gpuid))
else:
    dtype = torch.FloatTensor

print(model)

if args.test_on_saved_model == False:
    print("==>initializing a new model")
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)

optimizer = opt.Adam(model.parameters(), lr=args.lr)
# optimizer = opt.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)

best_valid_loss = 10000000000
best_valid_avgAUPR = -1
best_valid_avgAUC = -1
best_test_avgAUC = -1
losses, AUCs = {r'train': [], r'eval': []}, {r'train': [], r'eval': []}
if args.test_on_saved_model == False:
    for epoch in range(0, args.epochs):
        print('---------------------------------------- Training '+str(epoch+1)+' ----------------------------------------')
        predictions, diff_targets, alpha_train, beta_train, train_loss, _ = train_test.train(model, Train)
        train_avgAUPR, train_avgAUC = evaluate.compute_metrics(predictions, diff_targets)

        predictions, diff_targets, alpha_valid, beta_valid, valid_loss, gene_ids_valid = train_test.test(model, Valid, 'Validation')
        valid_avgAUPR, valid_avgAUC = evaluate.compute_metrics(predictions, diff_targets)

        predictions, diff_targets, alpha_test, beta_test, test_loss, gene_ids_test = train_test.test(model, Test, 'Testing')
        test_avgAUPR, test_avgAUC = evaluate.compute_metrics(predictions, diff_targets)

        losses[r'train'].append(train_loss)
        AUCs[r'train'].append(train_avgAUC)
        losses[r'eval'].append(test_loss)
        AUCs[r'eval'].append(test_avgAUC)

        print('train loss:', train_loss)
        print('valid loss:', valid_loss)
        print('test loss:', test_loss)

        if valid_avgAUC >= best_valid_avgAUC:
            # save best epoch -- models converge early
            best_valid_avgAUC = valid_avgAUC
            best_test_avgAUC = test_avgAUC
            torch.save(model.cpu().state_dict(), model_dir + "/" + model_name + '_avgAUC_model.pt')
            model.type(dtype)

        print("Epoch:", epoch)
        print("train avgAUC:", train_avgAUC)
        print("valid avgAUC:", valid_avgAUC)
        print("test avgAUC:", test_avgAUC)
        print("best valid avgAUC:", best_valid_avgAUC)
        print("best test avgAUC:", best_test_avgAUC)

    print("\nFinished training")
    # plot.plot_performance(loss=losses, auc=AUCs, log_file=f'{model_dir}/' + model_name + '.pdf')
    print("Best validation avgAUC:", best_valid_avgAUC)
    print("Best test avgAUC:", best_test_avgAUC)

    # add by eric 2020.9.18 to save the final auc score on each cell type
    # with open('./Results/auc.txt', 'a') as auc_file:
    # 	auc_file.write(args.cell_type + ':' + str(best_valid_avgAUC))
    # 	auc_file.write(args.cell_type + ':' + str(best_test_avgAUC) + '\n')


    if args.save_attention_maps:
        attentionfile = open(attentionmapfile, 'w')
        attentionfilewriter = csv.writer(attentionfile)
        beta_test = beta_test.numpy()
        for i in range(len(gene_ids_test)):
            gene_attention = []
            gene_attention.append(gene_ids_test[i])
            for e in beta_test[i, :]:
                gene_attention.append(str(e))
            attentionfilewriter.writerow(gene_attention)
        attentionfile.close()


else:
    model = torch.load(model_dir+"/" + model_name + '_avgAUC_model.pt')
    predictions, diff_targets, alpha_test, beta_test, test_loss, gene_ids_test = train_test.test(Test, 'Testing')
    test_avgAUPR, test_avgAUC = evaluate.compute_metrics(predictions, diff_targets)
    print("test avgAUC:", test_avgAUC)

    if args.save_attention_maps:
        attentionfile = open(attentionmapfile, 'w')
        attentionfilewriter = csv.writer(attentionfile)
        beta_test = beta_test.numpy()
        for i in range(len(gene_ids_test)):
            gene_attention = []
            gene_attention.append(gene_ids_test[i])
            for e in beta_test[i, :]:
                gene_attention.append(str(e))
            attentionfilewriter.writerow(gene_attention)
        attentionfile.close()

end_time = time.time()
print('\n Time cost:', end_time-start_time)