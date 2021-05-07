import torch
import collections
import pdb
import torch.utils.data
import csv
import json
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils
import math
from pdb import set_trace as stop
import numpy as np


def loadData(filename, windows):
    with open(filename) as fi:
        csv_reader = csv.reader(fi)
        data = list(csv_reader)
        ncols = (len(data[0]))
    fi.close()

    n_rows = len(data)
    n_genes = n_rows / windows
    n_features = ncols - 3
    print("Number of genes: %d" % n_genes)
    print("Number of entries: %d" % n_rows)
    print("Number of HMs: %d" % n_features)

    count = 0
    attr = collections.OrderedDict()

    for i in range(0, n_rows, windows):
        hm1 = torch.zeros(windows, 1)
        hm2 = torch.zeros(windows, 1)
        hm3 = torch.zeros(windows, 1)
        hm4 = torch.zeros(windows, 1)
        hm5 = torch.zeros(windows, 1)
        for w in range(0, windows):
            hm1[w][0] = int(data[i + w][2])
            hm2[w][0] = int(data[i + w][3])
            hm3[w][0] = int(data[i + w][4])
            hm4[w][0] = int(data[i + w][5])
            hm5[w][0] = int(data[i + w][6])
        geneID = str(data[i][0].split("_")[0])

        thresholded_expr = int(data[i + w][7])

        attr[count] = {
            'geneID': geneID,
            'expr': thresholded_expr,
            'hm1': hm1,
            'hm2': hm2,
            'hm3': hm3,
            'hm4': hm4,
            'hm5': hm5
        }
        count += 1

    return attr


class HMData(Dataset):
    # Dataset class for loading data
    def __init__(self, data_cell1, transform=None):
        self.c1 = data_cell1

    def __len__(self):
        return len(self.c1)

    def __getitem__(self, i):
        final_data_c1 = torch.cat((self.c1[i]['hm1'], self.c1[i]['hm2'],
                                   self.c1[i]['hm3'], self.c1[i]['hm4'],
                                   self.c1[i]['hm5']), 1)
        label = self.c1[i]['expr']
        geneID = self.c1[i]['geneID']
        sample = {'geneID': geneID, 'input': final_data_c1, 'label': label}
        # print(sample)
        return sample


def load_data(args):
    # Loads data into a 3D tensor for each of the 3 splits.

    print("==>loading train data")
    cell_train_dict1 = loadData(args.data_root + "/" + args.cell_type + "/classification/train.csv", args.n_bins)
    train_inputs = HMData(cell_train_dict1)
    # print('train_input input:', len(train_inputs['input']))
    # print('train_input geneID:', len(train_inputs['geneID']))
    # print('train_input label:', len(train_inputs['label']))

    print("==>loading valid data")
    cell_valid_dict1 = loadData(args.data_root + "/" + args.cell_type + "/classification/valid.csv", args.n_bins)
    valid_inputs = HMData(cell_valid_dict1)

    print("==>loading test data")
    cell_test_dict1 = loadData(args.data_root + "/" + args.cell_type + "/classification/test.csv", args.n_bins)
    test_inputs = HMData(cell_test_dict1)

    train = DataLoader(train_inputs, batch_size=args.batch_size, shuffle=True)  # change
    valid = DataLoader(valid_inputs, batch_size=args.batch_size, shuffle=False)
    test = DataLoader(test_inputs, batch_size=args.batch_size, shuffle=False)
    print('train len:', len(train))
    return train, valid, test


def load_data_sampling(args):
    # generate small scale dataset

    print("==>loading train data")
    cell_train_dict1 = loadData(args.data_root + "/" + args.cell_type + "/classification/train.csv", args.n_bins)
    train_inputs = HMData(cell_train_dict1)
    # print('train_inputs', len(train_inputs))

    print("==>loading valid data")
    cell_valid_dict1 = loadData(args.data_root + "/" + args.cell_type + "/classification/valid.csv", args.n_bins)
    valid_inputs = HMData(cell_valid_dict1)

    print("==>loading test data")
    cell_test_dict1 = loadData(args.data_root + "/" + args.cell_type + "/classification/test.csv", args.n_bins)
    test_inputs = HMData(cell_test_dict1)

    # split_rate = args.split
    random_seed = 33
    shuffle_dataset = True
    train_dataset_size, test_dataset_size = len(train_inputs), len(test_inputs)
    train_indices = list(range(train_dataset_size))
    test_indices = list(range(test_dataset_size))
    train_split = int(np.floor(args.split * train_dataset_size))
    test_split = int(np.floor(args.split * test_dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
    train_indices_shuffle, test_indices_shuffle = train_indices[:train_split], test_indices[:test_split]

    train_sampler = SubsetRandomSampler(train_indices_shuffle)
    test_sampler = SubsetRandomSampler(test_indices_shuffle)

    train = DataLoader(train_inputs, batch_size=args.batch_size, sampler=train_sampler)  # change
    valid = DataLoader(valid_inputs, batch_size=args.batch_size, sampler=train_sampler)
    test = DataLoader(test_inputs, batch_size=args.batch_size, sampler=test_sampler)
    print('train_split:', train_split)
    return train, valid, test, train_split, test_split



