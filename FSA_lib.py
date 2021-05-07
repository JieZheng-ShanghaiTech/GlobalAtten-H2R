
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.cluster import KMeans
import os
import time
import matplotlib.pyplot as plt
from numpy import linalg as LA # for calculating norm efficiently
from scipy.sparse import lil_matrix  # for sparse matrix
import pickle
import sys

class MGUModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MGUModule, self).__init__()
        self.hidden_size = hidden_size
        self.linearfx = nn.Linear(input_size, hidden_size)
        self.linearfh = nn.Linear(hidden_size, hidden_size)
        self.linearhf = nn.Linear(hidden_size, hidden_size)
        self.linearhx = nn.Linear(input_size, hidden_size)

    def forward(self, x, h):
        seq_len = x.cpu().data.numpy().shape[0]
        y = Variable(torch.zeros([seq_len, 1, self.hidden_size]).cuda())
        for i in range(seq_len):  # go through the input sequence (one instance)
            # f = F.sigmoid(self.linearfx(x[i]) + self.linearfh(h))
            # h_hat = F.tanh(self.linearhf(torch.mul(f, h)) + self.linearhx(x[i]))
            # edited by eric 2020.12.16
            # nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
            f = torch.sigmoid(self.linearfx(x[i]) + self.linearfh(h))
            h_hat = torch.tanh(self.linearhf(torch.mul(f, h)) + self.linearhx(x[i]))

            h = torch.mul((1 - f), h) + torch.mul(f, h_hat)
            y[i] = h[-1]
        return y, h


class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, NNtype):
        super(Net, self).__init__()
        self.NNtype = NNtype
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if NNtype == 'RNN':
            self.rnn1 = nn.RNN(input_size, hidden_size, num_layers)
        elif NNtype == 'MGU':
            self.rnn1 = MGUModule(input_size, hidden_size).cuda()  # I wrote a high level class for it
        elif NNtype == 'GRU':
            self.rnn1 = nn.GRU(input_size, hidden_size, num_layers)
        elif NNtype == 'LSTM':
            self.rnn1 = nn.LSTM(input_size, hidden_size, num_layers)
        self.classifier1 = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        if self.NNtype == 'LSTM':
            h, c = h
            y, (h, c) = self.rnn1(x, (h, c))
            return self.classifier1(y[-1]), h, c, y
        else:
            y, h = self.rnn1.forward(x, h)  # h contains all hidden layers at the last time step of a sequence
            return self.classifier1(y[-1]), h, y  # y contains the last hidden layer of all the time steps

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))


class Point:
    def __init__(self, point, paragraphID, index):
        self.point = point
        self.paragraphID = paragraphID
        self.index = index  # index in the paragraph whose id is paragraphID


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(model, criterion, optimizer, num_epochs, train_data, train_label):
    #model.train()
    hidden_size = model.hidden_size
    hidden = model.init_hidden()
    cell = model.init_hidden()
    data_size = len(train_data)
    num_symbols = train_data[0][1].size
    num_class = train_label[0].size
    hidden_all = np.empty((0, hidden_size))  # record all the final layer hidden state along with each input symbol
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i in range(0, data_size):
            # get inputs:
            seq_len = len(train_data[i])
            # hidden_all_temp = Variable(torch.zeros(seq_len, hidden_size))
            inputs = torch.from_numpy(train_data[i]).view(seq_len, 1, num_symbols).float()
            labels = torch.from_numpy(train_label[i]).view(1, num_class).float()

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            model.zero_grad()

            # forward, backward, optimize
            hidden = Variable(hidden.data.cuda())

            if model.NNtype == 'LSTM':
                cell = Variable(cell.data.cuda())
                output, hidden, cell, y = model(inputs, (hidden, cell))  # the hidden here comes from the last round over the whole sequence
            else:
                output, hidden, y = model(inputs, hidden)

            # for j in range(0, seq_len):
            #     hidden = Variable(hidden.data.cuda())  # put the hidden onto GPU; note: only data onto GPU, then wrapped by Variable
            #     if model.NNtype == 'LSTM':
            #         cell = Variable(cell.data.cuda())
            #         output, hidden, cell = model(inputs[j].view(1, 1, num_symbols), (hidden, cell))
            #     else:
            #         output, hidden = model(inputs[j].view(1, 1, num_symbols), hidden)
            #     hidden_all_temp[j] = hidden[-1, :, :]

            hidden_all = np.concatenate((hidden_all, y.view(seq_len, hidden_size).cpu().data.numpy()), axis=0)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # change eric
            # running_loss += loss.data[0]
            num_ave = 100
            if i % num_ave == 0:  # print every ave mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / num_ave))
                running_loss = 0.0
    return model, hidden_all


def test(model, test_data, test_label):
    hidden_size = model.hidden_size
    data_size = len(test_label)
    num_symbols = test_data[0][1].size

    model.eval()
    hidden = model.init_hidden()
    cell = model.init_hidden()
    prediction = np.zeros(data_size, )
    hidden_all = np.zeros((0, hidden_size))  # record all the final layer hidden state along with each input symbol
    # hidden_list_for_each_sequence = [] # record all the final layer hidden state for each sequence

    numHidden = 0
    for i in range(data_size):
        numHidden += len(test_data[i])

    distMatrix = lil_matrix((numHidden, numHidden))  # sparse pre-calculated distance matrix
    simiMatrix = lil_matrix((numHidden, numHidden))  # sparse pre-calculated similarity matrix
    extraFea = np.zeros((numHidden, data_size))  # extra features to regularize the distance

    all_seq_len = 0
    for i in range(0, data_size):
        seq_len = len(test_data[i])
        # hidden_all_temp = Variable(torch.zeros(seq_len, hidden_size))
        inputs = torch.from_numpy(test_data[i]).view(seq_len, 1, num_symbols).float()
        inputs = Variable(inputs.cuda())  # put the inputs onto GPU

        hidden = Variable(hidden.data.cuda())  # put the hidden onto GPU; note: only data onto GPU, then wrapped by Variable
        if model.NNtype == 'LSTM':
            cell = Variable(cell.data.cuda())
            output, hidden, cell, y = model(inputs, (hidden, cell))
        else:
            output, hidden, y = model(inputs, hidden)

        final_layer_hidden_state = y.view(seq_len, hidden_size).cpu().data.numpy()
        hidden_all = np.concatenate((hidden_all, final_layer_hidden_state), axis=0)

        # construct the pre-distance matrix in instance i, e.g.
        # [[0,1,2,3],
        #  [1,0,1,2],
        #  [2,1,0,1],
        #  [3,2,1,0]]
        temp_dist_matrix = np.zeros((seq_len, seq_len))
        for index, row in enumerate(temp_dist_matrix):  # construct a up-triangle matrix
            row[index:len(row)] = range(len(row)-index)
        # double the up-triangle matrix and add a small value (1e-15) to the diagonal to make them different from 0
        temp_dist_matrix += temp_dist_matrix.T + np.diag(np.tile(1e-15,(seq_len,)))

        # construct the pre-similarity matrix in instance i, e.g.
        # [[4,3,2,1],
        #  [3,4,3,2],
        #  [2,3,4,3],
        #  [1,2,3,4]]
        temp_simi_matrix = np.zeros((seq_len, seq_len))
        for index, row in enumerate(temp_simi_matrix):
            row[index:len(row)] = range(len(row), index, -1)
        # double the up-triangle matrix
        temp_simi_matrix += temp_simi_matrix.T - np.diag(temp_simi_matrix.diagonal())

        # the distMatrix is a block matrix which consists of several blocks on the diagonal region
        distMatrix[all_seq_len:(all_seq_len+seq_len), all_seq_len:(all_seq_len+seq_len)] = temp_dist_matrix
        simiMatrix[all_seq_len:(all_seq_len+seq_len), all_seq_len:(all_seq_len+seq_len)] = temp_simi_matrix
        extraFea[all_seq_len:(all_seq_len+seq_len), i] = range(1, seq_len+1)

        all_seq_len += seq_len

        output = output.cpu().data.numpy()
        # print('in:', test_data[i], 'label:', test_label[i], 'out:', output)
        if output[0][1] > output[0][0]:
            prediction[i] = 1

    accuracy = sum((test_label[0:data_size][:, 1] == prediction).astype(int)) / data_size
    return accuracy, prediction, hidden_all, distMatrix, simiMatrix, extraFea


def obtain_state(model, n_clusters, hidden, pre_dist, pre_simi, cluster_method, num, rnn_type, para, extraFea):
    if cluster_method == 'kmeans':
        # ------------first clustering method from scikit learn-------------- #
        time1 = time.time()
        kmeans = KMeans(n_clusters, random_state=0).fit(hidden)
        state = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        time2 = time.time()
        print('Time consuming of clustering %d: %f s' % (n_clusters, time2 - time1))
    elif cluster_method == 'with-extra-fea':
        time1 = time.time()
        data = np.concatenate((hidden, para * extraFea), axis=1)
        kmeans = KMeans(n_clusters, random_state=0).fit(data)
        state = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        time2 = time.time()
        print('Time consuming of clustering %d: %f s' % (n_clusters, time2 - time1))
    else:
        # -------------second clustering method from zouxx----------------#
        cluster_centers, state, _ = kmedoids(hidden, n_clusters, pre_dist, pre_simi, cluster_method, num, rnn_type, para)

    result = model.classifier1(Variable(torch.from_numpy(cluster_centers[:, 0:len(hidden[0])]).float().cuda())) # to find out which cluster is the accepting state
    ind_accept = (result[:, 1] > result[:, 0]).cpu().data.numpy()
    accept_state = np.where(ind_accept==1)
    return state, accept_state


def gen_neighbour_matrix(data, state, n_clusters):
    neighbour_matrix_0 = np.zeros((n_clusters+1, n_clusters))
    neighbour_matrix_1 = np.zeros((n_clusters+1, n_clusters))
    seq_len = 0
    for i in range(0, len(data)):
        # state n_clusters is the start state, it is an external state, no state will transport to this start state
        if data[i][0, 0] == 0:
            neighbour_matrix_0[n_clusters, state[seq_len]] += 1
        elif data[i][0, 0] == 1:
            neighbour_matrix_1[n_clusters, state[seq_len]] += 1

        for j in range(len(data[i])-1):
            if data[i][j + 1][0] == 0:
                neighbour_matrix_0[state[seq_len + j], state[seq_len + j + 1]] += 1
            elif data[i][j + 1][0] == 1:
                neighbour_matrix_1[state[seq_len + j], state[seq_len + j + 1]] += 1
        seq_len = seq_len + len(data[i])

    return neighbour_matrix_0, neighbour_matrix_1


def gen_neighbour_matrix_scale(data, state, n_clusters, vocabulary, word_array): # this function scales to much more symbols instead of 0 and 1
    # 'vocabulary' is a set which includes all the distinct words in data,
    # 'word_array' is an array including arrays which contain all the separated words in an instance
    vocabulary = list(vocabulary)  # change vocabulary from set to list, thus can index it
    neighbour_matrix_dict = {}  # the dictionary records all the neighbour matrix w.r.t the corresponding word / symbol
    for _, word in enumerate(vocabulary):
        neighbour_matrix_dict[word] = np.zeros((n_clusters+1, n_clusters))
    seq_len = 0
    for i in range(len(data)):  # go through each instance (sequence) in data set
        # state n_clusters is the start state, it is an external state, no state will transport to this start state
        neighbour_matrix_dict[word_array[i][0]][n_clusters, state[seq_len]] += 1  # the first word causes the transition
                                                                                  # from start state to the state the first word belongs to
        for j in range(len(data[i])-1):
            neighbour_matrix_dict[word_array[i][j+1]][state[seq_len + j], state[seq_len + j + 1]] += 1 # the following words (except the first one)
            # causes the transition from the state corresponding to the previous word to the state corresponding to the current word
        seq_len = seq_len + len(data[i])

    return neighbour_matrix_dict


def generate_transition_matrix(matrix_a, matrix_b):
    # generate transition matrix
    transit_matrix = np.zeros((matrix_a.shape[0], 2), dtype=int)  # only two symbols in alphabet
    START = matrix_a.shape[1]  # note that the start state lies in the last row of matrix_a and matrix_b
    transit_matrix[0, 0] = matrix_a[START].argmax()  # in transition matrix, the start state lies in the first row
    transit_matrix[0, 1] = matrix_b[START].argmax()
    for i in range(matrix_a.shape[1]):
        if matrix_a[i].max() > 0:
            transit_matrix[i + 1, 0] = matrix_a[i].argmax()
    for i in range(matrix_b.shape[1]):
        if matrix_b[i].max() > 0:
            transit_matrix[i + 1, 1] = matrix_b[i].argmax()
    return transit_matrix



def generate_transition_matrix_scale(matrix_dict, vocabulary, type='biggest'):  # this function scales to much more symbols instead of 0 and 1
    # generate transition matrix
    vocabulary = list(vocabulary)
    START = matrix_dict[vocabulary[0]].shape[1]  # note that the start state lies in the last row of the neighbour matrix

    if type == 'biggest':
        transit_matrix = np.zeros((matrix_dict[vocabulary[0]].shape[0], len(vocabulary)), dtype=int)  # the alphabet is equal to the vocabulary
        for id, word in enumerate(vocabulary): # go through every word in vocabulary
            transit_matrix[0, id] = matrix_dict[word][START].argmax()
            for i in range(matrix_dict[word].shape[1]):
                if matrix_dict[word][i].max() > 0:
                    transit_matrix[i + 1, id] = matrix_dict[word][i].argmax()

    elif type == 'weight':
        transit_matrix = []
        temp = []
        for id, word in enumerate(vocabulary):  # go through every word in vocabulary
            temp.append(matrix_dict[word][START])
        transit_matrix.append(temp)
        for i in range(matrix_dict[vocabulary[0]].shape[1]):
            temp = []
            for id, word in enumerate(vocabulary):  # go through every word in vocabulary
                temp.append(matrix_dict[word][i])
            transit_matrix.append(temp)

    else:
        print('Please input \'biggest\' or \'weight\' for the last argument')
        sys.exit(1)

    return transit_matrix


def test_dfa_using_state(data, label, state, accept_state):
    seq_len = 0
    data_size = len(data)
    prediction = np.zeros((data_size,))
    for i in range(data_size):
        seq_len = seq_len + len(data[i])
        if state[seq_len - 1] in accept_state[0]:  # here we need to notice that the accepting state might be more than one
            prediction[i] = 1
    accuracy = sum((label[0:data_size, 1] == prediction).astype(int)) / data_size  # astype(int) means convert to int type
    return accuracy, prediction


def test_dfa_using_transition_matrix(data, label, transit_matrix, accept_state):
    # testing using transition matrix is worse than testing using state directly is because the the transition matrix only
    # reserve the most probable route (transition from some state to another state) rather than reserving all the state
    # use transition matrix to test
    data_size = len(data)
    prediction = np.zeros((data_size,))
    for i in range(data_size):
        sequence = data[i][:, 0]
        next_state = transit_matrix[0, int(sequence[0])]
        for _, item in enumerate(sequence):
            next_state = transit_matrix[next_state+1, int(item)]
        if next_state in accept_state:
            prediction[i] = 1
    accuracy = sum((label[0:data_size, 1] == prediction).astype(int)) / data_size  # astype(int) means convert to int type
    return accuracy, prediction


def test_dfa_using_transition_matrix_scale(label, transit_matrix, accept_state, vocabulary, word_array, type='biggest'): # 'scale' means the function is applied to real application
    # use transition matrix to test
    data_size = len(word_array)
    prediction = np.zeros((data_size,))
    vocabulary = list(vocabulary)
    n_clusters = len(transit_matrix) - 1 # n_clusters means the number of clusters without start state
    if type == 'biggest':
        for i in range(data_size):
            next_state = transit_matrix[0, int(vocabulary.index(word_array[i][0]))]
            for _, word in enumerate(word_array[i]):
                next_state = transit_matrix[next_state+1, int(vocabulary.index(word))]
            if next_state in accept_state:
                prediction[i] = 1
    elif type == 'weight':
        for i in range(data_size):
            next_state_array = transit_matrix[0][int(vocabulary.index(word_array[i][0]))]
            prob = list(next_state_array / sum(next_state_array))
            next_state = np.random.choice(n_clusters, 1, p=prob)[0] # choose 1 element from 0 to n_clusters according to p
            for _, word in enumerate(word_array[i]):
                next_state_array = transit_matrix[next_state+1][int(vocabulary.index(word))]
                prob = list(next_state_array / sum(next_state_array))
                next_state = np.random.choice(n_clusters, 1, p=prob)[0]
            if next_state in accept_state:
                prediction[i] = 1
    accuracy = sum((label[0:data_size, 1] == prediction).astype(int)) / data_size  # astype(int) means convert to int type
    return accuracy, prediction


def test_dfa_sequnce_using_trans_matrix(sequence, transit_matrix, accept_state):
    # test only one sequence using transition matrix
    next_state = transit_matrix[0, int(sequence[0])]
    for _, item in enumerate(sequence):
        next_state = transit_matrix[next_state+1, int(item)]
    if next_state in accept_state:
        return 1
    else:
        return 0


def gen_dot(matrix_a, matrix_b, accept_state, rnn_type, pic_type, num):
    path = './dot' + num
    if not os.path.exists(path):
        os.makedirs(path)
    output = open(path+'/'+rnn_type+'.dot', 'w')
    output.writelines('digraph finite_state_automata {\n')
    output.writelines('rankdir=LR;\nsize="8,5"\n')
    output.writelines('node [shape = doublecircle];\n')
    for i in range(accept_state[0].shape[0]):
        output.writelines('S_%d ' % accept_state[0][i])
    output.writelines(';\n')
    output.writelines('node [shape = circle];\n')
    output.writelines('S_%d [label = "start", shape = circle, style = filled, color = "lightgrey", width = 0.9];\n' % matrix_a.shape[1])
    # draw all the edge
    # for i in range(matrix_a.shape[0]):
    #     for j in range(matrix_a.shape[1]):
    #         if matrix_a[i,j] > 0:
    #             output.writelines('S_%d -> S_%d [label = "a"];\n' % (i, j))
    # for i in range(matrix_b.shape[0]):
    #     for j in range(matrix_b.shape[1]):
    #         if matrix_b[i, j] > 0:
    #             output.writelines('S_%d -> S_%d [label = "b"];\n' % (i, j))
    if pic_type == 'biggest':
        # only draw the edge with biggest weight
        for i in range(matrix_a.shape[0]):
            if matrix_a[i].max() > 0:
                output.writelines('S_%d -> S_%d [label = "0"];\n' % (i, matrix_a[i].argmax()))
        for i in range(matrix_b.shape[0]):
            if matrix_b[i].max() > 0:
                output.writelines('S_%d -> S_%d [label = "1"];\n' % (i, matrix_b[i].argmax()))
    elif pic_type == 'weight':
        # draw all the edge with weight percent
        for i in range(matrix_a.shape[0]):
            for j in range(matrix_a.shape[1]):
                if matrix_a[i,j] > 0:
                    output.writelines('S_%d -> S_%d [label = "0 (%.2f)"];\n' % (i, j, matrix_a[i,j]/sum(matrix_a[i])))
        for i in range(matrix_b.shape[0]):
            for j in range(matrix_b.shape[1]):
                if matrix_b[i, j] > 0:
                    output.writelines('S_%d -> S_%d [label = "1 (%.2f)"];\n' % (i, j, matrix_b[i,j]/sum(matrix_b[i])))

    output.writelines('}\n')
    output.close()


def gen_dot_scale(vocab_cluster, matrix_dict, vocabulary, vocab_embed, accept_state, rnn_type, pic_type): # too many words, so that there are too many edges in dot
    vocabulary = np.array(list(vocabulary))
    kmeans = KMeans(vocab_cluster, random_state=0).fit(vocab_embed)
    # concatenate all the data
    # data_compress = np.vstack((data[0], data[1]))
    # word_compress = np.hstack((word_array[0], word_array[1]))
    # for idx in range(2, len(data)):
    #     data_compress = np.vstack((data_compress, data[idx]))
    #     word_compress = np.hstack((word_compress, word_array[idx]))
    #
    # kmeans = KMeans(num_cluster, random_state=0).fit(data_compress)
    # distance = kmeans.transform(data_compress)

    distance = kmeans.transform(vocab_embed)
    index = np.argmin(distance, axis=0)
    closest_word_list = vocabulary[index]
    #for i in range(num_cluster):
        # distance = kmeans.transform(data_compress)[:, i]
        # index = np.argsort(distance)[::-1][:10] # 10 closest to centroid i
        # closest_word_list.append(word_compress[index[0]]) # obtain the word closest to centroid i
        # closest_index_list.append(index)

    # cluster_centers = kmeans.cluster_centers_

    path = './dot-word2vec/'
    if not os.path.exists(path):
        os.makedirs(path)
    output = open(path+rnn_type+'-vocab-cluster-'+str(vocab_cluster)+'.dot', 'w')
    output.writelines('digraph finite_state_automata {\n')
    output.writelines('rankdir=LR;\nsize="8,5"\n')
    output.writelines('node [shape = doublecircle];\n')
    for i in range(accept_state[0].shape[0]):
        output.writelines('S_%d ' % accept_state[0][i])
    output.writelines(';\n')
    output.writelines('node [shape = circle];\n')
    output.writelines('S_%d [label = "start", shape = circle, style = filled, color = "lightgrey", width = 0.9];\n' % matrix_dict[vocabulary[0]].shape[1])

    if pic_type == 'biggest':
        # only draw the edge with biggest weight
        for _, word in enumerate(closest_word_list): # go through every word in closest_word_list
            for i in range(matrix_dict[word].shape[0]):
                if matrix_dict[word][i].max() > 0:
                    output.writelines('S_%d -> S_%d [label = "%s"];\n' % (i, matrix_dict[word][i].argmax(), word))
    elif pic_type == 'weight':
        # draw all the edge with weight percent
        for _, word in enumerate(closest_word_list):
            for i in range(matrix_dict[word].shape[0]):
                for j in range(matrix_dict[word].shape[1]):
                    if matrix_dict[word][i,j] > 0:
                        output.writelines('S_%d -> S_%d [label = "%s (%.2f)"];\n' % (i, j, word, matrix_dict[word][i,j]/sum(matrix_dict[word][i])))

    # if pic_type == 'biggest':
    #     # only draw the edge with biggest weight
    #     for _, word in enumerate(vocabulary): # go through every word in vocabulary
    #         for i in range(matrix_dict[word].shape[0]):
    #             if matrix_dict[word][i].max() > 0:
    #                 output.writelines('S_%d -> S_%d [label = "%s"];\n' % (i, matrix_dict[word][i].argmax(), word))
    # elif pic_type == 'weight':
    #     # draw all the edge with weight percent
    #     for _, word in enumerate(vocabulary):
    #         for i in range(matrix_dict[word].shape[0]):
    #             for j in range(matrix_dict[word].shape[1]):
    #                 if matrix_dict[word][i,j] > 0:
    #                     output.writelines('S_%d -> S_%d [label = "%s (%.2f)"];\n' % (i, j, word, matrix_dict[word][i,j]/sum(matrix_dict[word][i])))

    output.writelines('}\n')
    output.close()


def gen_dot_shrink(transition_matrix, accept_state, rnn_type, num): # In this function, I shrink the edges

    all_index = [] # if I have 10 states, then 'all_index' have 11 elements including the start state
    for row in transition_matrix:
        word_index_list = [] # 'word_index_list' contains the index where each word transits from state 'row' to 'state_num',
        # note that the first row is the start state
        for state_num in range(transition_matrix.shape[0] - 1):
            word_index_list.append(np.argwhere(row == int(state_num)))
        all_index.append(word_index_list)

    # cluster_centers = kmeans.cluster_centers_
    path = './dot-word2vec' + str(num)
    if not os.path.exists(path):
        os.makedirs(path)
    output = open(path + '/' + rnn_type + '.dot', 'w')
    output.writelines('digraph finite_state_automata {\n')
    output.writelines('rankdir=LR;\nsize="8,5"\n')
    output.writelines('node [shape = doublecircle];\n')
    for i in range(accept_state[0].shape[0]):
        output.writelines('S_%d ' % accept_state[0][i])
    output.writelines(';\n')
    output.writelines('node [shape = circle];\n')
    output.writelines('S_%d [label = "start", shape = circle, style = filled, color = "lightgrey", width = 0.9];\n' % len(all_index))

    for i in range(len(all_index)):
        for j in range(len(all_index)-1):
            if len(all_index[i][j]) > 0: # there are edges between ith and jth state
                if i == 0:
                    output.writelines('S_%d -> S_%d [label = "word_class_start-%d"];\n' % (len(all_index), j, j))
                else:
                    output.writelines('S_%d -> S_%d [label = "word_class_%d-%d"];\n' % (i - 1, j, i - 1, j))

    output.writelines('}\n')
    output.close()
    return all_index


# k-means cluster
def kmeans(dataSet, k):
    time1 = time.time()
    numSamples = dataSet.shape[0]

    labels = np.empty((numSamples,), dtype='int32')
    loss = np.empty((numSamples,), dtype='float64')
    clusterChanged = True

    ## step 1: init centroids
    centroids = init_centroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False

        ## step 2: calculate the distance between each sample and each centroid
        # dist = np.empty((numSamples, k))
        # for j in range(k):
        #     dist[:,j] = LA.norm(dataSet - centroids[j], axis=1)
        dist = distmat(dataSet, centroids)
        minIndex = np.argmin(dist, axis=1)
        minDist = np.min(dist, axis=1)

        ## step 3: for each sample
        if True in (labels != minIndex):
            clusterChanged = True
            labels = minIndex
            loss = minDist

        ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(labels == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis=0)

    inertia = sum(loss)
    time2 = time.time()
    print('Congratulations, cluster complete! Time consuming of clustering %d: %f s' % (k, time2 - time1))
    return centroids, labels, inertia


# k-medoids cluster
def kmedoids(dataSet, k, pre_dist, pre_simi, cluster_method, num, rnn_type, para, maxiter = 50):
    time1 = time.time()
    iter = 0
    numSamples = dataSet.shape[0]

    labels = np.empty((numSamples,), dtype='int32')
    loss = np.empty((numSamples,), dtype='float64')
    clusterChanged = True

    ## step 1: init centroids
    with open('./result-word2vec' + str(num) + '/centroid_index-' + rnn_type + '.pkl', 'rb') as f:
        index_dict = pickle.load(f)
    # centroids, index = init_centroids2(dataSet, k)
    index = index_dict[k]
    centroids = dataSet[index]

    while clusterChanged:
        clusterChanged = False

        ## step 2: calculate the distance between each sample and each centroid
        # dist = np.empty((numSamples, k))
        # for j in range(k):
        #     dist[:,j] = LA.norm(dataSet - centroids[j], axis=1)
        dist = distmat(dataSet, centroids)
        # extra_dist = pre_dist[:,index]
        # extra_dist[extra_dist == 0.0] = 1000 # set the distance between points from different sequences 1000
        # dist += 0.1 * extra_dist # add the pre-calculated distance
        if cluster_method == 'with-pre-dist':
            dist += para * pre_dist[:,index]
        elif cluster_method == 'with-pre-simi':
            dist -= para * pre_simi[:,index]
        
        minIndex = np.argmin(dist, axis=1)
        minDist = np.min(dist, axis=1)

        ## step 3: for each sample
        # print('The len of labels != minIndex is %d' % len(labels != minIndex))
        if True in (labels != minIndex): # only if one point's label happens to change, the cluster changes
            clusterChanged = True
            labels = minIndex
            loss = minDist

        ## step 4: update centroids
        index = []
        for j in range(k):
            idx = np.nonzero(labels == j)[0]
            if len(idx) != 0:
                pointsInClusterj = dataSet[idx]  # find all the points in cluster j
                centroids[j, :] = np.mean(pointsInClusterj, axis=0)  # calculate the mean of points in cluster j
                distj = distmat(pointsInClusterj, centroids[j, :].reshape(1, len(centroids[j, :])))  # calculate the distance between points in cluster j and the mean
                # print('The shape of dist %d is' % j)
                # print(distj.shape)
                icenter = distj.argmin()  # find the point closest to the mean
                index.append(idx[icenter])
                centroids[j, :] = pointsInClusterj[icenter]  # regard this point as the real centroid in cluster j
            else: # some cluster has no points, then restart
                centroids, index = init_centroids2(dataSet, k)
                break

        # ## step 4: update centroids
        # for j in range(k):
        #     idx = np.nonzero(labels == j)[0]
        #     pointsInClusterj = dataSet[idx]
        #     distj = distmat(pointsInClusterj, pointsInClusterj)
        #     distj += 0.1 * pre_dist[idx,idx]
        #     distsum = np.sum(distj, axis=1)
        #     icenter = distsum.argmin()
        #     centroids[j, :] = pointsInClusterj[icenter]
        iter += 1
        if iter >= maxiter:
            print('Already 50 rounds!\n')
            break

    # inertia = sum(loss.A1) # .A1 means change matrix to array
    inertia = sum(loss)
    time2 = time.time()
    print('Time consuming of clustering %d: %f s' % (k, time2 - time1))
    return centroids, labels, inertia


# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return sum(np.power(vector2 - vector1, 2))


# init centroids with cluster centers from scikit learn
def init_centroids(X, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    centroids = np.empty((k, len(X[0])))
    index = []
    # find the points closest to the centroids which is found by scikit learn
    for idx, centroid in enumerate(kmeans.cluster_centers_):
        dist = distmat(X, centroid.reshape(1, len(centroid)))
        min_idx = dist.argmin()
        centroids[idx, :] = X[min_idx]
        index.append(min_idx)
    return centroids, index


# init centroids with random samples
def init_centroids2(X, k):
    idxs = np.random.choice(range(X.shape[0]), k, replace=False)
    return X[idxs], idxs


# calculate the square of euclidean distance between X and Y
def distmat(X, Y):
    n = len(X)
    m = len(Y)
    xx = np.sum(X * X, axis=1)
    yy = np.sum(Y * Y, axis=1)
    xy = np.dot(X, Y.T)

    return np.tile(xx, (m, 1)).T + np.tile(yy, (n, 1)) - 2 * xy  # shape is n*m

