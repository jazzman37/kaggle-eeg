import numpy as np
import itertools
import gzip
import pickle
import os
import pdb

def get_labels(labels_loc,labels_in_data):
    # get safe and unsafe labels
    labels = itertools.islice(open(labels_loc, encoding = 'utf-8'), 1, None)
    safe_labels = [i.replace("\n","").split(",")[0:2] for i in labels if int(i.replace("\n","").split(",")[2]) == 1]

    # get and flatten all combination of coversongs
    positive_examples = [i[0] for i in safe_labels if int(i[1]) == 1]
    positive_examples = [i for i in positive_examples if i in labels_in_data]
    positive_labels = [[0,1] for _ in positive_examples]
    # generate negative examples of an equivalent length to the positive examples list
    negative_examples = [i[0] for i in safe_labels if int(i[1]) == 0]
    negative_examples = [i for i in negative_examples if i in labels_in_data]
    negative_labels = [[1,0] for _ in negative_examples]

    x = positive_examples + negative_examples
    y = positive_labels + negative_labels
    return x,y

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data, dtype=object)
    data_size = len(data)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffled_data = np.random.permutation(data)
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def shuffle_pickles(pickle_folder, num_epochs, shuffle=True):
    """
    make a shuffled list of pickles num_epochs long to read from during training
    """
    load_order = np.array([], dtype = '<U58')
    picks = [os.path.join(pickle_folder,i) for i in os.listdir(pickle_folder)]
    for _ in range(num_epochs):
        load_order = np.concatenate((load_order,np.random.permutation(picks)), axis=0)
    return load_order

def read_from_pickles(path_to_pickles):
    '''
    function that loads a dictionary of filename : cqt matrix
    from a directory of gzipped pickles containing the chunked data
    '''
    spect_dict = {}
    #path_to_pickles = "/Users/Mike/Documents/kaggle-ecg/testpickle/"
    for file in os.listdir(path_to_pickles):
        if file.endswith('.pickle.gz'):
            with gzip.open(os.path.join(path_to_pickles,file),'rb') as f:
                temp_dict = pickle.load(f, encoding='latin-1')
                for sample in temp_dict.keys():
                    spect_dict[os.path.basename(sample)] = temp_dict[sample][0]
    return spect_dict


def cliques_to_dev_train(cliques,percent_dev):
    '''
    splits all cliques into dev/train sets so as to not have
    overlapping songs in dev/train to prevent overfitting
    '''
    dev_len = int(len(cliques)*percent_dev)
    cliques_list = list(cliques.items())
    train_cliques = dict(cliques_list[:-dev_len])
    dev_cliques = dict(cliques_list[-dev_len:])
    return train_cliques, dev_cliques

def cliques_to_test(cliques):
    '''
    Makes the cliques into a test dictionary
    '''
    cliques_list = list(cliques.items())
    test_cliques = dict(cliques_list)
    return test_cliques

def randomly_shuffle_xy_data(x,y):
    # Randomly shuffle data
    np.random.seed(420)
    if len(x) != len(y):
        raise Exception
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x,y = np.array(x), np.array(y)
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    return x_shuffled, y_shuffled

if __name__ == "__main__":
    read_from_pickles(".")
