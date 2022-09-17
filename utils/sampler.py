import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def get_random_subset_sampler(len_train_data, valid_size=0.2):
    # split the dataset in train and test set
    # valid_size = 0.2
    # num_train = len(train_data)
    num_train = len_train_data

    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    # np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    print(len(train_idx), len(valid_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, val_sampler
