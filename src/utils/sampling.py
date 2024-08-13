import numpy as np
import torch
from torchvision import datasets, transforms

def iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 and CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

# def non_iid(dataset, num_users, classes_size, label_split=None):
#     label = np.array(dataset.target)
#     shard_per_user = 200
#     data_split = {i: [] for i in range(num_users)}
#     label_idx_split = {}
#     for i in range(len(label)):
#         label_i = label[i].item()
#         if label_i not in label_idx_split:
#             label_idx_split[label_i] = []
#         label_idx_split[label_i].append(i)
#     shard_per_class = int(shard_per_user * num_users / classes_size)
#     for label_i in label_idx_split:
#         label_idx = label_idx_split[label_i]
#         num_leftover = len(label_idx) % shard_per_class
#         leftover = label_idx[-num_leftover:] if num_leftover > 0 else []
#         new_label_idx = np.array(label_idx[:-num_leftover]) if num_leftover > 0 else np.array(label_idx)
#         new_label_idx = new_label_idx.reshape((shard_per_class, -1)).tolist()
#         for i, leftover_label_idx in enumerate(leftover):
#             new_label_idx[i] = np.concatenate([new_label_idx[i], [leftover_label_idx]])
#         label_idx_split[label_i] = new_label_idx
#     if label_split is None:
#         label_split = list(range(classes_size)) * shard_per_class
#         label_split = torch.tensor(label_split)[torch.randperm(len(label_split))].tolist()
#         label_split = np.array(label_split).reshape((num_users, -1)).tolist()
#         for i in range(len(label_split)):
#             label_split[i] = np.unique(label_split[i]).tolist()
#     for i in range(num_users):
#         for label_i in label_split[i]:
#             idx = torch.arange(len(label_idx_split[label_i]))[torch.randperm(len(label_idx_split[label_i]))[0]].item()
#             data_split[i].extend(label_idx_split[label_i].pop(idx))
#     return data_split, label_split


def non_iid(dataset, num_users, classes_size, alpha = 100):

    train_labels = np.array(dataset.targets)
    label_distribution = np.random.dirichlet([alpha]*num_users, classes_size)
 
    class_idcs = [np.argwhere(train_labels==y).flatten()
           for y in range(classes_size)]

    client_idcs = [[] for _ in range(num_users)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]
    data_split = [np.concatenate(idcs) for idcs in client_idcs]
    
    return data_split
