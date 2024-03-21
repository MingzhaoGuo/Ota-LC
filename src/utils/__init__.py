import copy
import torch
import numpy as np


def als_sgd_update_P(grad, q):
    p = []
    i = 0
    grads = copy.deepcopy(grad)
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        matrix = tensor.view(tensor.shape[0],-1)
        temp = matrix @ q[i]
        i += 1
        p.append(temp)
    return p

def als_sgd_update_Q(grad, p):
    q = []
    i = 0
    grads = copy.deepcopy(grad)
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        matrix = tensor.view(tensor.shape[0],-1)
        temp = torch.t(matrix) @ p[i]
        i += 1
        q.append(temp)
    return q



def inverse(x1, x2):
    for l in range(len(x1)):
        x1[l] = x1[l] @(torch.inverse(torch.t(x2[l]) @ x2[l]))
    return x1

def als_sgd(p, q, grads):
    i = 0
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        y = (p[i] @ torch.t(q[i])).view(tensor.shape)
        l = torch.norm(y - tensor, p='fro')/torch.norm(tensor, p='fro')
        grads[k] = y 
        i += 1
    return grads
