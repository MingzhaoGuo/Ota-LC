import copy
import torch
import numpy as np
import math
from scipy.linalg import hadamard
import time



def init_A(grad, rank, device):
    grads = copy.deepcopy(grad)
    As = []
    ms = 0
    total = 0
    shapes = []
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        matrix = tensor.view(tensor.shape[0],-1)
        n,m = matrix.shape
        total += n*m
        if n*m % 32 != 0:
            matrix = matrix.view(32, -1)
        else:
            matrix = matrix.view(32, -1)
        ms += matrix.shape[1]
        shapes.append(matrix.shape)
    r = math.ceil(total*rank / ms)
    print(r)
    for shape in shapes:
        A = hadamard(shape[0])[:r,:]
        A = torch.from_numpy(A.copy()).to(device).float()
        bern = torch.distributions.bernoulli.Bernoulli(0.5)
        mask = bern.sample((1, shape[0]))
        one = -1*torch.ones((1, shape[0]))
        R = torch.where(mask > 0, mask, one)[0].float()
        R = torch.diag(R).to(device)
        A = A @ R
        As.append(A)
    return As

def RLC(A, grad, res, timer):
    compressed_grad = []
    i = 0
    grads = copy.deepcopy(grad)
    res_cur = []
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        matrix = (tensor+res[i]).view(A[i].shape[1], -1)
        start = time.process_time()
        c =  A[i] @ matrix
        matrix_hat = 1/A[i].shape[1] * torch.t(A[i]) @ c
        res_cur_temp = tensor + res[i] - matrix_hat.view(grads[k].shape)
        end = time.process_time()
        compressed_grad.append(c)
        res_cur.append(res_cur_temp)
        timer += end - start
        i+=1
    return compressed_grad, res_cur, timer

def RLCR(A, compressed_grad, grads):
    i = 0
    grads = copy.deepcopy(grads)
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        y = 1 / A[i].shape[1] * torch.t(A[i]) @ compressed_grad[i]
        Y = y.view(tensor.shape)
        grads[k] = Y
        i+=1
    return grads


