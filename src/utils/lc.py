import copy
import torch
import numpy as np
import math
import time

lam = 0.0005

def sca_sgd_update_P_Q(grad, res, inv_q, inv_p, timer):
    p = []
    q= []
    i = 0
    grads = copy.deepcopy(grad)
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        
        matrix = (tensor+res[i]).view(tensor.shape[0],-1) 
        start = time.process_time()
        temp_p = matrix @ inv_q[i]
        temp_q =  torch.t(matrix) @ inv_p[i]
        end = time.process_time()
        timer += end - start
        
        p.append(temp_p)
        
        q.append(temp_q) 
        
        i += 1
        
    return p, q, timer


def sca_sgd_update_Q(grad, res, inv):
    q_t = []
    i = 0
    grads = copy.deepcopy(grad)
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        matrix = (tensor+res[i]).view(tensor.shape[0],-1) 
        temp =  torch.t(matrix) @ inv[i]
        q_t.append(copy.deepcopy(temp))  
        i += 1
    return q_t

def error_feedback_update(q, p, grad_k, m):
    grads = copy.deepcopy(grad_k)
    i = 0
    res_cur = []
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        
        matrix_hat = 1/m * (p[i] @ torch.t(q[i])).view(grads[k].shape)
        res_cur.append(copy.deepcopy(tensor - matrix_hat))
        i += 1
    return res_cur


def sca_global(p,q,p_bar, q_bar, eta):
    p_new = copy.deepcopy(p)
    q_new = copy.deepcopy(q)
    for i in range(len(p)):
        p_new[i] = p[i]+ (p_bar[i] - p[i])*eta[i]
        q_new[i] = q[i]+ (q_bar[i] - q[i])*eta[i]
    return p_new, q_new


def sca_sgd(p, q, grads, eta):
    i = 0
    grad = copy.deepcopy(grads)
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        
        y = (p[i] @ torch.t(q[i])).view(tensor.shape)
        l = torch.norm(y - tensor)/torch.norm(tensor)
        matrix = (tensor).view(tensor.shape[0],-1)
        n,m = matrix.shape
        if p[i].shape[1] != min(n,m):

            grad[k] = copy.deepcopy(y)
        else:
            pass

        
        i += 1
        
        
    return grad, eta




def init_q_power(grad, rank, device):
    qs = []
    ps = []
    grads = copy.deepcopy(grad)
    total = 0
    d = 0
    mini = 99999
    eta = []
    i = 0
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        matrix = tensor.view(tensor.shape[0],-1)
        i +=1
        n,m = matrix.shape
        total += n*m
        d += n
        d += m
        mini = min(mini, min(m,n))
        eta.append(0.5)

    r = math.ceil(total * rank /d)
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        matrix = tensor.view(tensor.shape[0],-1)
        n,m = matrix.shape
        if min(m,n) < r:
            r_ = min(m,n)
            SVD = torch.linalg.svd(matrix)
            init_p = SVD.U[:, :r_]
            init_q = torch.t(torch.diag_embed(SVD.S[:r_]) @ SVD.Vh[:r_,:])
        else:
            
            SVD = torch.linalg.svd(matrix)
            init_p = SVD.U[:, :r]
            init_q = torch.t(torch.diag_embed(SVD.S[:r]) @ SVD.Vh[:r,:])
        
        ps.append(init_p)
        qs.append(init_q)
    return ps, qs, eta

def inverse(x1, device):
    inv = []
    for l in range(len(x1)):
        n, r = x1[l].shape
        temp = x1[l] @(torch.linalg.inv(torch.t(x1[l]) @ x1[l] + lam * torch.eye(r).to(device)))
        inv.append(temp)
    return inv

def float2complex(signal, device):
    complex_signal = []
    size = [] 
    i = 0
    for s in signal:
        size.append(s.shape)
        n, l = s.shape
        if (n%2 == 1):
            n += 1
            s.resize_(n,l)
            s[-1,:]= 0
            
        real = s[:n//2,:]
        imag = s[n//2:,:]
        s_complex = torch.complex(real, imag).to(device)
        complex_signal.append(s_complex)
    return complex_signal, size

def complex2float(signal, device, size):
    float_signal = []
    for (s,i) in zip(signal, range(len(signal))):
        n,l = s.shape
        x = torch.zeros(n*2,l).to(device)
        real = s.real
        imag = s.imag
        x[:n,:] = real
        x[n:,:] = imag
        x.resize_(size[i])
        float_signal.append(x)
    return float_signal










