import copy
import torch
import math
import time

EXP_MAX = 30
EXP_MIN = -30

def topk(grad, K, R):
    topk_grads = copy.deepcopy(grad)
    res = []
    for k,r in zip(topk_grads.keys(),R):
        tensor = topk_grads[k] + r
        tensor_shape = tensor.shape
        array = tensor.flatten()
        total = array.shape[0] - K
        array_abs = torch.abs(array)
        array_sort_idx = torch.argsort(array_abs)[:total]
        array[array_sort_idx] = 0
        topk_tensor = array.reshape(tensor_shape)
        res.append(tensor - topk_tensor)
        topk_grads[k] = topk_tensor
    return topk_grads, res


def initial_S(grad, C, device, Ns):
    
    '''
    C is the compression level
    '''
    
    s = []
    tensor_shapes = []
    grads = copy.deepcopy(grad)
    s_dist = torch.distributions.bernoulli.Bernoulli(0.5)
    S = []
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        x = tensor.view(1,-1) 
        # flatten tensor
        l = x.shape[1]
        # Save the orignal shape
        tensor_shapes.append(l//2)

        s_k = s_dist.sample((1,l//2)).to(device)
        temp = -1 * torch.ones((1,l//2)).to(device)
        s_k = torch.where(s_k > 0,s_k,temp)
        s.append(s_k)

        c = int(C * l//2)
        if c % (Ns) != 0:
            c += (Ns)- c%(Ns)

        permutation = torch.randperm(l//2).to(device).long()
        index = permutation[:c]
        S.append(index)
    return tensor_shapes, s, S
        

def partial_DFT(grad, sparsity, device, res, s, S, timer):
    grads = copy.deepcopy(grad)
    bern = torch.distributions.bernoulli.Bernoulli(sparsity)
    res_nxt = []
    sigma = []
    compressed_grad = []
    i = 0
    for k in grads.keys():
        tensor = grads[k]
        if tensor.ndimension()<=1:
            continue
        x = tensor.view(1,-1) 
        l = x.shape[1]
        real = x[:,:l//2]
        imag = x[:,l//2:]
        x_complex = torch.complex(real, imag).to(device) + res[i]
        x_complex.view(1,-1)

        mask = bern.sample((1, l//2)).to(device)
        start = time.process_time()
        x_sparse = x_complex * mask
        sigma_g =  torch.norm(x_sparse, p='fro')**2/l
        x_no = (x_sparse * s[i]) / math.sqrt(sigma_g) 
        z = torch.fft.fft(x_no) / math.sqrt(l//2)
        end = time.process_time()
        timer += end - start
        res_nxt.append(x_complex - x_sparse)
        compressed_grad.append(z[:,S[i]].clone())
        sigma.append(sigma_g)
        i += 1
    return compressed_grad, sigma, res_nxt, timer


def turbo_cs(Y, sigma_g, sigma_n, device, tensor_shape, s, S, iters, compressed_level,sparsity, G):
    global_grad = copy.deepcopy(G)
    k = 0
    for layer in global_grad.keys():
        if global_grad[layer].ndimension()<=1:
            continue
        N = tensor_shape[k]
        v_A_pri = 1
        temp = torch.zeros((1,N)).to(device)
        zero_complex_array = torch.complex(temp,temp).to(device)
        z_A_pri = zero_complex_array.clone()
        for iter in range(iters):
            z_A_post = zero_complex_array.clone()
            z_A_post[:,S[k]] = v_A_pri / (v_A_pri + sigma_n) * (Y[k] - z_A_pri[:,S[k]])
            z_A_post = z_A_post + z_A_pri
            v_A_post = v_A_pri - compressed_level * v_A_pri**2 / (v_A_pri + sigma_n)
            v_A_ext = 1 / (1 / v_A_post - 1 / v_A_pri)
            z_A_ext = (v_A_ext / v_A_post) * z_A_post - (v_A_ext / v_A_pri) * z_A_pri

            x_B_pri = torch.fft.ifft(z_A_ext) * math.sqrt(N)
            v_B_pri = v_A_ext

            exponent = - torch.pow(x_B_pri.abs(), 2) * 1 / sparsity / (v_B_pri * (v_B_pri + 1 / sparsity))
            exp_max = EXP_MAX * torch.ones_like(exponent).to(device)
            exp_min = EXP_MIN * torch.ones_like(exponent).to(device)
            exponent = torch.where(exponent < EXP_MAX, exponent, exp_max)
            exponent = torch.where(exponent > EXP_MIN, exponent, exp_min)
            
            den = 1 + (v_B_pri + 1 / sparsity) / v_B_pri * (1 - sparsity) / sparsity * torch.exp(exponent)
            C = 1 /  den
            VAR = 1 / sparsity * v_B_pri / (v_B_pri + 1 / sparsity) * C + torch.pow(torch.abs(1/ sparsity /(1/sparsity +v_B_pri) * x_B_pri), 2) * C * (1-C)
            v_B_post = torch.mean(VAR)
            x_B_post = (1 / sparsity) / (1 / sparsity + v_B_pri) * x_B_pri * C
            
            v_B_ext = 1 / (1 / v_B_post - 1 / v_B_pri)
            x_B_ext = (v_B_ext / v_B_post) * x_B_post - (v_B_ext / v_B_pri) * x_B_pri

            z_A_pri = torch.fft.fft(x_B_ext) / math.sqrt(N)
            v_A_pri = v_B_ext
        tensor = global_grad[layer]
        x = tensor.view(1,-1)
        l = x.shape[1]
        x_B_post = x_B_post * s[k] * math.sqrt(sigma_g[k])
        real = x_B_post.real
        imag = x_B_post.imag
        x[:,:l//2] = real
        x[:,l//2:] = imag
        tensor = x.view(tensor.shape)
        
        global_grad[layer] = tensor
        k += 1
    return global_grad

        
def all_reduce(list):
    temp = copy.deepcopy(list[0])
    for t in range(len(temp)):
        for i in range(len(list)-1):
            temp[t] += copy.deepcopy(list[i+1][t])
        temp[t]  /= len(list)
    return temp

def all_sum(list):
    temp = list[0]
    for t in range(len(temp)):
        for i in range(len(list)-1):
            temp[t] += list[i+1][t]
    return temp

def transimit(g, P, H_k, Ns, idx):
    for layer in range(len(g)):
        G = g[layer].view(Ns, -1)
        g[layer] = H_k @ P[layer][idx] @ G
    return g


            

