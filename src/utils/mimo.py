import copy
import torch
import numpy as np
import math

def estimate_H(Nr, Nt, grad, K, device):
    H = []
    for k in range(K):
        H_real = torch.randn(Nr, Nt).to(device) + 1
        H_img = torch.randn(Nr, Nt).to(device) + 1
        H_k=torch.complex(H_real, H_img).to(device)
        H.append(H_k)
    return H

def beamforming_init(H, device, P0, grad,dimension):
    grads = copy.deepcopy(grad)
    Nr = H[0].shape[0]
    G = torch.zeros(Nr, Nr, dtype= torch.complex64).to(device)
    for h in H:
        U, S, Vh = torch.linalg.svd(h)
        S2 = S ** 2
        S2 = torch.diag_embed(S2)
        l_min = float(torch.linalg.eigvals(S2)[-1])
        G += l_min * U @ torch.t(U)
    U_G, S_G, Vh_G = torch.linalg.svd(G)

    A = []
    B = []
    for layer in grads.keys():
        tensor = grads[layer]
        if tensor.ndimension()<=1:
            continue
        matrix = tensor.view(tensor.shape[0],-1)
        n, m = matrix.shape
        F = U_G[:,:dimension]
        gamma = -99999
        p_ = 1
        for h in H:
            pp = torch.linalg.inv(torch.t(F) @ h @ torch.t(h) @ F)
            temp_gamma = float(torch.sum(torch.linalg.eigh(pp).eigenvalues))
            if temp_gamma < 0:
                temp_gamma = abs(temp_gamma)
                if temp_gamma > gamma:
                    gamma = temp_gamma
                    p_ = -1
            else:
                if temp_gamma > gamma:
                    gamma = temp_gamma
                    p_ = 1
        gamma = 1/P0 * gamma
        A_k = math.sqrt(gamma) * F * p_
        A.append(A_k)
        B_l = []
        for h in H:
            B_k = torch.t(torch.t(A_k) @ h) @ torch.linalg.inv(torch.t(A_k) @ h @ torch.t(h) @ A_k)
            B_l.append(B_k)
        B.append(B_l)
    return A, B

def transmit(signal, B, H, idx, dimension, SNR, device):
    shape = []
    sigma = []
    sigma_n = 10**(int(-SNR/10))
    for (s,i) in zip(signal, range(len(signal))):
        # print("layer:",i)
        shape.append(s.shape)
        a, b = s.shape
        d = math.ceil(a*b/dimension)
        s = s.resize_(dimension,d)

        transmit_signal = B[i][idx] @ s

        signal[i] = copy.deepcopy(H[idx]) @ copy.deepcopy(transmit_signal)
        
        P_signal = (torch.norm(transmit_signal)**2)/(a*b)
        # P_db = 10 * torch.log10(P_signal)
        
        # noise_db = P_db - SNR
        # P_signal = 1
        P_noise  = P_signal*sigma_n
        # P_noise = 10 ** (noise_db/10)

        noise = math.sqrt(P_noise)*torch.randn_like(signal[i], dtype = torch.complex64).to(device)

        signal[i] += noise

    return signal, shape
    
def beamforming(signal, A, Shape):
    for (s,i) in zip(signal, range(len(signal))):
        s = (torch.t(A[i]) @ s)
        signal[i] = s.resize_(Shape[i])
    return signal