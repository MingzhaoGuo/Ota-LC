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

def transmit_ota(signal, B, H, idx, dimension, SNR, device):
    shape = []
    sigma_n = 10**(int(SNR/10))
    for (s,i) in zip(signal, range(len(signal))):
        shape.append(s.shape)
        a, b = s.shape
        d = math.floor(a*b/dimension)
        s = s.resize_(dimension,d)

        transmit_signal = B[i][idx] @ s

        signal[i] = copy.deepcopy(H[idx]) @ copy.deepcopy(transmit_signal)
        
        P_noise  = math.sqrt(1 / (2 * sigma_n))
        noise = P_noise*torch.randn_like(signal[i], dtype = torch.complex64).to(device)

        signal[i] += noise

    return signal, shape

def digital_transmit(signal, H, idx, dimension, SNR, device):
    shape = []
    sigma_n = 10**(int(SNR/10))
    for (s,i) in zip(signal, range(len(signal))):
        shape.append(s.shape)
        d = math.ceil(s.nelement()/dimension)
        s = s.resize_(dimension,d)

        transmit_signal =  s

        signal[i] =  copy.deepcopy(H[idx]) @ copy.deepcopy(transmit_signal)

        P_noise  = math.sqrt(1 / (2 * sigma_n))
        noise = P_noise*(torch.randn_like(signal[i]) + 1j * torch.randn_like(signal[i])).to(device)
        signal[i] += noise
        signal[i].reshape(s.shape)

    return signal, shape

def receive(signal, H, bit_width=2):
    channel_uses = 0
    receive_signal = []
    for (s_k, h) in zip(signal,H):
        channel_uses = 0
        receive_signal_k = []
        for s in s_k:
            channel_uses += s.shape[1]*bit_width
            receive_signal_k.append(torch.pinverse(h) @ s)
        receive_signal.append(receive_signal_k)
    return receive_signal, channel_uses

def float_to_bits(signal,  min_val, max_val, bit_width=2):
    quantized_signal = []
    for (s,i) in zip(signal, range(len(signal))):
        quantized = torch.round((s - min_val[i]) / (max_val[i] - min_val[i]) * (2**bit_width - 1))
        quantized_signal.append(quantized)
    return quantized_signal

def bits_to_float(bits,  min_val, max_val, bit_width=2):
    float_signal = []
    for (s,i) in zip(bits, range(len(bits))):
        float_i = (s / (2**bit_width - 1)) * (max_val[i] - min_val[i]) + min_val[i]
        float_signal.append(float_i)
    return float_signal


def qam4_modulation(quantized_signal,device):
    symbol_map = {
        (0, 0): -1-1j,
        (0, 1): -1+1j,
        (1, 0): 1-1j,
        (1, 1): 1+1j
    }
    size = [] 
    modulated_signal = []
    for s in quantized_signal:
        size.append(s.shape)
        n,l = s.shape
        print(s)
        if (n%2 == 1):
            n += 1
            s.resize_(n,l)
            s[-1,:]= 0
        bits = quantized_values.unsqueeze(-1).long()
        bits = ((bits >> torch.arange(bit_width).to(bits.device)) & 1).view(-1, bit_width)
        reshaped_bits = s.view(-1, 2)
        symbols = []
        for b in reshaped_bits:
            bit_pair = (b[0].item(), b[1].item()) 
            symbols.append(symbol_map[bit_pair])
        symbols = torch.tensor(symbols, dtype=torch.cfloat).to(device)
        modulated_signal.append(symbols)
    return modulated_signal, size

def qam4_demodulation(received_symbols,shape, device):
    symbol_map = {
        -1-1j: (0, 0),
        -1+1j: (0, 1),
        1-1j: (1, 0),
        1+1j: (1, 1)
    }
    demod_bits = []
    for (s_k,shape_k) in zip(received_symbols,shape):
        demod_bits_k = []
        for (s_k_l, i) in zip(s_k,range(len(s_k))):
            demod_bits_k_l = []
            for symbol in s_k_l:
                distances = {k: torch.abs(symbol - k) for k in symbol_map}
                closest_symbol = min(distances, key=distances.get)
                demod_bits_k_l.extend(symbol_map[closest_symbol])
            demod_k_l = torch.tensor(demod_bits_k).resize_(shape_k[i]).to(device)
            demod_bits_k.append(demod_k_l)
        demod_bits.append(demod_bits_k)
    return demod_bits



    
def beamforming(signal, A, Shape):
    channel_uses = 0;
    for (s,i) in zip(signal, range(len(signal))):
        s = (torch.t(A[i]) @ s)
        channel_uses += s.shape[1]
        signal[i] = s.resize_(Shape[i])
    return signal, channel_uses