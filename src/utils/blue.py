import torch
import math

def blue_transmit(g_k, H_k, SNR, Nt):
    for layer in range(len(g_k)):
        G = g_k[layer].view(Nt, -1)
        g_k[layer] = math.sqrt(SNR) * H_k  @ G
    return g_k

def blue_estimate(H, Y, SNR, device, K, Nr, Nt, num_user):
    for layer in range(len(Y)):
        temp = torch.zeros(Nr, K*Nt).to(device)
        H_hat_l = torch.complex(temp, temp)
        for k in range(K):
            H_hat_l[:,k*Nt:k*Nt+Nt] = H[k][:,:]
        X_layer = torch.t(1/math.sqrt(SNR) * torch.linalg.inv(torch.t(H_hat_l) @ H_hat_l) @ torch.t(H_hat_l) @ Y[layer])
        x = torch.split(X_layer, Nt, dim = 1)
        x = torch.t(sum(x)/K)
        # x = X_layer.view(X_layer.shape[0], Nt, K)
        Y_l = x.reshape(1,-1)
        
        Y[layer] = Y_l
    return Y