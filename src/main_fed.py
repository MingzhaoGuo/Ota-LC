import pandas as pd
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

import math
import os


from utils.sampling import iid, mnist_noniid, non_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar,  CNNCifarResNet
from utils.averaging import FedAvg
from utils.compressor import initial_S, partial_DFT, turbo_cs, all_reduce, \
    all_sum, sparse_k, sparse_sgd, powersgd_update_P, powersgd_update_Q, orthogonalize, de_sparse_k, quantization, find_minmax, qsgd
from utils.mimo import estimate_H, beamforming_init, beamforming, transmit_ota, digital_transmit, receive, float_to_bits, bits_to_float, qam4_modulation, qam4_demodulation
from utils.blue import blue_transmit, blue_estimate
from utils.rlc import init_A, RLC, RLCR
from utils.lc import sca_sgd, sca_sgd_update_P_Q, error_feedback_update, \
    sca_global, inverse, init_q_power, float2complex, complex2float, find_minmax_power
from utils.utils import set_rand_seed
from models.test import test_img


import time
import logging


logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
warm_up = 2

log_path = './logger/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
fh = logging.FileHandler('./logger/logger_Fed_Performance_{:.4f}.log'.format(time.time()))

fh.setLevel(logging.DEBUG)


ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)


logger.addHandler(fh)
logger.addHandler(ch)

def update_lr(optimaizer, lr):
    for param in optimaizer.param_groups:
        param['lr'] = lr


logger.info('fed_train')
counter = 0

 
if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    csv_path = './outputs/csv/{name}/{model}_{data}/{iid}/{C}'.format(name=args.mode,iid=args.iid, C = args.C, model = args.model, data = args.dataset)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df = pd.DataFrame(columns=['iter','channel_uses','train_loss', 'train_acc', 'test_loss', 'test_acc', 'time'])
    df.to_csv('./outputs/csv/{name}/{model}_{data}/{iid}/{C}/{Nr}_{Nt}_{SNR}_{seed}.csv'.format(name=args.mode,iid = args.iid, Nr = args.Nr, Nt = args.Nt, SNR = args.SNRdB, C = args.C, model = args.model, data = args.dataset, seed = args.seed),index=False)
    
    set_rand_seed(args.seed)

    if args.dataset == 'mnist':
        args.num_classes = 10
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar10':
        args.num_classes = 10
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trans_cifar_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = iid(dataset_train, args.num_users)
        else:
            dict_users = non_iid(dataset_train, args.num_users, args.num_classes)
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        trans_cifar_test = transforms.Compose([transforms.ToTensor()])

        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform= trans_cifar_test)
        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar_train)
        if args.iid:
            dict_users = iid(dataset_train, args.num_users)
        else:
            dict_users = non_iid(dataset_train, args.num_users, args.num_classes)
    elif args.dataset == 'emnist':
        args.num_classes = 62
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.EMNIST('../data/emnist',split='letters',train=True, download=True, transform=transform)
        dataset_test = datasets.EMNIST('../data/emnist',split='letters',train=False, download=True, transform=transform)
        if args.iid:
            dict_users = iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model

    
    if args.model == 'cnn' and args.dataset == 'cifar10':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif (args.model == 'resnet18' or 'resnet34' or 'resnet50' or 'resnet101' or 'resnet152')  and args.dataset == 'cifar10':
        net_glob = CNNCifarResNet(args=args).to(args.device)
    elif (args.model == 'resnet18' or 'resnet34' or 'resnet50' or 'resnet101' or 'resnet152') and args.dataset == "cifar100":
        net_glob = CNNCifarResNet(args=args).to(args.device)
    elif args.model == "cnn" and args.dataset == "EMNIST":
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob.train()

    grad_glob = net_glob.state_dict()
    # training

    logger.info(args)
  
    net_total_params = sum(p.numel() for p in net_glob.parameters())
    print('| net_total_params:', net_total_params)

    error_feedback = [0 for _ in range(args.num_users)]
    sigma_n = 10**(int(-args.SNRdB/10))
    
    timer = 0
    channel_uses = 0
    for iter in range(1, args.epochs+1):
        
        grad_locals, loss_locals = [], []
        buffer_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_glob = copy.deepcopy(net_glob).state_dict()
        
        if args.mode == "sgd" or iter <= warm_up:
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                grad,  loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                grad_locals.append(copy.deepcopy(grad))
                loss_locals.append(copy.deepcopy(loss))
                  
            grad_glob = FedAvg(grad_locals)
        else:
            if (args.mode == "ota_lc" or args.mode == "ota_powersgd" or args.mode == "powersgd" or args.mode == "ota_lc_NEF") and iter==warm_up+1:
                p, q, eta = init_q_power(grad_glob, args.C, args.device)
            if (iter == warm_up+1 and args.mode != "powersgd"):
                layer_max, layer_min = find_minmax(grad_glob)
            if (args.mode == "powersgd"):
                p_max, p_min = find_minmax_power(p)
                q_max, q_min = find_minmax_power(q)

            if args.mode == "ota_cs":
                H = estimate_H(args.Nr, args.Nt, grad_glob, m, args.device)
                
                shape, s, S = initial_S(grad_glob, args.C, args.device, args.Ns)
                A, B  = beamforming_init(H, args.device, 1,grad_glob, args.dimension)
                compressed_grad = []
                sigma = []
                sigma_S = []
                if iter == warm_up+1:
                    error_feedback = []
                    for usr in range((args.num_users)):
                        res_k = []
                        for layer_shape in shape:
                            res_k_l = torch.zeros((1, layer_shape)).to(args.device)
                            res_k.append(res_k_l)
                        error_feedback.append(res_k)
                timer = 0
                for (idx,cur_idx) in zip(idxs_users,range(m)):
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    grad,  loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    grad_locals.append(copy.deepcopy(grad))
                    g_k, sigma_k, res_k, timer = partial_DFT(grad, args.C, args.device, error_feedback[idx], s, S,timer)
                    s_k,shape_s = transmit_ota(g_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    compressed_grad.append(s_k)
                    sigma.append(sigma_k)
                    error_feedback[idx] = res_k
                    loss_locals.append(copy.deepcopy(loss))
                Y = all_reduce(compressed_grad)
                sigma_g = all_reduce(sigma)
                Y,channel_uses_t = beamforming(Y, A, shape_s)
                channel_uses += channel_uses_t
                grad_truth = FedAvg(grad_locals)
                grad_glob = turbo_cs(Y, sigma_g, 0, args.device, shape, s, S, args.iter_cs, args.C, args.C, grad_truth)
            
            elif args.mode == "blue_cs":
                H = estimate_H(args.Nr, args.Nt, grad_glob, m, args.device)
                shape, s, S = initial_S(grad_glob, args.C, args.device,args.Ns)
                compressed_grad = []
                sigma = []
                if iter == warm_up+1:
                    error_feedback = []
                    for usr in range((args.num_users)):
                        res_k = []
                        for layer_shape in shape:
                            res_k_l = torch.zeros((1, layer_shape)).to(args.device)
                            res_k.append(res_k_l)
                        error_feedback.append(res_k)
                timer = 0
                for (idx,cur_idx) in zip(idxs_users,range(m)):
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    grad,  loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    grad_locals.append(copy.deepcopy(grad))
                    g_k, sigma_k, res_k, timer = partial_DFT(grad, args.C, args.device, error_feedback[idx], s, S,timer)
                    g_k = blue_transmit(g_k, H[cur_idx], args.SNRdB, args.Nt)
                    compressed_grad.append(g_k)
                    sigma.append(sigma_k)
                    error_feedback[idx] = res_k
                    loss_locals.append(copy.deepcopy(loss))
                Y = all_sum(compressed_grad)
                
                for y in Y:
                    a,b = y.shape
                    P_signal = (torch.norm(y)**2)/(a*b)
                    P_db = 10 * torch.log10(P_signal)
                    noise_db = P_db - args.SNRdB
                    P_noise = 10 ** (noise_db/10)

                    noise = math.sqrt(P_noise)*torch.randn_like(y).to(args.device)
                    
                    y = y + noise
                Y = blue_estimate(H, Y, args.SNRdB, args.device, m, args.Nr, args.Nt, args.num_users)
                grad_truth = FedAvg(grad_locals)
                sigma_g = all_reduce(sigma)
                grad_glob = turbo_cs(Y, sigma_g, 0, args.device, shape, s, S, args.iter_cs, args.C, args.C, grad_truth)
            
            elif args.mode == "ota_rlc":
                Linear_A = init_A(grad_glob, args.C, args.device)
                H = estimate_H(args.Nr, args.Nt, grad_glob, m, args.device)
                A, B  = beamforming_init(H, args.device, 1, grad_glob, args.Nt)
                compressed_grad = []
                sigma_g = []
                if iter == warm_up+1:
                    error_feedback = []
                    for usr in range((args.num_users)):
                        res_k = []
                        for temp_l in grad_glob.keys():
                            if grad_glob[temp_l].ndimension() <= 1:
                                continue
                            res_k_l = torch.zeros_like(grad_glob[temp_l]).to(args.device)
                            res_k.append(res_k_l)
                        error_feedback.append(res_k)
                timer = 0
                for (idx,cur_idx) in zip(idxs_users,range(m)):
                    local = LocalUpdate(args, dataset_train, dict_users[idx])
                    grad,  loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    loss_locals.append(copy.deepcopy(loss))
                      
                    grad_locals.append(copy.deepcopy(grad))
                    g_k, error_feedback[idx], timer = RLC(Linear_A, grad, error_feedback[idx], timer)
                    g_k, g_size = float2complex(g_k, args.device)
                    g_k, g_shape = transmit_ota(g_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    compressed_grad.append(g_k)
                g = all_reduce(compressed_grad)
                grad_truth = FedAvg(grad_locals)
                g_new,channel_uses_t = beamforming(g, A, g_shape)
                channel_uses += channel_uses_t
                g_new = complex2float(g_new, args.device, g_size)
                grad_glob = RLCR(Linear_A, g_new, grad_truth)

            elif args.mode == "ota_lc":
                H = estimate_H(args.Nr, args.Nt, grad_glob, m, args.device)
                A, B  = beamforming_init(H, args.device, 5, grad_glob, args.Nt)
                ps = []
                qs = []
                
                if iter == warm_up+1:
                    error_feedback = []
                    for usr in range((args.num_users)):
                        res_k = []
                        for temp_l in grad_glob.keys():
                            if grad_glob[temp_l].ndimension() <= 1:
                                continue
                            res_k_l = torch.zeros_like(grad_glob[temp_l]).to(args.device)
                            res_k.append(res_k_l)
                        error_feedback.append(res_k)
                sigma_p = []
                sigma_q = []

                inv_q = inverse(q, args.device)
                inv_p = inverse(p, args.device)
                P_truth = []
                Q_truth = []
                timer = 0
                for (idx,cur_idx) in zip(idxs_users,range(m)):
                    local = LocalUpdate(args, dataset_train, dict_users[idx])
                    grad,  loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    loss_locals.append(copy.deepcopy(loss))
                      
                    grad_locals.append(copy.deepcopy(grad))
                    
                    p_k, q_k, timer = sca_sgd_update_P_Q(grad, error_feedback[idx], inv_q,inv_p, timer)
                    P_truth.append(copy.deepcopy(p_k))
                    
                    Q_truth.append(copy.deepcopy(q_k))

                    p_k, p_size = float2complex(p_k, args.device)
                    q_k, q_size = float2complex(q_k, args.device)

                    p_k, p_shape = transmit_ota(p_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    q_k, q_shape = transmit_ota(q_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)

                    ps.append(p_k)
                    qs.append(q_k)
                p_n = all_reduce(ps)
                q_n = all_reduce(qs)
                P_t = all_reduce(P_truth)
                Q_t = all_reduce(Q_truth)
                p_n2, channel_uses_p = beamforming(p_n, A, p_shape)
                p_new = complex2float(p_n2, args.device, p_size)

                q_n2, channel_uses_q = beamforming(q_n, A, q_shape)
                q_new = complex2float(q_n2, args.device, q_size)
                channel_uses += channel_uses_p + channel_uses_q
                p, q = sca_global(p,q, p_new,q_new, eta)


                grad_truth = FedAvg(grad_locals)
                

                grad_glob= sca_sgd(p, q, grad_truth)

                for (idx,cur_idx) in zip(idxs_users,range(m)):
                   error_feedback[idx] = error_feedback_update(q, p ,grad_locals[cur_idx], args.num_users)

            elif args.mode == "topk":
                H = estimate_H(args.Nr, args.Nt, grad_glob, m, args.device)

                compressed_grad = []
                sigma = []
                sigma_S = []
                if iter == warm_up+1:
                    error_feedback = []
                    for usr in range((args.num_users)):
                        res_k = []
                        for temp_l in grad_glob.keys():
                            if grad_glob[temp_l].ndimension() <= 1:
                                continue
                            res_k_l = torch.zeros_like(grad_glob[temp_l]).to(args.device)
                            res_k.append(res_k_l)
                        error_feedback.append(res_k)
                timer = 0
                shape = []
                indices = []
                g_sizes = []
                for (idx,cur_idx) in zip(idxs_users,range(m)):
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    grad,  loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    grad_locals.append(copy.deepcopy(grad))
                    g_k, res_k, indices_sparse, grad_shape, timer = sparse_k(grad, args.C ,error_feedback[idx], "topk", timer)
                    
                    g_k= float_to_bits(g_k, layer_min, layer_max)
                    g_k, g_size = float2complex(g_k, args.device)
                    # g_k, g_size = qam4_modulation(g_k,args.device)
                    s_k,shape_s = digital_transmit(g_k, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    compressed_grad.append(s_k)
                    error_feedback[idx] = res_k
                    loss_locals.append(copy.deepcopy(loss))
                    indices.append(indices_sparse)
                    shape.append(grad_shape)
                    g_sizes.append(g_size)
                s_receive,channel_use =receive(compressed_grad, H)
                channel_uses += channel_use
                sparse_grad = []
                for idx in range(m):
                    # g_new = qam4_demodulation(s_receive[idx], g_sizes[idx], args.device)
                    g_new = complex2float(s_receive[idx], args.device, g_sizes[idx])
                    g_new = bits_to_float(g_new, layer_min, layer_max)
                    g_new = de_sparse_k(g_new, indices[idx], shape[idx], args.device)
                    sparse_grad.append(g_new)
                Y = all_reduce(sparse_grad)
                grad_truth = FedAvg(grad_locals)
                grad_glob = sparse_sgd(Y, grad_truth)
                layer_max, layer_min = find_minmax(grad_truth)
            
            elif args.mode == "randk":
                H = estimate_H(args.Nr, args.Nt, grad_glob, m, args.device)

                compressed_grad = []
                sigma = []
                sigma_S = []
                shape = []
                indices = []
                g_sizes = []
                if iter == warm_up+1:
                    error_feedback = []
                    for usr in range((args.num_users)):
                        res_k = []
                        for temp_l in grad_glob.keys():
                            if grad_glob[temp_l].ndimension() <= 1:
                                continue
                            res_k_l = torch.zeros_like(grad_glob[temp_l]).to(args.device)
                            res_k.append(res_k_l)
                        error_feedback.append(res_k)
                timer = 0
                for (idx,cur_idx) in zip(idxs_users,range(m)):
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    grad,  loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    grad_locals.append(copy.deepcopy(grad))
                    g_k, res_k, indices_sparse, grad_shape, timer = sparse_k(grad, args.C ,error_feedback[idx], "randk", timer)
                    g_k= float_to_bits(g_k, layer_min, layer_max)
                    g_k, g_size = float2complex(g_k, args.device)
                    # g_k, g_size = qam4_modulation(g_k,args.device)
                    s_k,shape_s = digital_transmit(g_k, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    compressed_grad.append(s_k)
                    error_feedback[idx] = res_k
                    loss_locals.append(copy.deepcopy(loss))
                    indices.append(indices_sparse)
                    shape.append(grad_shape)
                    g_sizes.append(g_size)
                s_receive,channel_use =receive(compressed_grad, H)
                channel_uses += channel_use
                sparse_grad = []
                for idx in range(m):
                    # g_new = qam4_demodulation(s_receive[idx], g_sizes[idx], args.device)
                    g_new = complex2float(s_receive[idx], args.device, g_sizes[idx],)
                    g_new = bits_to_float(g_new, layer_min, layer_max)
                    g_new = de_sparse_k(g_new, indices[idx], shape[idx], args.device)
                    sparse_grad.append(g_new)
                Y = all_reduce(sparse_grad)
                grad_truth = FedAvg(grad_locals)
                grad_glob = sparse_sgd(Y, grad_truth)
                layer_max, layer_min = find_minmax(grad_truth)
            
            elif args.mode == "ota_powersgd":
                H = estimate_H(args.Nr, args.Nt, grad_glob, m, args.device)
                A, B  = beamforming_init(H, args.device, 5, grad_glob, args.Nt)
                ps = []
                qs = []
                
                if iter == warm_up+1:
                    error_feedback = []
                    for usr in range((args.num_users)):
                        res_k = []
                        for temp_l in grad_glob.keys():
                            if grad_glob[temp_l].ndimension() <= 1:
                                continue
                            res_k_l = torch.zeros_like(grad_glob[temp_l]).to(args.device)
                            res_k.append(res_k_l)
                        error_feedback.append(res_k)
                sigma_p = []
                sigma_q = []

                P_truth = []
                Q_truth = []
                for (idx,cur_idx) in zip(idxs_users,range(m)):
                    local = LocalUpdate(args, dataset_train, dict_users[idx])
                    grad,  loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    loss_locals.append(copy.deepcopy(loss))
                    grad_locals.append(copy.deepcopy(grad))
                    p_k, timer = powersgd_update_P(grad, error_feedback[idx], q, timer)
                    P_truth.append(copy.deepcopy(p_k))
                    p_k, p_size = float2complex(p_k, args.device)
                    p_k, p_shape = transmit_ota(p_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    ps.append(p_k)
                p_n = all_reduce(ps)
                P_t = all_reduce(P_truth)
                p_n2 = beamforming(p_n, A, p_shape)
                p_new = complex2float(p_n2, args.device, p_size)
                p = orthogonalize(p_new)
                for (idx,cur_idx) in zip(idxs_users,range(m)):
                    q_k, timer = powersgd_update_Q(grad_locals[cur_idx], error_feedback[idx], p, timer)
                    Q_truth.append(copy.deepcopy(q_k))
                    q_k, q_size = float2complex(q_k, args.device)
                    q_k, q_shape = digital_transmit(q_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    qs.append(q_k)
                q_n = all_reduce(qs)
                Q_t = all_reduce(Q_truth)
                q_n2 = beamforming(q_n, A, q_shape)
                q = complex2float(q_n2, args.device, q_size)


                grad_truth = FedAvg(grad_locals)
                grad_glob= sca_sgd(p, q, grad_truth)

                for (idx,cur_idx) in zip(idxs_users,range(m)):
                   error_feedback[idx] = error_feedback_update(q, p ,grad_locals[cur_idx], args.num_users)
            
            elif args.mode == "powersgd":
                H = estimate_H(args.Nr, args.Nt, grad_glob, m, args.device)
                ps = []
                qs = []
                
                if iter == warm_up+1:
                    error_feedback = []
                    for usr in range((args.num_users)):
                        res_k = []
                        for temp_l in grad_glob.keys():
                            if grad_glob[temp_l].ndimension() <= 1:
                                continue
                            res_k_l = torch.zeros_like(grad_glob[temp_l]).to(args.device)
                            res_k.append(res_k_l)
                        error_feedback.append(res_k)
                sigma_p = []
                sigma_q = []

                P_truth = []
                Q_truth = []
                p_sizes = []
                for (idx,cur_idx) in zip(idxs_users,range(m)):
                    local = LocalUpdate(args, dataset_train, dict_users[idx])
                    grad,  loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    loss_locals.append(copy.deepcopy(loss))
                    grad_locals.append(copy.deepcopy(grad))
                    p_k, timer = powersgd_update_P(grad, error_feedback[idx], q, timer)
                    P_truth.append(copy.deepcopy(p_k))
                    p_k= float_to_bits(p_k, p_min, p_max)
                    p_k, p_size = float2complex(p_k, args.device)
                    # p_k, p_size = qam4_modulation(p_k,args.device)
                    p_sizes.append(p_size)
                    p_k, p_shape = digital_transmit(p_k, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    ps.append(p_k)
                P_t = all_reduce(P_truth)
                s_receive,channel_use_p =receive(ps, H)
                p_global = []
                for idx in range(m):
                    # p_new = qam4_demodulation(s_receive[idx], p_sizes[idx], args.device)
                    p_new = complex2float(s_receive[idx],   args.device, p_sizes[idx])
                    p_new = bits_to_float(p_new, p_min, p_max)
                    p_global.append(p_new)
                p_new = all_reduce(p_global)
                p = orthogonalize(p_new)
                q_sizes = []
                for (idx,cur_idx) in zip(idxs_users,range(m)):
                    q_k, timer = powersgd_update_Q(grad_locals[cur_idx], error_feedback[idx], p, timer)
                    Q_truth.append(copy.deepcopy(q_k))
                    
                    q_k= float_to_bits(q_k, q_min, q_max)
                    q_k, q_size = float2complex(q_k, args.device)
                    # q_k, q_size = qam4_modulation(q_k,args.device)
                    q_k, q_shape = digital_transmit(q_k, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    qs.append(q_k)
                    q_sizes.append(q_size)
                Q_t = all_reduce(Q_truth)
                s_receive,channel_use_q =receive(qs, H)
                q_global = []
                for idx in range(m):
                    # q_new = qam4_demodulation(s_receive[idx], q_sizes[idx], args.device)
                    q_new = complex2float(s_receive[idx], args.device, q_sizes[idx])
                    q_new = bits_to_float(q_new, q_min, q_max)
                    q_global.append(q_new)
                q = all_reduce(q_global)

                grad_truth = FedAvg(grad_locals)
                grad_glob= sca_sgd(p, q, grad_truth)


                for (idx,cur_idx) in zip(idxs_users,range(m)):
                   error_feedback[idx] = error_feedback_update(q, p ,grad_locals[cur_idx], args.num_users)
                channel_uses += channel_use_q + channel_use_p
                p_max, p_min = find_minmax_power(P_t)
                q_max, q_min = find_minmax_power(Q_t)

            elif args.mode == "ota_qsgd":
                H = estimate_H(args.Nr, args.Nt, grad_glob, m, args.device)
                A, B  = beamforming_init(H, args.device, 1,grad_glob, args.dimension)
                compressed_grad = []
                if iter == warm_up+1:
                    error_feedback = []
                    for usr in range((args.num_users)):
                        res_k = []
                        for temp_l in grad_glob.keys():
                            if grad_glob[temp_l].ndimension() <= 1:
                                continue
                            res_k_l = torch.zeros_like(grad_glob[temp_l]).to(args.device)
                            res_k.append(res_k_l)
                        error_feedback.append(res_k)
                timer = 0
                for (idx,cur_idx) in zip(idxs_users,range(m)):
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    grad, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    grad_locals.append(copy.deepcopy(grad))
                    loss_locals.append(copy.deepcopy(loss))
                    g_k, error_feedback[idx], intervals, timer = quantization(grad, args.C, error_feedback[idx], layer_max, layer_min,timer)
                    g_k, g_size = float2complex(g_k, args.device)
                    g_k, g_shape = transmit_ota(g_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    compressed_grad.append(g_k)
                g = all_reduce(compressed_grad)
                grad_truth = FedAvg(grad_locals)
                g_new = beamforming(g, A, g_shape)
                g_new = complex2float(g_new, args.device, g_size)
                grad_glob = qsgd(g_new, grad_truth,layer_min,intervals)
                layer_max, layer_min = find_minmax(grad_truth)
            elif args.mode == "ota_lc_NEF":
                H = estimate_H(args.Nr, args.Nt, grad_glob, m, args.device)
                A, B  = beamforming_init(H, args.device, 5, grad_glob, args.Nt)
                ps = []
                qs = []
                
                if iter == warm_up+1:
                    error_feedback = []
                    for usr in range((args.num_users)):
                        res_k = []
                        for temp_l in grad_glob.keys():
                            if grad_glob[temp_l].ndimension() <= 1:
                                continue
                            res_k_l = torch.zeros_like(grad_glob[temp_l]).to(args.device)
                            res_k.append(res_k_l)
                        error_feedback.append(res_k)
                sigma_p = []
                sigma_q = []

                inv_q = inverse(q, args.device)
                inv_p = inverse(p, args.device)
                P_truth = []
                Q_truth = []
                timer = 0
                for (idx,cur_idx) in zip(idxs_users,range(m)):
                    local = LocalUpdate(args, dataset_train, dict_users[idx])
                    grad,  loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    loss_locals.append(copy.deepcopy(loss))
                      
                    grad_locals.append(copy.deepcopy(grad))
                    
                    p_k, q_k, timer = sca_sgd_update_P_Q(grad, error_feedback[idx], inv_q,inv_p, timer)
                    P_truth.append(copy.deepcopy(p_k))
                    
                    Q_truth.append(copy.deepcopy(q_k))

                    p_k, p_size = float2complex(p_k, args.device)
                    q_k, q_size = float2complex(q_k, args.device)

                    p_k, p_shape = transmit_ota(p_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    q_k, q_shape = transmit_ota(q_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)

                    ps.append(p_k)
                    qs.append(q_k)
                p_n = all_reduce(ps)
                q_n = all_reduce(qs)
                P_t = all_reduce(P_truth)
                Q_t = all_reduce(Q_truth)
                p_n2 = beamforming(p_n, A, p_shape)
                p_new = complex2float(p_n2, args.device, p_size)

                q_n2 = beamforming(q_n, A, q_shape)
                q_new = complex2float(q_n2, args.device, q_size)

                p, q = sca_global(p,q, p_new,q_new, eta)
                grad_truth = FedAvg(grad_locals)
                grad_glob= sca_sgd(p, q, grad_truth)
            

        for k in w_glob.keys():
            if iter == 1:
                w_glob[k] = w_glob[k].long() + grad_glob[k]
            else:
                w_glob[k] = w_glob[k] + copy.deepcopy(grad_glob[k])
            
            
        net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)

        logger.info('Epoch: {}'.format(iter))
        logger.info('Train loss: {:.4f}'.format(loss_avg))

        del grad_locals, loss_locals
        
        if iter%1==0:
            acc_train, loss_train = test_img(net_glob, dataset_train, args)
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            channel_use = channel_uses / 10000
            i = "%d"%iter
            c_u = "%f"%channel_use
            t_ = "%f"%timer
            a_train = "%f"%acc_train
            l_train = "%f"%loss_train
            a_test = "%f"%acc_test
            l_test = "%f"%loss_test
            list = [i,c_u,l_train,a_train,l_test,a_test,t_]
            data = pd.DataFrame([list])
            data.to_csv('./outputs/csv/{name}/{model}_{data}/{iid}/{C}/{Nr}_{Nt}_{SNR}_{seed}.csv'.format(name=args.mode,iid = args.iid, Nr = args.Nr, Nt = args.Nt, SNR = args.SNRdB, C = args.C, model = args.model, data = args.dataset, seed = args.seed),mode= 'a',header=False,index=False)
            logger.info("average train acc: {:.2f}%".format(acc_train))
            logger.info("average train loss: {:.4f}".format(loss_train))

            logger.info("average test acc: {:.2f}%".format(acc_test))
            logger.info("average test loss: {:.4f}".format(loss_test))


