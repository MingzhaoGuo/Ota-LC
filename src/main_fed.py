import pandas as pd
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

import math
import os


from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar,  CNNCifarRes18
from utils.averaging import FedAvg
from utils.compressor import initial_S, partial_DFT, turbo_cs, all_reduce, all_sum
from utils.mimo import estimate_H, beamforming_init, transmit, beamforming
from utils.blue import blue_transmit, blue_estimate
from utils.rlc import init_A, RLC, RLCR
from utils.lc import sca_sgd, sca_sgd_update_P_Q, error_feedback_update, sca_global, inverse, init_q_power, float2complex, complex2float
from utils.utils import set_rand_seed
from models.test import test_img


import time
import logging


logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
warm_up = 1

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
    csv_path = './outputs/csv/{name}/{model}_{data}/{C}'.format(name=args.mode, C = args.C, model = args.model, data = args.dataset)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df = pd.DataFrame(columns=['iter','train_loss', 'train_acc', 'test_loss', 'test_acc', 'time'])
    df.to_csv('./outputs/csv/{name}/{model}_{data}/{C}/{Nr}_{Nt}_{SNR}_{seed}.csv'.format(name=args.mode, Nr = args.Nr, Nt = args.Nt, SNR = args.SNRdB, C = args.C, model = args.model, data = args.dataset, seed = args.seed),index=False)
    
    set_rand_seed(args.seed)

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
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
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net_glob = CNNCifarRes18(args=args).to(args.device)
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
            if (args.mode == "ota_lc") and iter==warm_up+1:
                p, q, eta = init_q_power(grad_glob, args.C, args.device)
            
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
                    s_k,shape_s = transmit(g_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    compressed_grad.append(g_k)
                    sigma.append(sigma_k)
                    error_feedback[idx] = res_k
                    loss_locals.append(copy.deepcopy(loss))
                Y = all_reduce(compressed_grad)
                sigma_g = all_reduce(sigma)
                Y = beamforming(Y, A, shape_s)
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
                    g_k, g_shape = transmit(g_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    compressed_grad.append(g_k)
                g = all_reduce(compressed_grad)
                grad_truth = FedAvg(grad_locals)
                g_new = beamforming(g, A, g_shape)
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

                    p_k, p_shape = transmit(p_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)
                    q_k, q_shape = transmit(q_k, B, H, cur_idx, args.Nt, args.SNRdB, args.device)

                    
                    
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
                

                grad_glob, eta = sca_sgd(p, q, grad_truth, eta)

                for (idx,cur_idx) in zip(idxs_users,range(m)):
                   error_feedback[idx] = error_feedback_update(q, p ,grad_locals[cur_idx], args.num_users)

            

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
            i = "%d"%iter
            t_ = "%f"%timer
            a_train = "%f"%acc_train
            l_train = "%f"%loss_train
            a_test = "%f"%acc_test
            l_test = "%f"%loss_test
            list = [i,l_train,a_train,l_test,a_test,t_]
            data = pd.DataFrame([list])
            data.to_csv('./outputs/csv/{name}/{model}_{data}/{C}/{Nr}_{Nt}_{SNR}_{seed}.csv'.format(name=args.mode, Nr = args.Nr, Nt = args.Nt, SNR = args.SNRdB, C = args.C, model = args.model, data = args.dataset, seed = args.seed),mode= 'a',header=False,index=False)
            logger.info("average train acc: {:.2f}%".format(acc_train))
            logger.info("average train loss: {:.4f}".format(loss_train))

            logger.info("average test acc: {:.2f}%".format(acc_test))
            logger.info("average test loss: {:.4f}".format(loss_test))


