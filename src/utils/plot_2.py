import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

seeds = [1234]

C = [0.01,0.05, 0.1, 0.2, 0.4]

T = [244]

Nr = 8

Nt = 8

SNR = 20

sgd = []
lc = []
cs = []
power = []
rlc = []
topk = []
randk = []
model  = 'resnet18'
data = 'cifar10'
iid = 'True'



for t in T:
    sgd_seed_acc = []
    lc_seed_acc = []
    cs_seed_acc = []
    power_seed_acc = []
    rlc_seed_acc = []
    topk_seed_acc = []
    randk_seed_acc = []
    for s in seeds:
        sgd_ = pd.read_csv("./../outputs/csv/sgd/{model}_{data}/{iid}/1.0/8_8_20_{se}.csv".format(model = model, data = data, iid = iid,se = s))
        sgd_acc = np.array(sgd_[['test_acc']])[t]
        sgd_seed_acc.append(sgd_acc)
        lc_c_acc = []
        cs_c_acc = []
        power_c_acc = []
        rlc_c_acc = []
        topk_c_acc = []
        randk_c_acc = []
        for c in C:    
            lc_ = pd.read_csv("./../outputs/csv/ota_lc/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(c=c, se = s,model = model, data = data,iid = iid, Nr = Nr, Nt = Nt, SNR = SNR))
            lc_c_acc.append(np.array(lc_[['test_acc']])[t])
            if c != 0.01:
                cs_ = pd.read_csv("./../outputs/csv/ota_cs/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(c=c, se = s,model = model, data = data,iid = iid, Nr = Nr, Nt = Nt, SNR = SNR))
                cs_c_acc.append(np.array(cs_[['test_acc']])[t])
            rlc_ = pd.read_csv("./../outputs/csv/ota_rlc/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(c=c, se = s,model = model, data = data,iid = iid, Nr = Nr, Nt = Nt, SNR = SNR))
            rlc_c_acc.append(np.array(rlc_[['test_acc']])[t])
            power_ = pd.read_csv("./../outputs/csv/ota_powersgd/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(c=c, se = s,model = model, data = data,iid = iid, Nr = Nr, Nt = Nt, SNR = SNR))
            power_c_acc.append(np.array(power_[['test_acc']])[t])
            topk_ = pd.read_csv("./../outputs/csv/topk/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(c=c, se = s,model = model, data = data,iid = iid, Nr = Nr, Nt = Nt, SNR = SNR))
            topk_c_acc.append(np.array(topk_[['test_acc']])[t])
            randk_ = pd.read_csv("./../outputs/csv/randk/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(c=c, se = s,model = model, data = data,iid = iid, Nr = Nr, Nt = Nt, SNR = SNR))
            randk_c_acc.append(np.array(randk_[['test_acc']])[t])
        lc_seed_acc.append(lc_c_acc)
        cs_seed_acc.append(cs_c_acc)
        power_seed_acc.append(power_c_acc)
        rlc_seed_acc.append(rlc_c_acc)
        topk_seed_acc.append(topk_c_acc)
        randk_seed_acc.append(randk_c_acc)
    sgd.append(sgd_seed_acc)
    lc.append(lc_seed_acc)
    cs.append(cs_seed_acc)
    power.append(power_seed_acc)
    topk.append(topk_seed_acc)
    randk.append(randk_seed_acc)
    rlc.append(rlc_seed_acc)

# als_0 = []
# als_1 = []
# als_2 = []

# rlc_0 = []
# rlc_1 = []
# rlc_2 = []
# for t in T:
#     als_seed_0 = []
#     als_seed_1 = []
#     rlc_seed_0 = []
#     rlc_seed_1 = []
#     for s in seeds:
#         als_ = pd.read_csv("./../checkpoint/csv1/w_als_1/resnet18_cifar/0.05/{Nr}_{Nt}_{SNR}_{se}.csv".format(se = s, Nr = Nr, Nt = Nt, SNR = SNR))
#         als__ = pd.read_csv("./../checkpoint/csv1/w_als_1/resnet18_cifar/0.01/{Nr}_{Nt}_{SNR}_{se}.csv".format(se = s, Nr = Nr, Nt = Nt, SNR = SNR))
#         als_acc = np.array(als_[['test_acc']])[t]
#         als_acc_1 = np.array(als__[['test_acc']])[t]
#         als_seed_0.append(als_acc)
#         als_seed_1.append(als_acc_1)

#         rlc_ = pd.read_csv("./../checkpoint/csv1/w_RLC/resnet18_cifar/0.01/{Nr}_{Nt}_{SNR}_{se}.csv".format(se = s, Nr = Nr, Nt = Nt, SNR = SNR))
#         rlc_acc = np.array(rlc_[['test_acc']])[t]
#         rlc_seed_0.append(rlc_acc)

#         # rlc__ = pd.read_csv("./../checkpoint/csv/w_RLC/resnet18_cifar/0.005/{Nr}_{Nt}_{SNR}_{se}.csv".format(se = s, Nr = Nr, Nt = Nt, SNR = SNR))
#         # rlc_acc_1 = np.array(rlc__[['test_acc']])[t]
#         # rlc_seed_1.append(rlc_acc_1)

#     als_0.append(als_seed_0)
#     als_1.append(als_seed_1)
#     rlc_0.append(rlc_seed_0)
#     # rlc_1.append(rlc_seed_1)
    


# sgd = np.array(sgd)
# sgd = np.mean(sgd, axis = 2)
# sgd = np.mean(sgd, 1)




# als_1 = np.array(als_1)
# als_1 = np.mean(als_1, axis = 2)
# als_error_1 = np.std(als_1,1)
# als_1 = np.mean(als_1, 1)

# rlc_0 = np.array(rlc_0)
# rlc_0 = np.mean(rlc_0, axis = 2)
# rlc_error_0 = np.std(rlc_0,1)
# rlc_0 = np.mean(rlc_0, 1)



lc = np.array(lc)
lc = np.mean(lc, 3)
lc_error = np.std(lc, 1)
lc= np.mean(lc, 1)

rlc = np.array(rlc)
rlc = np.mean(rlc, 3)
rlc_error = np.std(rlc, 1)
rlc = np.mean(rlc, 1)

cs = np.array(cs)
cs = np.mean(cs, 3)
cs_error = np.std(cs, 1)
cs = np.mean(cs, 1)


power = np.array(power)
power = np.mean(power, 3)
power_error = np.std(power, 1)
power = np.mean(power, 1)

topk = np.array(topk)
topk = np.mean(topk, 3)
topk_error = np.std(topk, 1)
topk = np.mean(topk, 1)

randk = np.array(randk)
randk = np.mean(randk, 3)
randk_error = np.std(randk, 1)
randk = np.mean(randk, 1)

x = [2,10,20,40,80]



plt.figure(figsize=(12, 6))
for t in range(len(T)):
    # sgd__ = np.ones(6) * sgd[t]
    # plt.plot(x,sgd__,label='sgd', linewidth = 3, color = "black", alpha = 0.4,linestyle="--")
    plt.figure(figsize=(5, 4))
    plt.axhline(y=sgd[t], c = "black", alpha = 0.8, ls = "--", lw = 1, label = "sgd")


    plt.plot(x,lc[t],label='Ota-LC',color = "blue", linewidth=1, marker = 'v')
    plt.plot(x,power[t],label='Ota_PowerSGD[12]',color = "red",linewidth=1, ls="--",marker = '*')
    plt.plot(x,rlc[t],label='OtA-RLC[13]',color = "orange", linewidth=1,ls='-.' , marker = 'o')
    plt.plot(x[1:],cs[t],label='OtA-CS[8]',color = "purple",linewidth=1,ls = ":", marker = '^')
    plt.plot(x,topk[t],label='AWGN-TopK[4]',color = "teal", linewidth=1,ls='-.' , marker = '1')
    plt.plot(x,randk[t],label='AWGN-RandK[5]',color = "salmon", linewidth=1,ls='-.' , marker = '+')

    xLim = ["0.01","0.05", "0.1", "0.2", "0.4"]
    y_low = int(cs[t][0])-5 - (int(cs[t][0])-5)%2
    y_high = 96
    y_ticks = np.arange(y_low, y_high, 4)
    plt.xticks(x,xLim)
    plt.yticks(y_ticks)
    plt.title("testing accurcy,T={t}".format(t=T[t]+1))
    plt.ylabel('Test Accuracy (%)',fontsize = 12)
    plt.xlabel(r'$r$',fontsize = 12)
    plt.title(r' $T$={t},{data}-iid'.format(t=T[t]+1,data = data),fontsize = 12)
    # plt.xlabel(r'$#$'+r'$(\times 10^4)$, T={t}'.format(t=T[t]+1),fontsize = 12)
    plt.legend(fontsize = 12)
    plt.grid(linewidth = 0.5)
    plt.savefig('./../fig/test_acc_{t}_{data}w.jpg'.format(data = data,t=T[t]+1),bbox_inches='tight')
    plt.clf()

    
    
