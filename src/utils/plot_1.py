import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# plt.rc('font',family='Times New Roman')
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

c = 0.1
s = 1234
Nr = 8
Nt = 8
SNR = 20
data = 'cifar10'
model = 'resnet18'
iid = 'True'

sgd = pd.read_csv("./../outputs/csv/sgd/{model}_{data}/{iid}/1.0/8_8_20_{se}.csv".format(model = model, data = data, iid = iid,se = s))

lc = pd.read_csv("./../outputs/csv/ota_lc/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(model = model, data = data,iid = iid,c=c, se = s, Nr = Nr, Nt = Nt, SNR = SNR))
rlc = pd.read_csv("./../outputs/csv/ota_rlc/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(model = model, data = data,iid = iid,c=c, se = s, Nr = Nr, Nt = Nt, SNR = SNR))
cs = pd.read_csv("./../outputs/csv/ota_cs/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(model = model, data = data,iid = iid,c=c, se = s, Nr = Nr, Nt = Nt, SNR = SNR))
# blue = pd.read_csv("./../checkpoint/csv1/blue_cs/resnet18_cifar/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(c=c, se = s, Nr = Nr, Nt = Nt, SNR = SNR))
# cs = pd.read_csv("./../outputs/csv/ota_cs/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(model = model, data = data,iid = iid,c=c, se = s, Nr = Nr, Nt = Nt, SNR = SNR))
powersgd = pd.read_csv("./../outputs/csv/powersgd/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(model = model, data = data,iid = iid,c=c, se = s, Nr = Nr, Nt = Nt, SNR = SNR))
topk = pd.read_csv("./../outputs/csv/topk/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(model = model, data = data,iid = iid,c=c, se = s, Nr = Nr, Nt = Nt, SNR = SNR))
randk = pd.read_csv("./../outputs/csv/randk/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(model = model, data = data,iid = iid,c=c, se = s, Nr = Nr, Nt = Nt, SNR = SNR))
signsgd = pd.read_csv("./../outputs/csv/signsgd/{model}_{data}/{iid}/{c}/{Nr}_{Nt}_{SNR}_{se}.csv".format(model = model, data = data,iid = iid,c=c, se = s, Nr = Nr, Nt = Nt, SNR = SNR))
i_ = 4
i = 300


gd_train_loss = np.array(sgd[['train_loss']])[i_:i]
gd_train_acc = np.array(sgd[['train_acc']])[i_:i]
gd_test_loss = np.array(sgd[['test_loss']])[i_:i]
gd_test_acc = np.array(sgd[['test_acc']])[i_:i]

lc_train_loss = np.array(lc[['train_loss']])[i_:i]
lc_train_acc = np.array(lc[['train_acc']])[i_:i]
lc_test_loss = np.array(lc[['test_loss']])[i_:i]
lc_test_acc = np.array(lc[['test_acc']])[i_:i]
lc_comm = np.array(lc[['channel_uses']])[i_:i]

cs_train_loss = np.array(cs[['train_loss']])[i_:i]
cs_train_acc = np.array(cs[['train_acc']])[i_:i]
cs_test_loss = np.array(cs[['test_loss']])[i_:i]
cs_test_acc = np.array(cs[['test_acc']])[i_:i]
cs_comm = np.array(cs[['channel_uses']])[i_:i]

rlc_train_loss = np.array(rlc[['train_loss']])[i_:i]
rlc_train_acc = np.array(rlc[['train_acc']])[i_:i]
rlc_test_loss = np.array(rlc[['test_loss']])[i_:i]
rlc_test_acc = np.array(rlc[['test_acc']])[i_:i]
rlc_comm = np.array(rlc[['channel_uses']])[i_:i]

powersgd_train_loss = np.array(powersgd[['train_loss']])[i_:i]
powersgd_train_acc = np.array(powersgd[['train_acc']])[i_:i]
powersgd_test_loss = np.array(powersgd[['test_loss']])[i_:i]
powersgd_test_acc = np.array(powersgd[['test_acc']])[i_:i]
power_comm = np.array(powersgd[['channel_uses']])[i_:i]

topk_train_loss = np.array(topk[['train_loss']])[i_:i]
topk_train_acc = np.array(topk[['train_acc']])[i_:i]
topk_test_loss = np.array(topk[['test_loss']])[i_:i]
topk_test_acc = np.array(topk[['test_acc']])[i_:i]
topk_comm = np.array(topk[['channel_uses']])[i_:i]

randk_train_loss = np.array(randk[['train_loss']])[i_:i]
randk_train_acc = np.array(randk[['train_acc']])[i_:i]
randk_test_loss = np.array(randk[['test_loss']])[i_:i]
randk_test_acc = np.array(randk[['test_acc']])[i_:i]
randk_comm = np.array(randk[['channel_uses']])[i_:i]

signsgd_train_loss = np.array(signsgd[['train_loss']])[i_:31]
signsgd_train_acc = np.array(signsgd[['train_acc']])[i_:31]
signsgd_test_loss = np.array(signsgd[['test_loss']])[i_:31]
signsgd_test_acc = np.array(signsgd[['test_acc']])[i_:31]
signsgd_comm = np.array(signsgd[['channel_uses']])[i_:31]

# blue_train_loss = np.array(blue[['train_loss']])[0:i]
# blue_train_acc = np.array(blue[['train_acc']])[0:i]
# blue_test_loss = np.array(blue[['test_loss']])[0:i]
# blue_test_acc = np.array(blue[['test_acc']])[0:i]

x = np.arange(0,len(gd_train_loss),1)

# plt.subplot(2,2,1)
# plt.title("Convergence Curves, #parameter = 223"+r'$(\times 10^4)$',fontsize = 10)
plt.figure(figsize=(8, 5))

# plt.plot(x,gd_test_acc,label='SGD',c = "black",linewidth='1',linestyle="-")
plt.plot(lc_comm,lc_test_acc,label='OtA-LC',color = "blue",linewidth='1.5',linestyle="-")
plt.plot(rlc_comm,rlc_test_acc,label='OtA-RLC[13]',color = "orange",linewidth='1.5',linestyle="-")
plt.plot(cs_comm,cs_test_acc,label='OtA-CS[8]',color = "purple",linewidth='1.5',linestyle="-")
plt.plot(power_comm,powersgd_test_acc,label='PowerSGD[12]',color = "red",linewidth='1.5', linestyle="-")
plt.plot(topk_comm,topk_test_acc,label='TopK[4]',color = "teal",linewidth='1.5', linestyle="-")
plt.plot(randk_comm,randk_test_acc,label='RandK[5]',color = "salmon",linewidth='1.5', linestyle="-")
# plt.plot(signsgd_comm,signsgd_test_acc,label='SignSGD[3]',color = "green",linewidth='1.5', linestyle="-")


plt.ylabel('Test Accuracy (%)',fontsize = 12)
y_high = 95
y_ticks = np.arange(0, y_high, 10)
plt.yticks(y_ticks)
plt.tick_params(labelsize=12)
plt.xlabel(r'$N_{cu}\times 10^3$',fontsize = 12)
plt.legend(fontsize = 12)
# plt.title(r'$r$={t}'.format(t=20),fontsize = 12)
plt.title('{data}-iid'.format(data = data),fontsize = 12)
plt.savefig('./../fig/ac_{Nr}_{SNR}_{c}_{iid}_{data}.jpg'.format(c=c, Nr = Nr, SNR = SNR, iid = iid,data = data))

plt.clf()

# plt.subplot(2,2,2)
# plt.plot(x,gd_train_loss,label='SGD',c = "black",linewidth='1',linestyle="-")
plt.plot(lc_comm,lc_test_loss,label='OtA-LC',color = "blue",linewidth='1.5',linestyle="-")
plt.plot(rlc_comm,rlc_test_loss,label='OtA-RLC[13]',color = "orange",linewidth='1.5',linestyle="-")
plt.plot(cs_comm,cs_test_loss,label='OtA-CS[8]',color = "purple",linewidth='1.5',linestyle="-")
plt.plot(power_comm,powersgd_test_loss,label='PowerSGD[12]',color = "red",linewidth='1.5', linestyle="-")
plt.plot(topk_comm,topk_test_loss,label='TopK[4]',color = "teal",linewidth='1.5', linestyle="-")
plt.plot(randk_comm,randk_test_loss,label='RandK[5]',color = "salmon",linewidth='1.5', linestyle="-")
# plt.plot(signsgd_comm,signsgd_test_loss,label='SignSGD[3]',color = "green",linewidth='1.5', linestyle="-")

plt.xlabel(r'$N_{cu}\times 10^3$',fontsize = 12)
plt.ylabel('Loss',fontsize = 12)
plt.tick_params(labelsize=12)
# lg = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
# plt.legend(fontsize = 12)
plt.title('{data}-iid'.format(data = data),fontsize = 12)

# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)


plt.savefig('./../fig/loss_{Nr}_{SNR}_{c}_{iid}_{data}.jpg'.format(c=c, Nr = Nr, SNR = SNR,iid = iid,data = data),bbox_inches='tight')
# plt.savefig('./../fig/loss_{Nr}_{SNR}_{c}_1.jpg'.format(c=c, Nr = Nr, SNR = SNR))

# plt.savefig('./../fig/loss_{Nr}_{SNR}_{c}_1.jpg'.format(c=c, Nr = Nr, SNR = SNR),bbox_inches='tight',bbox_extra_artists=(lg,))
plt.clf()




# plt.plot(x,gd_train_loss,label='sgd',linewidth='1',linestyle="-")
# plt.plot(x,als_train_loss,label='OtA-ALS',linewidth='1.5',linestyle="--")
# plt.plot(x,rlc_train_loss,label='OtA-RLC',linewidth='1.5',linestyle="--")
# plt.plot(x,cs_train_loss,label='OtA-CS')
# plt.plot(x,blue_train_loss,label='Blue-CS')
# plt.title("training loss,SNR=20,Nt=2,Nr=20,Compression_rate={c}".format(c=c))
# plt.xlabel('iter')
# plt.legend()
# plt.savefig('./../fig/training_loss_20_100_{c}.jpg'.format(c=c))
# plt.clf()
