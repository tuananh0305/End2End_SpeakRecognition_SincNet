import matplotlib.pyplot as plt

CNN_epoch = []
CNN_loss_tr = []
CNN_err_tr = []
CNN_loss_te = []
CNN_err_te = []
CNN_err_te_rate = []
CNN_res_file = open("CNN_res.res", "r") 
lines = CNN_res_file.readlines()
for line in lines:
    tokens = line.split(" ")
    CNN_epoch.append(float(tokens[1].split(",")[0]))
    CNN_loss_tr.append(float(tokens[2].split("=")[1]))
    CNN_err_tr.append(float(tokens[3].split("=")[1]))
    CNN_loss_te.append(float(tokens[4].split("=")[1]))
    CNN_err_te.append(float(tokens[5].split("=")[1]))
    CNN_err_te_rate.append(float(tokens[6].split("=")[1]))

Sinc_epoch = []
Sinc_loss_tr = []
Sinc_err_tr = []
Sinc_loss_te = []
Sinc_err_te = []
Sinc_err_te_rate = []
Sinc_res_file = open("Sinc_res.res", "r") 
lines = Sinc_res_file.readlines()
for line in lines:
    tokens = line.split(" ")
    Sinc_epoch.append(float(tokens[1].split(",")[0]))
    Sinc_loss_tr.append(float(tokens[2].split("=")[1]))
    Sinc_err_tr.append(float(tokens[3].split("=")[1]))
    Sinc_loss_te.append(float(tokens[4].split("=")[1]))
    Sinc_err_te.append(float(tokens[5].split("=")[1]))
    Sinc_err_te_rate.append(float(tokens[6].split("=")[1]))

fig, ax = plt.subplots()
# ax.plot(CNN_epoch, CNN_err_te_rate, 'b', label='CNN-based')
# ax.plot(Sinc_epoch[:25], Sinc_err_te_rate[:25], 'r', label='Sinc_mel_scale')
# ax.plot(Sinc_epoch[25:], Sinc_err_te_rate[25:], 'g', label='Sinc_linearly_spaced')
# ax.set_title('Classification Rrror (measured at sentence level)')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Error Rate')
# leg = ax.legend()

ax.plot(CNN_epoch, CNN_err_te, 'b', label='CNN-based')
ax.plot(Sinc_epoch[:25], Sinc_err_te[:25], 'r', label='Sinc_mel_scale')
ax.plot(Sinc_epoch[25:], Sinc_err_te[25:], 'g', label='Sinc_linearly_spaced')
ax.set_title('Classification Error (measured at frame level)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Error Rate')
leg = ax.legend()

plt.show()