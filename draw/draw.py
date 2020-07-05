import os

import matplotlib.pyplot as plt
import numpy as np


path_train1 = '/home/panmeng/Pictures/train_log.txt'

path_validate1 = '/home/panmeng/Pictures/val_log.txt'

epoch = np.arange(0,2000)

def read_data(path):
	with open(path,'r') as f:
		train_loss = f.read()
		train_loss = train_loss.split()
		print(train_loss)
		train_loss = list(map(float,train_loss))
	return train_loss

train_loss1_05 = read_data(path_train1)

val_loss1_05 = read_data(path_validate1)


plt.plot(epoch, val_loss1_05, label='loss1_05')

plt.legend()
plt.xlabel('iterations')
plt.ylabel('train loss')
plt.grid()
plt.savefig(os.getcwd()+'/'+'draw')
plt.show()
