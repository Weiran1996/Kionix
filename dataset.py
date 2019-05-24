import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.fftpack import fft

g = 16000

normal1 = pd.read_csv('normal1.csv', sep=';', header=11)
normal1.columns = ['time', 'g', 'ax', 'ay', 'az']

time = np.transpose(normal1['time'].values)
normal1_ax = np.transpose(normal1['ax'].values)
normal1_ay = np.transpose(normal1['ay'].values)
normal1_az = np.transpose(normal1['az'].values)

normal2 = pd.read_csv('normal2.csv', sep=';', header=11)
normal2.columns = ['time', 'g', 'ax', 'ay', 'az']

#time = np.transpose(normal2['time'].values)
normal2_ax = np.transpose(normal2['ax'].values)
normal2_ay = np.transpose(normal2['ay'].values)
normal2_az = np.transpose(normal2['az'].values)

normal1_ax = normal1_ax/g
normal1_ay = normal1_ay/g
normal1_az = normal1_az/g

normal2_ax = normal2_ax/g
normal2_ay = normal2_ay/g
normal2_az = normal2_az/g

normal = np.empty((4800, 3));
normal[0:2400-1, 0] = normal1_ax[0:2400-1]
normal[0:2400-1, 1] = normal1_ay[0:2400-1]
normal[0:2400-1, 2] = normal1_az[0:2400-1]
normal[2400:4800-1, 0] = normal2_ax[0:2400-1]
normal[2400:4800-1, 1] = normal2_ay[0:2400-1]
normal[2400:4800-1, 2] = normal2_az[0:2400-1]

normal = normal.reshape(20, 240, 3)

abnormal1 = pd.read_csv('abnormal1.csv', sep=';', header=11)
abnormal1.columns = ['time', 'g', 'ax', 'ay', 'az']

time = np.transpose(normal1['time'].values)
abnormal1_ax = np.transpose(abnormal1['ax'].values)
abnormal1_ay = np.transpose(abnormal1['ay'].values)
abnormal1_az = np.transpose(abnormal1['az'].values)

abnormal2 = pd.read_csv('abnormal2.csv', sep=';', header=11)
abnormal2.columns = ['time', 'g', 'ax', 'ay', 'az']

#time = np.transpose(abnormal2['time'].values)
abnormal2_ax = np.transpose(abnormal2['ax'].values)
abnormal2_ay = np.transpose(abnormal2['ay'].values)
abnormal2_az = np.transpose(abnormal2['az'].values)

abnormal1_ax = abnormal1_ax/g
abnormal1_ay = abnormal1_ay/g
abnormal1_az = abnormal1_az/g

abnormal2_ax = abnormal2_ax/g
abnormal2_ay = abnormal2_ay/g
abnormal2_az = abnormal2_az/g

abnormal = np.empty((4800, 3));
abnormal[0:2400-1, 0] = abnormal1_ax[0:2400-1]
abnormal[0:2400-1, 1] = abnormal1_ay[0:2400-1]
abnormal[0:2400-1, 2] = abnormal1_az[0:2400-1]
abnormal[2400:4800-1, 0] = abnormal2_ax[0:2400-1]
abnormal[2400:4800-1, 1] = abnormal2_ay[0:2400-1]
abnormal[2400:4800-1, 2] = abnormal2_az[0:2400-1]

abnormal = abnormal.reshape(20, 240, 3)

fastRun1 = pd.read_csv('fastRun1.csv', sep=';', header=11)
fastRun1.columns = ['time', 'g', 'ax', 'ay', 'az']

time = np.transpose(fastRun1['time'].values)
fastRun1_ax = np.transpose(fastRun1['ax'].values)
fastRun1_ay = np.transpose(fastRun1['ay'].values)
fastRun1_az = np.transpose(fastRun1['az'].values)

fastRun2 = pd.read_csv('fastRun2.csv', sep=';', header=11)
fastRun2.columns = ['time', 'g', 'ax', 'ay', 'az']

#time = np.transpose(fastRun2['time'].values)
fastRun2_ax = np.transpose(fastRun2['ax'].values)
fastRun2_ay = np.transpose(fastRun2['ay'].values)
fastRun2_az = np.transpose(fastRun2['az'].values)

fastRun1_ax = fastRun1_ax/g
fastRun1_ay = fastRun1_ay/g
fastRun1_az = fastRun1_az/g

fastRun2_ax = fastRun2_ax/g
fastRun2_ay = fastRun2_ay/g
fastRun2_az = fastRun2_az/g

fastRun = np.empty((4400, 3));
fastRun[0:2200-1, 0] = fastRun1_ax[0:2200-1]
fastRun[0:2200-1, 1] = fastRun1_ay[0:2200-1]
fastRun[0:2200-1, 2] = fastRun1_az[0:2200-1]
fastRun[2200:4400-1, 0] = fastRun2_ax[0:2200-1]
fastRun[2200:4400-1, 1] = fastRun2_ay[0:2200-1]
fastRun[2200:4400-1, 2] = fastRun2_az[0:2200-1]

fastRun = fastRun.reshape(20, 220, 3)

singleFootHurt1 = pd.read_csv('singleFootHurt1.csv', sep=';', header=11)
singleFootHurt1.columns = ['time', 'g', 'ax', 'ay', 'az']

time = np.transpose(singleFootHurt1['time'].values)
singleFootHurt1_ax = np.transpose(singleFootHurt1['ax'].values)
singleFootHurt1_ay = np.transpose(singleFootHurt1['ay'].values)
singleFootHurt1_az = np.transpose(singleFootHurt1['az'].values)

singleFootHurt2 = pd.read_csv('singleFootHurt2.csv', sep=';', header=11)
singleFootHurt2.columns = ['time', 'g', 'ax', 'ay', 'az']

#time = np.transpose(singleFootHurt2['time'].values)
singleFootHurt2_ax = np.transpose(singleFootHurt2['ax'].values)
singleFootHurt2_ay = np.transpose(singleFootHurt2['ay'].values)
singleFootHurt2_az = np.transpose(singleFootHurt2['az'].values)

singleFootHurt1_ax = singleFootHurt1_ax/g
singleFootHurt1_ay = singleFootHurt1_ay/g
singleFootHurt1_az = singleFootHurt1_az/g

singleFootHurt2_ax = singleFootHurt2_ax/g
singleFootHurt2_ay = singleFootHurt2_ay/g
singleFootHurt2_az = singleFootHurt2_az/g

singleFootHurt = np.empty((4800, 3));
singleFootHurt[0:2400-1, 0] = singleFootHurt1_ax[0:2400-1]
singleFootHurt[0:2400-1, 1] = singleFootHurt1_ay[0:2400-1]
singleFootHurt[0:2400-1, 2] = singleFootHurt1_az[0:2400-1]
singleFootHurt[2400:4800-1, 0] = singleFootHurt2_ax[0:2400-1]
singleFootHurt[2400:4800-1, 1] = singleFootHurt2_ay[0:2400-1]
singleFootHurt[2400:4800-1, 2] = singleFootHurt2_az[0:2400-1]

singleFootHurt = singleFootHurt.reshape(20, 240, 3)

slowRun1 = pd.read_csv('slowRun1.csv', sep=';', header=11)
slowRun1.columns = ['time', 'g', 'ax', 'ay', 'az']

#time = np.transpose(slowRun1['time'].values)
slowRun1_ax = np.transpose(slowRun1['ax'].values)
slowRun1_ay = np.transpose(slowRun1['ay'].values)
slowRun1_az = np.transpose(slowRun1['az'].values)

slowRun2 = pd.read_csv('slowRun2.csv', sep=';', header=11)
slowRun2.columns = ['time', 'g', 'ax', 'ay', 'az']

#time = np.transpose(slowRun2['time'].values)
slowRun2_ax = np.transpose(slowRun2['ax'].values)
slowRun2_ay = np.transpose(slowRun2['ay'].values)
slowRun2_az = np.transpose(slowRun2['az'].values)

slowRun1_ax = slowRun1_ax/g
slowRun1_ay = slowRun1_ay/g
slowRun1_az = slowRun1_az/g

slowRun2_ax = slowRun2_ax/g
slowRun2_ay = slowRun2_ay/g
slowRun2_az = slowRun2_az/g

slowRun = np.empty((4800, 3));
slowRun[0:2400-1, 0] = slowRun1_ax[0:2400-1]
slowRun[0:2400-1, 1] = slowRun1_ay[0:2400-1]
slowRun[0:2400-1, 2] = slowRun1_az[0:2400-1]
slowRun[2400:4800-1, 0] = slowRun2_ax[0:2400-1]
slowRun[2400:4800-1, 1] = slowRun2_ay[0:2400-1]
slowRun[2400:4800-1, 2] = slowRun2_az[0:2400-1]

slowRun = slowRun.reshape(20, 240, 3)

np.save('normal.npy', normal)
np.save('abnormal.npy', abnormal)
np.save('fastRun.npy', fastRun)
np.save('singleFootHurt.npy', singleFootHurt)
np.save('slowRun.npy', slowRun)