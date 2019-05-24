import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.fftpack import fft

normal = np.load('normal.npy')
abnormal = np.load('abnormal.npy')
fastRun = np.load('fastRun.npy')
slowRun = np.load('slowRun.npy')
singleFootHurt = np.load('singleFootHurt.npy')

norm = np.empty((20, 15))
abn = np.empty((20, 15))
fast = np.empty((20, 15))
slow = np.empty((20, 15))
single = np.empty((20, 15))

##data preprocessing
#mean
for i in range(20):
	norm[i, 0] = np.mean(normal[i, :, 0])
	norm[i, 1] = np.mean(normal[i, :, 1])
	norm[i, 2] = np.mean(normal[i, :, 2])
	abn[i, 0] = np.mean(abnormal[i, :, 0])
	abn[i, 1] = np.mean(abnormal[i, :, 1])
	abn[i, 2] = np.mean(abnormal[i, :, 2])
	fast[i, 0] = np.mean(fastRun[i, :, 0])
	fast[i, 1] = np.mean(fastRun[i, :, 1])
	fast[i, 2] = np.mean(fastRun[i, :, 2])
	slow[i, 0] = np.mean(slowRun[i, :, 0])
	slow[i, 1] = np.mean(slowRun[i, :, 1])
	slow[i, 2] = np.mean(slowRun[i, :, 2])
	single[i, 0] = np.mean(singleFootHurt[i, :, 0])
	single[i, 1] = np.mean(singleFootHurt[i, :, 1])
	single[i, 2] = np.mean(singleFootHurt[i, :, 2])


#standard deviation
for i in range(20):
	norm[i, 3] = np.std(normal[i, :, 0])
	norm[i, 4] = np.std(normal[i, :, 1])
	norm[i, 5] = np.std(normal[i, :, 2])
	abn[i, 3] = np.std(abnormal[i, :, 0])
	abn[i, 4] = np.std(abnormal[i, :, 1])
	abn[i, 5] = np.std(abnormal[i, :, 2])
	fast[i, 3] = np.std(fastRun[i, :, 0])
	fast[i, 4] = np.std(fastRun[i, :, 1])
	fast[i, 5] = np.std(fastRun[i, :, 2])
	slow[i, 3] = np.std(slowRun[i, :, 0])
	slow[i, 4] = np.std(slowRun[i, :, 1])
	slow[i, 5] = np.std(slowRun[i, :, 2])
	single[i, 3] = np.std(singleFootHurt[i, :, 0])
	single[i, 4] = np.std(singleFootHurt[i, :, 1])
	single[i, 5] = np.std(singleFootHurt[i, :, 2])

#energy
for i in range(20):
	norm[i, 6] = sum(abs(fft(normal[i, :, 0]))**2)/240
	norm[i, 7] = sum(abs(fft(normal[i, :, 1]))**2)/240
	norm[i, 8] = sum(abs(fft(normal[i, :, 2]))**2)/240
	abn[i, 6] = sum(abs(fft(abnormal[i, :, 0]))**2)/240
	abn[i, 7] = sum(abs(fft(abnormal[i, :, 1]))**2)/240
	abn[i, 8] = sum(abs(fft(abnormal[i, :, 2]))**2)/240
	fast[i, 6] = sum(abs(fft(fastRun[i, :, 0]))**2)/220
	fast[i, 7] = sum(abs(fft(fastRun[i, :, 1]))**2)/220
	fast[i, 8] = sum(abs(fft(fastRun[i, :, 2]))**2)/220
	slow[i, 6] = sum(abs(fft(slowRun[i, :, 0]))**2)/240
	slow[i, 7] = sum(abs(fft(slowRun[i, :, 1]))**2)/240
	slow[i, 8] = sum(abs(fft(slowRun[i, :, 2]))**2)/240
	single[i, 6] = sum(abs(fft(singleFootHurt[i, :, 0]))**2)/240
	single[i, 7] = sum(abs(fft(singleFootHurt[i, :, 1]))**2)/240
	single[i, 8] = sum(abs(fft(singleFootHurt[i, :, 2]))**2)/240

#max min average
def maxminAVG(data, time):
	Max = []
	Min = []
	for i in range(time//20):
		Max.append(max(data[i*20:(i+1)*20-1]))
		Min.append(min(data[i*20:(i+1)*20-1]))
	averageMax = sum(Max)/(time/20)
	averageMin = sum(Min)/(time/20)
	return averageMax, averageMin

for i in range(20):
	[norm[i, 9], norm[i, 12]] = maxminAVG(normal[i, :, 0], 240)
	[norm[i, 10], norm[i, 13]] = maxminAVG(normal[i, :, 1], 240)
	[norm[i, 11], norm[i, 14]] = maxminAVG(normal[i, :, 2], 240)
	[abn[i, 9], abn[i, 12]] = maxminAVG(abnormal[i, :, 0], 240)
	[abn[i, 10], abn[i, 13]] = maxminAVG(abnormal[i, :, 1], 240)
	[abn[i, 11], abn[i, 14]] = maxminAVG(abnormal[i, :, 2], 240)
	[fast[i, 9], fast[i, 12]] = maxminAVG(fastRun[i, :, 0], 220)
	[fast[i, 10], fast[i, 13]] = maxminAVG(fastRun[i, :, 1], 220)
	[fast[i, 11], fast[i, 14]] = maxminAVG(fastRun[i, :, 2], 220)
	[slow[i, 9], slow[i, 12]] = maxminAVG(slowRun[i, :, 0], 240)
	[slow[i, 10], slow[i, 13]] = maxminAVG(slowRun[i, :, 1], 240)
	[slow[i, 11], slow[i, 14]] = maxminAVG(slowRun[i, :, 2], 240)
	[single[i, 9], single[i, 12]] = maxminAVG(singleFootHurt[i, :, 0], 240)
	[single[i, 10], single[i, 13]] = maxminAVG(singleFootHurt[i, :, 1], 240)
	[single[i, 11], single[i, 14]] = maxminAVG(singleFootHurt[i, :, 2], 240)

np.save('norm.npy', norm)
np.save('abn.npy', abn)
np.save('fast.npy', fast)
np.save('single.npy', single)
np.save('slow.npy', slow)