import torch
import torch.nn as nn
import torch.optim as optim
import networkUtil
from torch.autograd import Variable
import numpy as np


def train(model, X, y, dtype, optimizer=optim.Adam, p=0.5,
		  epoch=1000, numBatch=5, lr=1e-2, verbose=False):
		
		printrate = epoch / 10
		
		epoch *= numBatch
		
		batchsize = (int)(len(X)/numBatch)
		
		
		
		X = np.split(X[:numBatch * batchsize], numBatch, axis=0)
		y = np.split(y[:numBatch * batchsize], numBatch)
		
		
		
		model.train()
		loss_fn = nn.L1Loss().type(dtype)
		optimizer = optimizer(model.parameters(), lr=lr)
		
		zeros = Variable(torch.zeros(len(y[0])).type(dtype))
		
		#print(zeros)
		for t in range(epoch):
			_t = t % numBatch
			
			  	
			X_var = Variable(torch.from_numpy(X[_t]).type(dtype))
			y_var = Variable(torch.from_numpy(y[_t]).type(dtype))
			
			scores = model(X_var)
			
			#tmp = (scores.view(-1)-y_var).abs()**p
			#print(tmp)
			#loss = loss_fn(scores, y_var)
			#loss = loss_fn((scores.view(-1)-y_var).abs()**p, zeros)
			tmp = (scores.view(-1)-y_var).abs()
			tmp[tmp>1] = tmp[tmp>1]**p
			print(tmp)
			loss = tmp.mean()
			#loss = loss.mean()
			
			if verbose and (t + 1) % printrate == 0 :
				#print(scores[25:70], y_var[25:70])
				print('t = %d/%d, loss = %.4f' % (t + 1, epoch, loss.data[0]))
				
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
