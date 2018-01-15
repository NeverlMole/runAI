import util
import random
import os
import pickle	
import numpy as np
import torch.nn as nn
from game import Game
from agent import Agent
import torch.nn as nn
import torch.optim as optim
import networkUtil
import torch
from torch.autograd import Variable

class DeepQAgent(Agent):
 
 
	def getQNet(self, networkSize):
 		
 		net = []
 		#print('getNet')
 		for i in range(len(networkSize)-1):
 			#print(i, networkSize[i],networkSize[i+1], self.dtype)
 			net.append(nn.Linear(networkSize[i] * self.numActions,networkSize[i+1] * self.numActions))
 			net.append(nn.LeakyReLU())
 			#net.append(nn.BatchNorm1d(networkSize[i+1]))
 		net.append(nn.Linear(networkSize[-1] * self.numActions, 1))
 		
 		return nn.Sequential(*net)
 		
	def loadParam(self):
 		
		if self.renew or not os.path.isfile(self.paramfile):	return
        	
		self.qNet = torch.load(self.paramfile).type(self.dtype)
 	
	def saveParam(self):
 		
		torch.save(self.qNet.type(torch.FloatTensor), self.paramfile)
		self.qNet.type(self.dtype)
 		
	def __init__(self, **args):
		Agent.__init__(self, **args)
		
		self.networkSize = self.hparams['netsize']
		self.networkSize.insert(0, self.game.getDim())
		self.numActions = self.game.getNumActions()
		
		self.networkSize = [i for i in self.networkSize]
		
		
		if self.hparams['gpu'] and torch.cuda.is_available():
			self.dtype = torch.cuda.FloatTensor
			print('GPU is avaliable!!!')
		else:
			self.dtype = torch.FloatTensor
		
		self.qNet = self.getQNet(self.networkSize).type(self.dtype)
		
		
		self.timer = 0
		self.batchsize = self.hparams['batchsize']
		self.discount = self.hparams['discount']
		self.epsilon = self.hparams['epsilon']
		self.epoch = self.hparams['epoch']
		self.numBatch = self.hparams['numbatch']
		self.lr = self.hparams['lr']
		self.p = self.hparams['p']
		self.states = []
		self.rewards = []
		self.endings = []
						
		
	def _calQValue(self, state):
		#print(Variable(torch.from_numpy(state).type(self.dtype)))
		
		#print([p for p in self.qNet])
		#torch.save(self.qNet, 'test.pt')
		self.qNet.eval()
		return self.qNet(Variable(torch.from_numpy(state).type(self.dtype))).data[0]
		
	def getQValue(self, state, action):
		return self._calQValue(self.game.tensorize(state, action))
		
	def computeActionFromQValues(self, state):
    
		actions = self.game.getLegalActions(state)
		
		if len(actions)==0: return None
        
        #print(actions)
        
		actionsValue = [(self._calQValue(self.game.tensorize(state, action)), action) for action in actions]

		maxValue = max(actionsValue, key=lambda x:x[0])[0]
		
		maxActions = [action for value, action in actionsValue if abs(value-maxValue) < 0.001]
        
		return random.choice(maxActions)
	
	def getAction(self, state):
		
		legalActions = self.game.getLegalActions(state)
		   
		if self.mode == 'Test':
			return self.computeActionFromQValues(state)
        	
		if util.flipCoin(self.epsilon):
			return random.choice(legalActions)
        	
		return self.computeActionFromQValues(state)
		
	def update(self, state, action, reward, ending=False):
		self.states.append(self.game.tensorize(state, action))
		self.rewards.append(reward)
		self.endings.append(ending)
		self.timer += 1
	 	
		if self.timer != self.batchsize: return
	 	
		X = np.array(self.states[:-1])
		y = np.zeros(len(self.states)-1)
	 	
		tmp = self._calQValue(self.states[-1])
	 	
		for i in range(len(self.states)-2,-1,-1):
			if self.endings[i+1]:
				y[i] = self.rewards[i]
				tmp = y[i]
			else:
				y[i] = self.rewards[i] + self.discount * tmp
				tmp = y[i]
				
		#print(self.discount)
		
		#print(self.rewards)
		#print(y)
	 			
		networkUtil.train(self.qNet, X, y, self.dtype, verbose=True, epoch=self.epoch,
						  numBatch=self.numBatch, lr=self.lr, p=self.p)
		
		self.states = []
		self.rewards = []
		self.endings = []
		self.timer = 0
		
class DeepACAgent(DeepQAgent):

	def getPNet(networkSize):
		net = []
		#print('getNet')
		for i in range(len(networkSize)-1):
			#print(i, networkSize[i],networkSize[i+1], self.dtype)
			net.append(nn.Linear(networkSize[i],networkSize[i+1] ))
			net.append(nn.LeakyReLU())
			#net.append(nn.BatchNorm1d(networkSize[i+1]))
		net.append(nn.Linear(networkSize[-1], self.numActions))
		
	def __init__(self, **args):
		DeepQAgent.__init__(self, **args)
		self.pNet = self.getPNet(self.networkSize)
		self.astates = []	
	
	
	
	def update(self, state, action, reward, ending=False):
		self.states.append(self.game.tensorize(state, action))
		self.astates.append(self.game.tensorizeState(state))
		self.rewards.append(reward)
		self.endings.append(ending)
		self.timer += 1
	 	
		if self.timer != self.batchsize: return
	 	
		X = np.array(self.states[:-1])
		y = np.zeros(len(self.states)-1)
	 	
		tmp = self._calQValue(self.states[-1])
	 	
		for i in range(len(self.states)-2,-1,-1):
			if self.endings[i+1]:
				y[i] = self.rewards[i]
			else:
				y[i] = self.rewards[i] + self.discount * tmp
			tmp = y[i]
		
				
		#print(self.discount)
		
		#print(self.rewards)
		#print(y)
	 			
		networkUtil.train(self.qNet, X, y, self.dtype, verbose=True, epoch=self.epoch, numBatch=self.numBatch)
		
		#
		
		self.states = []
		self.rewards = []
		self.endings = []
		self.timer = 0

class averageReward(nn.Module):
	def __init__(self):
		super(rewardLoss,self).__init__()

	def forward(self,x):
		loss = torch.mean(x)
		return loss

class criticNet(nn.Module):
	def __init__(self):
		super(criticNet,self).__init__()
		self.linear1 = nn.Linear(20,400)
		self.linear2 = nn.Linear(4,300)
		self.linear3 = nn.Linear(400,300)
		self.linear4 = nn.Linear(600,300)
		self.output = nn.Linear(300,1)
		self.batchNormalization = nn.BatchNorm1d(400)
		self.ReLU = nn.ReLU(inplace=True)
		self.layerList = [self.linear1,self.linear2,self.linear3,self.linear4,self.batchNormalization]


	def forward(self, state, action):
		s = self.linear1(state) #s:1*400
		s = self.batchNormalization(s)
		s = self.ReLU(s)
		s = self.linear3(s)#s:64*300
		a = self.linear2(action)#a:64*300
		o = torch.cat((s,a),1) #o:1*600
		o = self.linear4(o)
		return self.output(o)


class actor(object):

	def __init__(self):
		self.model = self.createNewActor()

	def createNewActor(self):
		model = nn.Sequential(
			nn.Linear(20,400),
			nn.ReLU(inplace = True),
			nn.BatchNorm1d(400),
			nn.Linear(400,300),
			nn.ReLU(inplace = True),
			nn.BatchNorm1d(300),
			nn.Linear(300,4),
			nn.Tanh(),
		)
		return model

	def target_update(self, new_actor, rate):
		for i in [0,2,3,5,6]:
			self.model._modules[str(i)].weight.data = (1-rate) * self.model._modules[str(i)].weight.data + rate * new_actor.model._modules[str(i)].weight.data
			self.model._modules[str(i)].bias.data = (1-rate) * self.model._modules[str(i)].bias.data + rate * new_actor.model._modules[str(i)].bias.data

class critic(object):

	def __init__(self):
		self.model = criticNet()

	def target_update(self, new_critic,rate):
		for i in range(len(self.model.layerList)):
			self.model.layerList[i].weight.data = (1-rate) * self.model.layerList[i].weight.data + rate * new_critic.model.layerList[i].weight.data
			self.model.layerList[i].bias.data = (1-rate) * self.model.layerList[i].bias.data + rate * new_critic.model.layerList[i].bias.data


class deepModel(DeepQAgent):
	"""docstring for ClassName"""
	def __init__(self, **args):#, discount=0.98,noiseWeight=0.1,miniBatchSize=64,targetRate=0.01):
		
		DeepQAgent.__init__(self, **args)
		self.discount = self.hparams['discount']
		self.critic = critic()
		self.actor = actor()
		self.critic.model.type(self.dtype)
		self.actor.model.type(self.dtype)
		self.noiseWeight = self.hparams['noiseWeight']
		self.buffer = []
		self.miniBatchSize = self.hparams['miniBatchSize']
		self.target_critic = critic()
		self.target_actor = actor()
		self.targetRate = self.hparams['targetRate']
		self.criticOptimizer = optim.Adam(self.critic.model.parameters(),lr=1e-3)
		self.actorOptimizer = optim.Adam(self.actor.model.parameters(),lr=1e-3)

	def getAction(self, state):
		state = self.game.tensorizeState(state)
		state = Variable(torch.FloatTensor(state))
		state = state.view(1,-1)
		self.actor.model.eval()
		return (self.actor.model(state).data+torch.randn(4)*self.noiseWeight).numpy().tolist()[0]

	def update(self, state, action, nextstate, reward):
		state = self.game.tensorizeState(state)
		nextstate = self.game.tensorizeState(nextstate)
		#print(state)
		self.actor.model.train()
		self.critic.model.train()
		self.buffer.append(list(state)+list(action)+list(nextstate)+[reward])
		if len(self.buffer)>1e6:
			self.buffer.pop(0)

		if len(self.buffer)>self.miniBatchSize:
			selected_sample = random.sample(self.buffer,self.miniBatchSize)
			selected_sample = np.asarray(selected_sample)
			s = torch.FloatTensor(selected_sample[:,0:20])
			a = torch.FloatTensor(selected_sample[:,20:24])
			n = torch.FloatTensor(selected_sample[:,24:44])
			r = torch.FloatTensor(selected_sample[:,44])

			#update critic
			si = Variable(s)
			ai = Variable(a)
			si1 = Variable(n)
			ri = Variable(r.contiguous().view(-1,1))
			a_estimate = self.target_actor.model(si1)
			value = self.target_critic.model(si1,a_estimate)
			yi = ri + self.discount * value
			yi_no_grad = yi.detach()
			lossfn = nn.MSELoss()
			loss = lossfn(self.critic.model(si,ai),yi_no_grad)
			self.criticOptimizer.zero_grad()
			loss.backward()
			self.criticOptimizer.step()


			#update actor
			si = Variable(s)
			a_outs = self.actor.model(si)
			predict_award = self.critic.model(si,a_outs)
			out = -1*predict_award.mean()
			self.actorOptimizer.zero_grad()
			out.backward()
			self.actorOptimizer.step()

			#update target
			self.target_actor.target_update(self.actor,self.targetRate)
			self.target_critic.target_update(self.critic,self.targetRate)

	def loadParam(self):
	
		if self.renew or not os.path.isfile(self.paramfile):	return
        
	
		self.actor.model = torch.load(self.paramfile + '_0').type(self.dtype)
		self.critic.model = torch.load(self.paramfile + '_1').type(self.dtype)
		self.target_actor.model = torch.load(self.paramfile + '_2').type(self.dtype)
		self.target_critic.model = torch.load(self.paramfile + '_3').type(self.dtype)

	def saveParam(self):
		torch.save(self.actor.model, self.paramfile + '_0')
		torch.save(self.critic.model, self.paramfile + '_1')
		torch.save(self.target_actor.model, self.paramfile + '_2')
		torch.save(self.target_critic.model, self.paramfile + '_3')



