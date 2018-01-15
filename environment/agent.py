import util
import random
import os
import pickle	
import numpy as np
from game import Game

class Agent:

	def __init__(self, paramfile, mode, game, hparams, renew):
		self.paramfile = paramfile + '_' + self.__class__.__name__ + '_' + game.__class__.__name__ +  '.pam'
		print(self.paramfile)
		self.hparams = util.Counter() + hparams
		self.mode = mode
		self.game = game
		self.renew = renew
		
	def loadParam(self):
		pass

	def getAction(self, state):
		pass
		
	def update(self, state, action, nextstate, reward):
		pass

	def saveParam(self):
		pass

	def getActionParameter(self, action):
		if action == 'Q':
			return [1,-1,0,0]
		if action == 'W':
			return [0,0,-1,0]
		if action == 'O':
			return [0,0,0,-1]
		if action == 'P':
			return [-1,1,0,0]
		if action == 'S':
			return [0,0,0,0]


class RunAgent(Agent):

    def __init__(self, **args):
        Agent.__init__(self, **args)
        self.actionDict = {113:'Q', 119:'W', 111:'O', 112:'P', 114:'R'}
        self.agentName = 'keyboardAgent'

    def getAction(self,state):

        action = util.get_ch()
		
        if action in self.actionDict:
            return self.actionDict[action]
		
        return 0
        

	
class RandomAgent(Agent):

	def __init__(self, **args):
		Agent.__init__(self, **args)
		self.actionList = ['Q', 'W', 'O', 'P', 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

	def getAction(self):
		
		return random.choice(self.actionList)
		

class QLearningAgent(Agent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
         
    def loadParam(self):
    
        self.qValues=util.Counter()
        
        if self.renew or not os.path.isfile(self.paramfile):
        	return
        
        parameterFile = open(self.paramfile, 'rb')
        
        tmp = util.pickleLoad(parameterFile)
        
        if tmp != None:
        	self.qValues += tmp
        parameterFile.close()
    
    def __init__(self, **args):
		
        Agent.__init__(self, **args)
        
        self.alpha = self.hparams['alpha']
        self.discount = self.hparams['dicount']
        self.epsilon = self.hparams['epsilon']
        self.agentname = 'QLearningAgent'
   
            
    def _updateQValue(self, stateAction, obj):
        self.qValues[stateAction] = (1 - self.alpha) * self.qValues[stateAction]\
        			+ self.alpha*obj
        			
        #print(stateAction)


    def getQValue(self, state, action):
       
       
        dState = self.game.discretize(state)
    

        return self.qValues[(dState,action)]


    def computeValueFromQValues(self, state):

        actions = self.game.getLegalActions(state)
        
        if len(actions) == 0:
        	return 0
        
        dState = self.game.discretize(state)
        
        return max([self.qValues[(dState, action)] for action in actions])
        
    def computeActionFromQValues(self, state):
    
        actions = self.game.getLegalActions(state)
        if len(actions)==0: return None
        maxQValue=-1e15
        
        dState = self.game.discretize(state)
        
        #print(actions)
        
        maxValue = max([self.qValues[(dState, action)] for action in actions])

        
        maxActions = [action for action in actions if abs(self.qValues[(dState, action)]-maxValue) < 0.001]
        
        return random.choice(maxActions)

    def getAction(self, state):
    
        legalActions = self.game.getLegalActions(state)
        
        if self.mode == 'Test':
        	return self.computeActionFromQValues(state)
        	
        if util.flipCoin(self.epsilon):
        	return random.choice(legalActions)
        	
        return self.computeActionFromQValues(state)
        
        	
    def getScore(self, dState, action):
    	return self.qValues[(dState, action)]
        

    def update(self, state, action, nextState, reward):
    
        maxQValue=self.computeValueFromQValues(nextState)
        dState = self.game.discretize(state)
        
        #print(reward+self.discount*maxQValue)
        self._updateQValue((dState,action), reward+self.discount*maxQValue)
        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def saveParam(self):
        parameterFile = open(self.paramfile,'wb')
        pickle.dump(self.qValues, parameterFile)
        parameterFile.close()
        
		
		

class MixQlAgent(QLearningAgent):

	def __init__(self, **args):
		QLearningAgent.__init__(self, **args)
		
		self.k = self.hparams['k']
		self.N = self.hparams['N']
	
	def loadParam(self):

		self.qValues=util.Counter()
		self.stateCounter = util.Counter()
        
		if not os.path.isfile(self.paramfile):
			return
        
		parameterFile = open(self.paramfile, 'rb')
        
		tmp = util.pickleLoad(parameterFile)
        
		if tmp != None:
			self.qValues += tmp
        	
		tmp = util.pickleLoad(parameterFile)
        
		if tmp != None:
			self.stateCounter += tmp
			
		#print('Here')
        	
    
	def saveParam(self):
		parameterFile = open(self.paramfile,'wb')
		pickle.dump(self.qValues, parameterFile)
		pickle.dump(self.stateCounter, parameterFile)
		parameterFile.close()
		
	
	def getAction(self, state):
	
		legalActions = self.game.getLegalActions(state)
        
		if self.mode == 'Test':
			return self.computeActionFromQValues(state)
        
		num = self.stateCounter[state]
		
		self.stateCounter[state] += 1
        
		eps = self.epsilon
		
		#print(num, eps)
        
		if num<self.N:
			eps = max(eps, (1 - num/self.N) ** (1/self.k))
			
		#print(eps)
        
		maxAction=self.computeActionFromQValues(state)
        
		if util.flipCoin(eps):
			return random.choice(legalActions)
		else:
			return maxAction
	

class TDnAgent(QLearningAgent):
	
	def __init__(self, **args):
		
		QLearningAgent.__init__(self, **args);
		
		self.preStates = []
		self.preRewards = []
		self.preG = 0
		self.discount_n = self.discount ** self.nboost;
		
	def reset(self):
		self.preRewards = []
		self.preStates = []
		self.preReward = 0
		
	def update(self, state, action, nextState, reward):
		self.preStates.append((state, action))
		
		if len(self.preStates) == self.nboost:
			maxQValue=self.computeValueFromQValues(nextState)
        	
			self.preG /= self.discount
			obj = self.preG + self.discount_n * maxQValue;
			
			self.preG += -self.preRewards.pop(0) + reward * self.discount_n
			
			stateAction = self.preStates.pop(0)
			
			self._updateQValue(stateAction, obj)
			
		else:
			self.preG = self.preG / self.discount + reward * self.discount_n
			
		#print(self.preRewards)
		
		self.preRewards.append(reward)
		

class CrAcAgent(Agent):
	
         
	def loadParam(self):
	
		self.qValues = util.Counter()
		self.theta = np.random.randn(self.game.getDim() * self.game.getNumActions()) * 1e-2
        
		if self.renew or not os.path.isfile(self.paramfile):
			return
        
		parameterFile = open(self.paramfile, 'rb')
        
		tmp = util.pickleLoad(parameterFile)
        
		if tmp != None:
			self.qValues += tmp
		
		tmp = util.pickleLoad(parameterFile)
		
		if np.any(tmp):
			self.theta = tmp
			
		parameterFile.close()
		
	def __init__(self, **args):
		Agent.__init__(self, **args)
		
		self.alpha = self.hparams['alpha']
		self.beta = self.hparams['beta']
		self.discount = self.hparams['discount']
		self.epsilon = self.hparams['epsilon']
        
	def getScore(self, state, action):
	
		#print("state:", state, " action:", action, "tensor:")
		#print(self.game.tensorize(state, action))
	
		return (self.game.tensorize(state, action) * self.theta).sum()
		
	def _getPGrad(self, state, action):
		
		actionRate = self.getActionRate(state)
		
		#print(actionRate)
		
		return self.game.tensorize(state, action)\
			   - sum([rate * self.game.tensorize(state, _action) for _action, rate in actionRate])
    
	def getActionRate(self, state):
	
		legalActions = self.game.getLegalActions(state)
		
		score = np.array([self.getScore(state, action) for action in legalActions])
		
		#print(score)
		score = util.softmax(score)
		
		
		#print(score)
		
		return [(legalActions[i], score[i]) for i in range(len(score))]
		
	def getQValue(self, state, action):
		#print(action, self.qValues[(self.game.discretize(state), action)])
		return self.qValues[(self.game.discretize(state), action)]
		
        
	def computeValueFromQValues(self, state):
    
		actionRate = self.getActionRate(state)
		
		dState = self.game.discretize(state)
		
		return sum([rate * self.qValues[(dState,action)] for action, rate in actionRate])

	def getAction(self, state):
        
		#if self.mode == 'Train' and util.flipCoin(self.epsilon):
		#	return random.choice(self.game.getLegalActions(state))
        	
		actionRate = self.getActionRate(state)
		
		p = random.random()
        
		for action, rate in actionRate:
			if p < rate:
				return action
			p -= rate
        

	def update(self, state, action, nextState, reward):
	    
		value = self.computeValueFromQValues(nextState)
		dState = self.game.discretize(state)
		
		self.qValues[(dState, action)] += self.alpha * (reward + self.discount * value - self.qValues[(dState, action)])
		
		#print(action, self.qValues[(self.game.discretize(state), action)])
		
		self.theta += self.beta * self.qValues[(dState, action)] * self._getPGrad(state, action)


	def saveParam(self):
		parameterFile = open(self.paramfile,'wb')
		pickle.dump(self.qValues, parameterFile)
		pickle.dump(self.theta, parameterFile)
		parameterFile.close()
        

class NaiveCrAcAgent(CrAcAgent):

	def loadParam(self):
	
		self.qValues = util.Counter()
		self.theta = util.Counter()

		#print(self.paramfile)
		  	
		print(self.paramfile)
        
		if self.renew or not os.path.isfile(self.paramfile):
			return
			
        
		parameterFile = open(self.paramfile, 'rb')
        
		
        
		tmp = util.pickleLoad(parameterFile)
        
		if tmp != None:
			self.qValues += tmp
		
		tmp = util.pickleLoad(parameterFile)
		
		if tmp != None:
			self.theta += tmp
			
		parameterFile.close()
	
	def getScore(self, dState, action):
	
		#print("state:", state, " action:", action, "tensor:")
		#print(self.game.tensorize(state, action))
	
		return self.theta[(dState, action)]
	
	
	def getActionRate(self, state):
	
		legalActions = self.game.getLegalActions(state)
		
		dState = self.game.discretize(state)
		
		score = np.array([self.getScore(dState, action) for action in legalActions])
		
		#print(score)
		score = util.softmax(score)
		
		
		#print(score)
		
		return [(legalActions[i], score[i]) for i in range(len(score))]
		
	
	def update(self, state, action, nextState, reward):
	    
		value = self.computeValueFromQValues(nextState)
		dState = self.game.discretize(state)
		
		self.qValues[(dState, action)] += self.alpha * (reward + self.discount * value - self.qValues[(dState, action)])
		
		self.theta[(dState, action)] += self.beta * self.qValues[(dState, action)]
		
		actionRate = self.getActionRate(state)
		
		for _action, rate in actionRate:
			self.theta[(dState, _action)] -= self.beta * self.qValues[(dState, action)] * rate

class ConCrAcAgent(NaiveCrAcAgent):
	
	def getAction(self, state):
        
		#if self.mode == 'Train' and util.flipCoin(self.epsilon):
		#	return random.choice(self.game.getLegalActions(state))
        	
		return self.getActionRate(state)
		
class ApproximateQAgent(QLearningAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self,  **args):
    
        Agent.__init__(self, **args)

        self.weights = util.Counter()
        self.agentName = 'ApproximateQAgent'
        
        if not os.path.isfile(self.paramfile):
        	return
        
        parameterFile = open(self.paramfile)
        line = parameterFile.readline()
        if line!='' :
            self.weights += eval(line)

    def getWeights(self):
        return self.weights

    def getFeatures(self, state, action):
        feature=util.Counter()
        feature+=state
        for j in range(4):
            feature['ctrl'+str(j)] = self.getActionParameter(action)[j]
        return feature
            
        '''feature['xvel_body'] = state[0]
        feature['yvel_body'] = state[1]
        feature['zvel_body'] = state[2]
        feature['height'] = state[3]
        feature['angle_leftUp'] = state[4]
        feature['angle_leftLow'] = state[5]
        feature['angle_rightUp'] = state[6]
        feature['angle_rightLow'] = state[7]
        feature['vel_leftUp'] = state[8]
        feature['']'''


    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        return self.weights*self.getFeatures(state,action)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        difference= reward+self.discount*self.computeValueFromQValues(nextState)-self.getQValue(state,action)
        m=self.getFeatures(state,action)
        self.weights+=m.multiplyAll(self.alpha*difference)
        return None

    def saveParam(self):
        parameterFile = open(self.paramfile,'w')
        parameterFile.write(str(self.weights))
        return None

