import util
import random
import os
import pickle	
	
class Agent:

	def __init__(self, paramfile, alpha, epsilon, discount, mode, agentparam):
		self.paramfile = paramfile
		self.alpha = alpha
		self.epsilon = epsilon
		self.discount = discount
		self.agentparam = util.Counter() + agentparam
		self.mode = mode
		
	def loadParam(self):
		pass

	def getAction(self, state):
		pass
		
	def update(self, state, action, nextstate, reward):
		pass

	def getLegalActions(self):
		return ['P', 'W', 'O', 'S', 'Q']

	def printParameter(self):
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
        
        if not os.path.isfile(self.paramfile):
        	return
        
        parameterFile = open(self.paramfile, 'rb')
        
        tmp = util.pickleLoad(parameterFile)
        
        if tmp != None:
        	self.qValues += tmp
        parameterFile.close()
    
    def __init__(self, **args):
		
        Agent.__init__(self, **args)
    	
        self.agentName = 'qLearningAgent'
        
   
            
    def _updateQValue(self, stateAction, obj):
    	self.qValues[stateAction] = (1 - self.alpha) * self.qValues[stateAction]\
        			+ self.alpha*obj


    def getQValue(self, state, action):

        return self.qValues[(state,action)]


    def computeValueFromQValues(self, state):
    

        actions=self.getLegalActions()
        if len(actions)==0: return 0
        maxQValue=-1e5
        for action in actions:
            if self.getQValue(state,action)>maxQValue:
                maxQValue=self.getQValue(state,action)
        return maxQValue

    def computeActionFromQValues(self, state):
    
        actions=self.getLegalActions()
        if len(actions)==0: return None
        maxQValue=-1e15
        for action in actions:
            if self.getQValue(state,action)>maxQValue:
                maxQValue=self.getQValue(state,action)
                maxAction=action
        return maxAction

    def getAction(self, state):
    
        legalActions = self.getLegalActions()
        
        if self.mode == 'Test':
        	return self.computeActionFromQValues(state)
        	
        if len(legalActions)==0:
        	return None
        	
        maxAction=self.computeActionFromQValues(state)
        
        if util.flipCoin(self.epsilon):
        	return random.choice(legalActions)
        else:
        	return maxAction
        

    def update(self, state, action, nextState, reward):
    
        maxQValue=self.computeValueFromQValues(nextState)
        self._updateQValue((state,action), reward+self.discount*maxQValue)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def printParameter(self):
        parameterFile = open(self.paramfile,'wb')
        pickle.dump(self.qValues, parameterFile)
        parameterFile.close()
        
		
		

class MixQlAgent(QLearningAgent):

	def __init__(self, **args):
		QLearningAgent.__init__(self, **args)
		
		self.k = self.agentparam['k']
		self.N = self.agentparam['N']

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
        	
    
	def printParameter(self):
		parameterFile = open(self.paramfile,'wb')
		pickle.dump(self.qValues, parameterFile)
		pickle.dump(self.stateCounter, parameterFile)
		parameterFile.close()
		
	
	def getAction(self, state):
	
		legalActions = self.getLegalActions()
        
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

    def printParameter(self):
        parameterFile = open(self.paramfile,'w')
        parameterFile.write(str(self.weights))
        return None

