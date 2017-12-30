from agent import Agent
import util


class sarsaAgent(Agent):

	def __init__(self, actionList=['Q','P'], epsilon=0.1, gamma=0.99, alpha=0.3):
		self.epsilon = epsilon
		self.gamma = gamma
		self.alpha = alpha
		self.qValues = util.Counter()
		self.actionList = actionList
		
	
	def getQValue(self, state, action):
		return self.qValues[(state, action)]


	def computeValueFromQValues(self, state):
       
        next_actions = self.getLegalActions(state)
        
        if next_actions:
        	return max([self.getQValue(state, action) for action in next_actions])
        	
        return 0

	def computeActionFromQValues(self, state):
        
        value = self.computeValueFromQValues(state)
        
        opt_actions = [action for action in self.getLegalActions(state) 
        								if self.getQValue(state, action) == value]
        
        if opt_actions:
        	return random.choice(opt_actions)
        	
        return None

    def getAction(self, state):
        
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        
        
        if not legalActions:
        	return None
        	
        if util.flipCoin(self.epsilon):
        	action = random.choice(legalActions)
        else:
        	action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.qValues[(state, action)] = (1 - self.alpha) * self.qValues[(state, action)] \
        							   + self.alpha * (reward \
        							   + self.discount * self.computeValueFromQValues(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
