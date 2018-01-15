import numpy as np
import math
import util

class Game:
	
	def getLegalActions(self, state):
		pass
	
	def discretize(self, state):
		pass
		
	def tensorize(self, state, action):
		pass
		
	def getDim(self):
		pass

class BasicGame(Game):

	def __init__(self):
		self.actionList = ['P', 'Q', 'S', 'W', 'O']
		self.dim = 20
		self.actionDict = {self.actionList[i] : i for i in range(len(self.actionList))}
		self.numActions = len(self.actionList)
		self.actionStates = 2
		self.real = {'P':[-10000, 10000, 0 ,0], 'Q':[10000, -10000, 0 , 0], 'O':[0, 0, 0, -10000], 'W':[0, 0, -10000, 0],
					 'S':[0, 0, 0, 0]}

	def getDim(self):
		return self.dim
		
	def getNumActions(self):
		return self.numActions
		
	def allActions(self):
		return self.actionList

	def getLegalActions(self, state):
		
		#print(state)
		if state['timer'] % 2:
			return ['P', 'Q', 'S']
		else:
			return ['W', 'O', 'S']
			
	
	def discretize(self, ori_state):
		
		bend_unit = 0.5
		rotate_unit = 0.5
		angleup_unit = 0.3
		angledown_unit = 0.8
		angleup_max = 8
		angledown_max = 3
		anglevup_unit = 3
		anglevdown_unit =4
		anglevup_max = 4
		anglevdown_max = 2
		
		state = {}
		state['bend_state'] = math.floor(ori_state['angle_bend']/bend_unit)
		state['rotate_state'] = math.floor(ori_state['angle_rotate']/rotate_unit)
		
		state['rightup_ang'] = util.chunk(ori_state['angle_joint_rightup'], angleup_unit, angleup_max)
		state['rightdown_ang'] = util.chunk(ori_state['angle_joint_rightdown'], 
									angledown_unit, angledown_max)
		state['leftup_ang'] = util.chunk(ori_state['angle_joint_leftup'], angleup_unit, angleup_max)
		state['leftdown_ang'] = util.chunk(ori_state['angle_joint_leftdown'], 
									angledown_unit, angledown_max)
		
		state['rightup_angv'] = util.chunk(ori_state['angle_joint_rightup_v'],
											anglevup_unit, anglevup_max)
		state['rightdown_angv'] = util.chunk(ori_state['angle_joint_rightdown_v'],
											anglevdown_unit, anglevdown_max)
		state['leftup_angv'] = util.chunk(ori_state['angle_joint_leftup_v'],
											anglevup_unit, anglevup_max)
		state['leftdown_angv'] = util.chunk(ori_state['angle_joint_leftdown_v'],
											anglevdown_unit, anglevdown_max)
		
		state['switch'] = ori_state['timer'] % self.actionStates
		#print(state)
		
		return tuple([x for x in state.values()])
		
	def tensorizeState(self, state):
		
		tState = np.zeros(self.dim)
		
		tState[0] = state['height']
		tState[1] = state['angle_bend']
		tState[2] = state['angle_rotate']
		tState[3] = state['vx']
		tState[4] = state['vy']
		tState[5] = state['vz']
		tState[6] = state['angle_joint_rightup']
		tState[7] = state['angle_joint_rightup_v']
		tState[8] = state['angle_joint_leftup']
		tState[9] = state['angle_joint_leftup_v']
		tState[10] = state['angle_joint_rightdown']
		tState[11] = state['angle_joint_rightdown_v']
		tState[12] = state['angle_joint_leftdown']
		tState[13] = state['angle_joint_leftdown_v']
		tState[14] = state['flx']
		tState[15] = state['fly']
		tState[16] = state['flz']
		tState[17] = state['frx']
		tState[18] = state['fry']
		tState[19] = state['frz']
		
		return tState
		
	def tensorize(self, state, action):
		tState = self.tensorizeState(state)
		
		tmp = np.zeros(self.dim * self.numActions)
		i = self.actionDict[action]
		
		tmp[i * self.dim: (i+1) * self.dim] = tState
		
		return tmp
		
	def realAction(self, action):
		return self.real[action]
		
class FlexGame(BasicGame):
	
	def __init__(self):
	
		BasicGame.__init__(self)
		self.actionList = ['P', 'Q', 'S', 'W', 'O', 'E', 'I']
		
		self.actionDict = {self.actionList[i] : i for i in range(len(self.actionList))}
		self.numActions = len(self.actionList)
		self.actionStates = 3
		
		self.real = {'P':[-10000, 10000, 0 ,0], 'Q':[10000, -10000, 0 , 0], 'O':[0, 0, 0, -10000], 'W':[0, 0, -10000, 0],
					 'S':[0, 0, 0, 0], 'I':[0, 0, 0, 10000], 'E':[0, 0, 10000, 0]}
		
	def getLegalActions(self, state):
		
		#print(state)
		if state['timer'] % 3 == 1:
			return ['P', 'Q', 'S']
		elif state['timer'] % 3 == 2:
			return ['W', 'E', 'S']
		else:
			return ['O', 'I', 'S']
			
class SimpleGame(BasicGame):
	def __init__(self):
	
		BasicGame.__init__(self)
		
		self.actionList = ['P', 'Q', 'S', 'W', 'O']
		
		self.actionDict = {self.actionList[i] : i for i in range(len(self.actionList))}
		self.numActions = len(self.actionList)
		self.actionStates = 2
		
		self.real = {'P':[-10000, 10000, 0, 0], 'Q':[10000, -10000, 0, 0], 'O':[0, 0, -30, 10000], 
					 'W':[0, 0, 10000, -30], 'S':[0, 0, -300, -300]}			 
	
			
