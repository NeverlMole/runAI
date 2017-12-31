'''
Running code:
python runner.py  [-a NAME] [-f FILEPATH] [-d] [-m Train/Test] [-p DICT] [-g NAME] [-v] [--renew]

-a 				type of the agent	(runAgent, randomAgent, QLearningAgent, ApproximateQAgent
									 TDnAgent)
-f 				file to store the parameter (../data/ql001)
--printrate		print the result after VALUE number of simulations
-d 				display or not
-m 				'Train' or 'Test'
-p 				the parameter for different agent
-g 				the kind of game
-v				verbose

--renew			renew the params

--nboost		number of boosting

Example command:
python runner.py -a CrAcAgent -f ../data/ac001.pickle \
--printrate 1000 -m Train \
-p {\'discount\':0.9999\,\'alpha\':0.1\,\'beta\':0.1\,\'k\':5\,\'N\':100} \
-g BasicGame
''' 

from mujoco_py import load_model_from_xml, MjSim
import math
import os
import time
import sys
import agent
import util
from simulator import Simulator
import game

'''version 1 of simulator
class Simulator:
	
	def __init__(self, model, agent):
		self.sim = MjSim(model)
		self.viewer = MjViewer(self.sim)
		self.agent = agent
		self.timer = 0
		
	def takeAction(self, action):
	
    	#0 is leftUp, 1 is rightUp, 2 is leftLow, 3 is rightLow
    	
		self.sim.data.ctrl[0] = 0
		self.sim.data.ctrl[1] = 0
		self.sim.data.ctrl[2] = 0
		self.sim.data.ctrl[3] = 0
		
		if action == 'Q':
			self.sim.data.ctrl[0] = 1
			self.sim.data.ctrl[1] = -1
		elif action == 'W':
			self.sim.data.ctrl[2] = -1
		elif action == 'O':
			self.sim.data.ctrl[3] = -1
		elif action == 'P':
			self.sim.data.ctrl[0] = -1
			self.sim.data.ctrl[1] = 1
		elif action == 'R':
			self.reset()
			
	def reset(self):
		self.sim.reset()
		self.timer = 0
	
	def step(self):
		self.sim.step()
		self.timer += 1
		
	def display(self):
		self.viewer.render()
		
	def checkDeath(self):
		if self.sim.data.get_geom_xpos('body')[2]<0.2:
			self.reset()
			
	def getData(self):
		data = {}
		
		data['pos_body'] = self.sim.data.get_geom_xpos('body')
		data['pos_leftUpLeg'] = self.sim.data.get_geom_xpos('leftUpLeg')
		data['pos_leftLowLeg'] = self.sim.data.get_geom_xpos('leftLowLeg')
		data['pos_rightUpLeg'] = self.sim.data.get_geom_xpos('rightUpLeg')
		data['pos_rightLowLeg'] = self.sim.data.get_geom_xpos('rightLowLeg')
		data['vel_body'] = self.sim.data.get_geom_xvelp('body')
		#data['direction'] = self.sim.data.get_geom_xmat('body')
		#data['vel_direction'] = self.sim.data.get_geom_xvelr('body')
		data['angle_leftUp'] = self.sim.data.get_joint_qpos('leftUp')
		data['angle_rightUp'] = self.sim.data.get_joint_qpos('rightUp')
		data['angle_leftLow'] = self.sim.data.get_joint_qpos('leftLow')
		data['angle_rightLow'] = self.sim.data.get_joint_qpos('rightLow')
		data['vel_leftUp'] = self.sim.data.get_joint_qvel('leftUp')
		data['vel_rightUp'] = self.sim.data.get_joint_qvel('rightUp')
		data['vel_leftLow'] = self.sim.data.get_joint_qvel('leftLow')
		data['vel_rightLow'] = self.sim.data.get_joint_qvel('rightLow')

		data['distance'] = data['pos_body'][0]
		data['avg_velocity'] = data['distance']/(self.timer+1e-5)*300
		data['time'] = time.time()
		
		return data
		
	def printData(self):
		data = self.getData()
		
		for key in data:
			print(key, ':', data[key])
'''	

####################################### Init #################################

lenArgv = len(sys.argv)

dictHparam = {'agent':'RunAgent',
			  'paramfile':'param.txt',
			  'epsilon':0.05,
			  'alpha':0.1,
			  'discount':0.2,
			  'printrate':10000,
			  'display':False,
			  'agentparam':'{}',
			  'mode':'Train',
			  'verbose':False,
			  'game':'BasicGame',
			  'renew':False}
			  
for i in range(lenArgv):
	if (sys.argv[i]=='-a'):
		dictHparam['agent'] = sys.argv[i+1]
	
	if (sys.argv[i]=='-f'):
		dictHparam['paramfile'] = sys.argv[i+1]
	
	if sys.argv[i] == '--printrate':
		dictHparam['printrate'] = int(sys.argv[i+1])
		
	
	if sys.argv[i] == '-m':
		dictHparam['mode'] = sys.argv[i+1]	
	
	if sys.argv[i] == '-d':
		dictHparam['display'] = True
	
	if sys.argv[i] == '-p':
		dictHparam['agentparam'] = eval(sys.argv[i+1])
	
	if sys.argv[i] == '-g':
		dictHparam['game'] = sys.argv[i+1]
		
	if sys.argv[i] == '-v':
		dictHparam['verbose'] = True
	
	if sys.argv[i] == '--renew':
		dictHparam['renew'] = True

print(dictHparam)
		


model = load_model_from_xml(open('./runner.xml', "r").read())
mode = dictHparam['mode']

t = 0

game = getattr(game, dictHparam['game'])()

agentClass = getattr(agent, dictHparam['agent'])

agent = agentClass(paramfile=dictHparam['paramfile'], 
				   hparams=dictHparam['agentparam'],
				   mode=mode,
				   game=game,
				   renew=dictHparam['renew'])

agent.loadParam()

if dictHparam['renew']:
	result_file = open(dictHparam['paramfile'] + '.txt', 'w')
	result_file.write(repr('Num of Iterations').ljust(20) + \
					  repr('Average Distance').ljust(25) + \
					  repr('Average Reward').ljust(25) + \
					  repr('Num of Reset').ljust(15)+'\n')
					  
else:
	result_file = open(dictHparam['paramfile'] + '.txt', 'a')

sim = Simulator(model, agent)

################################## End Init #######################################

while True:
	t+=1
	sim.updateData()
	
	if sim.timer%11 == 0:
		if sim.timer > 1 and mode == 'Train':
			reward = sim.getReward(action);
			nextState = sim.getState()
			#print(state, action, nextState)
			agent.update(state, action, nextState, reward)
			
		if sim.isDeath():
			sim.reset()
		
		state = sim.getState()
		action = agent.getAction(state)
		
		sim.takeAction(action)
		
		#print(action)
		
	if dictHparam['verbose'] and sim.timer%101 == 0 :
		state = sim.getState()
		actions = game.getLegalActions(state)
		print('At state :', state)
		for action in actions:
			print('Value of action ', action, ' is:', agent.getQValue(state, action), 
					'and the score is:', agent.getScore(game.discretize(state), action))
		
		print(agent.getActionRate(state))
			
		
	
	if mode == 'Train' and (not (t%dictHparam['printrate'])):
		agent.printParameter()
		print(t, 'average distance:', sim.averageDistance, 
				'reset times:', sim.resetTimes)
		print('averaged reward:', sim.totReward/sim.rewardtimes)
		
		result_file.write(repr(t).ljust(20) + \
						  repr(sim.averageDistance).ljust(25) + \
						  repr(sim.totReward/sim.rewardtimes).ljust(25) + \
						  repr(sim.resetTimes).ljust(15) + '\n')
		sim.resetTimer()
	
	sim.step()

	if (dictHparam['display']):
		sim.display()

			

	'''if agent.agentName=='qLearningAgent' and sim.timer%10==0:
		if sim.timer>10:
			#update qvalues
			reward = sim.getReward(action);
			nextState = sim.getDiscreteState()
			agent.update(state,action,nextState,reward)
			
			if sim.isDeath():
				sim.reset()
			#print(reward)

		#take action
		state = sim.getDiscreteState()
		action = agent.getAction(state)
		sim.takeAction(action)

	if agent.agentName=='keyboardAgent' and sim.timer%10==0:
		state = sim.getDiscreteState()
		action = agent.getAction(state)
		sim.takeAction(action)

	if agent.agentName=='ApproximateQAgent' and sim.timer%100==0:
		if sim.timer>100:
			#update
			reward = sim.data['velocity']
			nextState = sim.data
			agent.update(state,action,nextState,reward)

		#take action
		state = sim.data
		action = agent.getAction(state)
		sim.takeAction(action) '''
    

	#if not (t%300):
	#	sim.printData()

	
