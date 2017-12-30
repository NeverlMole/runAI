from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjViewerBasic
import math
import os
import time
import sys
import util
from agent import *
#from learningAgent import *

class Simulator:
	
	def __init__(self, model, agent):
		self.sim = MjSim(model)
		self.viewer = MjViewer(self.sim)
		self.agent = agent
		self.timer = 0
		self.data = {}
		self.averageDistance = 0
		self.totDistance = 0
		self.totReward = 0
		self.rewardtimes = 0
		self.resetTimes = 0
		
	def _updateData(self, key, value):
		'''Update the value for key and velocity for key.'''
		
		v_key = key + '_v'
		
		if key in self.data:
			self.data[v_key] = value - self.data[key]
		else:
			self.data[v_key] = 0
			
		self.data[key] = value
		
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
		elif action == 'E':
			self.sim.data.ctrl[2] = 1
		elif action == 'U':
			self.sim.data.ctrl[3] = 1
		elif action == 'R':
			self.reset()
			
	def reset(self):
		
		self.resetTimes += 1;
		self.totDistance += self.data['distance']
		self.averageDistance = self.totDistance/self.resetTimes;
		
		self.sim.reset()
		self.timer = 0
		
	def resetTimer(self):
		self.timer = 0
		self.averageDistance = 0
		self.totDistance = 0
		self.totReward = 0
		self.rewardtimes = 0
		self.resetTimes = 0
	
	def step(self):
		self.sim.step()
		self.timer += 1
		
	def display(self):
		self.viewer.render()
		
	def isDeath(self):
		return self.data['height']<0.3 or abs(self.data['angle_rotate'])>2  #\
			   #or abs(self.data['angle_joint_leftup']) > 2 \
			   
			   #or abs(self.data['angle_joint_rightup']) > 2 \
			   
			   #or abs(self.data['angle_joint_leftup'] - self.data['angle_joint_rightup']) > 3
			
	def updateData(self):
	
		rx, ry, rz = self.sim.data.get_geom_xpos('bodyRight')
		ux, uy, uz = self.sim.data.get_geom_xpos('bodyUp')
		cx, cy, cz = self.sim.data.get_geom_xpos('body')
		
		self.data['height'] = cz
		self.data['distance'] = cx
		#self.data['right_body'] = self.sim.data.get_geom_xpos('bodyRight')
		#self.data['up_body'] =  self.sim.data.get_geom_xpos('bodyUp') 
		#self.data['body_pos'] = self.sim.data.get_geom_xpos('body')
		self.data['angle_bend'] = math.atan2(util.norm2d((ux-cx,uy-cy)),uz-cz)
		self.data['angle_rotate'] = math.atan2(ry-cy,rx-cx) - math.pi/2
		self.data['velocity'] = self.sim.data.get_geom_xvelp('body')[0]
		self.data['angle_joint_rightup'] = self.sim.data.get_joint_qpos('rightUp')
		self.data['angle_joint_rightup_v'] = self.sim.data.get_joint_qvel('rightUp')
		self.data['angle_joint_leftup'] = self.sim.data.get_joint_qpos('leftUp')
		self.data['angle_joint_leftup_v'] = self.sim.data.get_joint_qvel('leftUp')
		self.data['angle_joint_rightdown'] = self.sim.data.get_joint_qpos('rightLow')
		self.data['angle_joint_rightdown_v'] = self.sim.data.get_joint_qvel('rightLow')
		self.data['angle_joint_leftdown'] = self.sim.data.get_joint_qpos('leftLow')
		self.data['angle_joint_leftdown_v'] = self.sim.data.get_joint_qvel('leftLow')
		self.data['avg_velocity'] = self.data['distance']/(self.timer+1e-5)*300
		
		#print(self.data['angle_joint_leftup_v'])
		
	def getDiscreteState(self):
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
		state['bend_state'] = math.floor(self.data['angle_bend']/bend_unit)
		state['rotate_state'] = math.floor(self.data['angle_rotate']/rotate_unit)
		
		state['rightup_ang'] = util.chunk(self.data['angle_joint_rightup'], angleup_unit, angleup_max)
		state['rightdown_ang'] = util.chunk(self.data['angle_joint_rightdown'], 
									angledown_unit, angledown_max)
		state['leftup_ang'] = util.chunk(self.data['angle_joint_leftup'], angleup_unit, angleup_max)
		state['leftdown_ang'] = util.chunk(self.data['angle_joint_leftdown'], 
									angledown_unit, angledown_max)
		
		state['rightup_angv'] = util.chunk(self.data['angle_joint_rightup_v'],
											anglevup_unit, anglevup_max)
		state['rightdown_angv'] = util.chunk(self.data['angle_joint_rightdown_v'],
											anglevdown_unit, anglevdown_max)
		state['leftup_angv'] = util.chunk(self.data['angle_joint_leftup_v'],
											anglevup_unit, anglevup_max)
		state['leftdown_angv'] = util.chunk(self.data['angle_joint_leftdown_v'],
											anglevdown_unit, anglevdown_max)
		
		#print(state)
		
		return tuple([x for x in state.items()])
		
	def getReward(self, action):
		reward = 0
		if self.isDeath():
			reward = -1
		elif action == 'R':
			reward = -99
		else:
			reward = (100+self.data['distance']) * self.data['velocity']
			
		reward = -reward
		
		self.totReward += reward
		self.rewardtimes += 1
		
		return reward
		
	def printData(self):
		
		for key in self.data:
			print(key, ':', self.data[key])
	


