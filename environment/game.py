class Game:

	def getLegalActions(state):
		
		#print(state)
		if state[-1][1]:
			return ['P', 'Q', 'S']
		else:
			return ['W', 'O', 'S']
			
	
	def discretize(ori_state):
		
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
		
		state['switch'] = self.timer % 2
		#print(state)
		
		return tuple([x for x in state.items()])
