import util

def read_file(filepath):
	file = open(filepath, 'r')
	
	file.readline()
	
	a = [list(filter(lambda x: x != '' and x != '\n', line.split(' '))) for line in file]
	
	a = [list(map(lambda x: eval(x), X)) for X in a]
	
	dic = {'Average Distance': [], 'Average Reward': []}#, 'Num of Reset': []}
	
	for i in range(0, (int)(len(a)/30)*2, 1):
		dic['Average Distance'].append(a[i][1])
		dic['Average Reward'].append(a[i][2]) 
		#dic['Num of Reset'].append(float(a[i][3])) 
	
	return dic
	
data = read_file('../data/Com.txt')


util.plot_figure(data , title='Deep Q-learning Agent with Smallnet', xlabel='Num of Iterations * 10000', ylabel='',
          		savepath='../figure/',
           	 	save_file='DeepSmallnet')

data = read_file('../data/MCom.txt')


util.plot_figure(data , title='Deep Q-learning Agent with Bignet', xlabel='Num of Iterations * 10000', ylabel='',
          		savepath='../figure/',
           	 	save_file='DeepBignet')

'''


data = read_file('../data/ql001B.txt')


util.plot_figure(data , title='Q-learning Agent for EasyModel', xlabel='Num of Iterations * 1000000', ylabel='',
          		savepath='../figure/',
           	 	save_file='Qlearning_B')
           	 	       	 	
data = read_file('../data/ql001F.txt')


util.plot_figure(data , title='Q-learning Agent for FlexModel', xlabel='Num of Iterations * 1000000', ylabel='',
          		savepath='../figure/',
           	 	save_file='Qlearning_F')
           	 	
data = read_file('../data/nac001B.txt')


util.plot_figure(data , title='Actor-Critic Agent for EasyModel', xlabel='Num of Iterations * 1000000', ylabel='',
          		savepath='../figure/',
           	 	save_file='AC_B')
           	 	
data = read_file('../data/nac001F.txt')


util.plot_figure(data , title='Actor-Critic Agent for FlexModel', xlabel='Num of Iterations * 1000000', ylabel='',
          		savepath='../figure/',
           	 	save_file='AC_F')'''
