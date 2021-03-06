import signal
try:
	from getch import getch
except:
	pass
import random
import math
import pickle
import numpy as np
try:
	import matplotlib.pyplot as plt 
except:
	pass


def get_input():
	try:
		return ord(getch())
	except:
		return 0
		
def interrupted(signum, frame):
	pass

TIMEOUT = 0.001

def get_ch():

	signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
	signal.signal(signal.SIGALRM, interrupted)

	s = get_input()

	signal.alarm(0)

	#print(s)
	
	return s

def convertDataToState(data):
    #convert the data extracted from muJoCo to the state of the robot
    state = tuple(data['vel_body'])
    state += (int(data['pos_body'][2]),)
    state += (int(data['angle_leftUp']),)
    state += (int(data['angle_leftLow']),)
    state += (int(data['angle_rightUp']),)
    state += (int(data['angle_rightLow']),)
    state += (int(data['vel_leftUp']),)
    state += (int(data['vel_leftLow']),)
    state += (int(data['vel_rightUp']),)
    state += (int(data['vel_rightLow']),)
    for i in range(3):
        state+=(int(data['vel_body'][i]),)
    '''state['pos_leftUpLeg']=[0,0,0]
    state['pos_rightUpLeg']=[0,0,0]
    state['pos_leftLowLeg']=[0,0,0]
    state['pos_rightLowLeg']=[0,0,0]
    for i in range(3):
		state['pos_leftUpLeg'][i] = int(data['pos_leftUpLeg'][i]*10)-int(data['pos_body'][i]*10)
		state['pos_rightUpLeg'][i] = int(data['pos_rightUpLeg'][i]*10)-int(data['pos_body'][i]*10)
		state['pos_leftLowLeg'][i] = int(data['pos_leftLowLeg'][i]*10)-int(data['pos_body'][i]*10)
		state['pos_rightLowLeg'][i] = int(data['pos_rightLowLeg'][i]*10)-int(data['pos_body'][i]*10)'''
    return state

class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting data.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print a['test']

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print a['test']
    0

    returns the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print a['test']
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print a['blah']
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(self.keys()) == 0: return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = self.items()
        compare = lambda x, y:  sign(y[1] - x[1])
        sortedItems.sort(cmp=compare)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0: return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y ):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x,y = y,x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in y.items():
            self[key] += value

    def __add__( self, y ):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__( self, y ):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend

    def multiplyAll(self, multiplier):
        multiplier = float(multiplier)
        newCounter=Counter()
        for key in self:

            newCounter[key] = self[key]*multiplier
        return newCounter

def flipCoin(p):
    r=random.random()
    return r<p

def norm2d(x):
    return math.sqrt(x[0] ** 2 + x[1] ** 2)
    
def chunk(x, unit, xmax):
	return max(min(math.floor(x/unit), xmax), -xmax)
	
def pickleLoad(loadfile):
	try:
		return pickle.load(loadfile)
	except EOFError:
		return None
		
def softmax(x):
	x -= x.max()
	
	ex = np.exp(x)
	
	return ex / ex.sum()
	

def plot_figure(stats, title='', xlabel='', ylabel='',
            savepath='./Figure/',
            save_file='acc'):
	"""
    Print data in the coordinate. each entry in data is in the
    form (x,y).
    
	Input: array of shape (N,2), 
	"""
	    
	plt.cla()
	for key, x in stats.items():
		#print(x)
		plt.plot(x, label=key)
   
	plt.title(title)
	plt.xlabel(xlabel)	
	plt.ylabel(ylabel)
	plt.legend()
	plt.savefig(savepath+save_file+'.png')

if __name__ == '__main__':
    a= Counter()
    print(a[9])
    a[9]+=1
    print(a[9])
