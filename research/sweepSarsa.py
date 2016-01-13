import sys
import random as r
import numpy as np
import statistics as stat

RMAX   = 9.0
params = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5.0]

class Bandits:
	numArms = 2
	mus     = [1.0, 3.0]
	sigmas  = [0.5, 9.0]

	print 'N(', mus[0], ',', sigmas[0], ')  N(', mus[1], ',', sigmas[1], ')  RMAX:', RMAX

	def __init__(self, s, numIterations):
		np.random.seed(seed=s)
		self.arms    = []
		self.currIdx = []

		for i in xrange(self.numArms):
			self.currIdx.append(0)
			self.arms.append([])
			self.arms[i] = np.random.normal(self.mus[i], self.sigmas[i], numIterations + 1)

	def pullArm(self, i):
		assert(i < self.numArms)
		self.currIdx[i] += 1
		return self.arms[i][self.currIdx[i] - 1]

	def resetEnv(self):
		for i in xrange(self.numArms):
			self.currIdx[i] = 0

def epsilonGreedy(theta, epsilon = 0.05):
	argmax = np.argmax(theta)

	if r.randint(1, 100) < int(epsilon * 100):
		return r.randint(0, len(theta)-1)	
	else:
		return argmax

''' Regular SARSA(0) implementation using a fixed step-size and
    an epsilon-greedy policy.'''
def SARSA(b, numIterations, stepSize, optimism=0):

	theta = []
	acumReturn = 0
	alpha      = stepSize

	for i in xrange(b.numArms):
		theta.append(optimism)

	for t in xrange(numIterations):
		i = epsilonGreedy(theta)
		
		reward = b.pullArm(i)
		acumReturn += reward
		theta[i] = theta[i] + alpha * (reward + 0 - theta[i])

	return acumReturn

def __init__():

	for param in params:

		print
		print 'Step Size ' + str(param)

	 	lvlOptimism   = RMAX
		numSeeds      = 2000
		numIterations = 1000

	    #Variables that will store the results for all methods
		res_MAX         = []
		res_SARSA_P     = []
		res_SARSA_O     = []

		'''Already pull all the arms and store them, this way one can
		   easily reproduce the results and come back to see other possibilities'''
		b = []
		for s in xrange(1, numSeeds+1):
			b.append(Bandits(s, numIterations))

	    #Run the experiment x times, where x is the number of seeds
		for s in xrange(1, numSeeds + 1):
			seed = s
			r.seed(seed)

			res_MAX.append([])
			res_SARSA_P.append([])
			res_SARSA_O.append([])

			maxReturn = 0

			'''First I just calculate the max return one could've get.'''
			for i in xrange(numIterations):
				maxReturn += max(b[s-1].pullArm(0), b[s-1].pullArm(1))
			
			res_MAX[s-1].append(maxReturn)
			b[s-1].resetEnv()

			'''Agent following the SARSA(0) pessimistically initialized.'''
			r.seed(seed)
			res_SARSA_P[s-1].append(SARSA(b[s-1], numIterations, param))
			b[s-1].resetEnv()

			'''Agent following the SARSA(0) optimistically initialized.'''
			r.seed(seed)
			res_SARSA_O[s-1].append(SARSA(b[s-1], numIterations, param, lvlOptimism))
			b[s-1].resetEnv()

		'''Now we can take the average return of each method:'''
		res_MAX_avg = []
		res_SARSA_P_avg = []
		res_SARSA_O_avg = []

		for i in xrange(numSeeds):
			res_MAX_avg.append(stat.mean(res_MAX[i]))
			res_SARSA_P_avg.append(stat.mean(res_SARSA_P[i]))
			res_SARSA_O_avg.append(stat.mean(res_SARSA_O[i]))

		print 'Max return       :', stat.mean(res_MAX_avg), ',', stat.stdev(res_MAX_avg)
		print 'SARSA(0) -- pess.:', stat.mean(res_SARSA_P_avg), ',', stat.stdev(res_SARSA_P_avg)
		print 'SARSA(0) -- opt. :', stat.mean(res_SARSA_O_avg), ',', stat.stdev(res_SARSA_O_avg)

__init__()
