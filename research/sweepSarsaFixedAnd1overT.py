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

''' SARSA(0) implementation using our new algorithm of separating
    concerns. Both the uncertainty and the value function regarding
    the reward itself decay at a 1/t rate.'''
def SARSA_SPLIT(b, numIterations, stepSize1, stepSize2, optimism):

	n          = []
	psi        = []
	theta      = []
	acumReturn = 0
	alpha      = stepSize1
	beta       = stepSize2

	for i in xrange(b.numArms):
		n.append(0)
		psi.append(optimism)
		theta.append(0)

	for t in xrange(numIterations):
		i = epsilonGreedy(np.add(theta, psi))
		print theta[i], psi[i]
		n[i] += 1

		reward      = b.pullArm(i)
		acumReturn += reward
		theta[i]    = theta[i] + stepSize1                   * (reward + 0 - theta[i])
		psi[i]      = psi[i]   + stepSize2 * (1.0/(n[i]+1))  * (0      + 0 - psi[i])

	return acumReturn

def __init__():

	for param1 in params:
		for param2 in params:

			print
			print 'Step Size 1: ' + str(param1)
			print 'Step Size 2: ' + str(param2)

		 	lvlOptimism   = RMAX
			numSeeds      = 2000
			numIterations = 1000

		    #Variables that will store the results for all methods
			res_MAX         = []
			res_SARSA       = []

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
				res_SARSA.append([])

				maxReturn = 0

				'''First I just calculate the max return one could've get.'''
				for i in xrange(numIterations):
					maxReturn += max(b[s-1].pullArm(0), b[s-1].pullArm(1))
				
				res_MAX[s-1].append(maxReturn)
				b[s-1].resetEnv()

				'''Agent following the SARSA(0) with two value functions.'''
				r.seed(seed)
				res_SARSA[s-1].append(SARSA_SPLIT(b[s-1], numIterations, param1, param2, lvlOptimism))
				b[s-1].resetEnv()

			'''Now we can take the average return of each method:'''
			res_MAX_avg = []
			res_SARSA_avg = []

			for i in xrange(numSeeds):
				res_MAX_avg.append(stat.mean(res_MAX[i]))
				res_SARSA_avg.append(stat.mean(res_SARSA[i]))

			print 'Max return :', stat.mean(res_MAX_avg), ',', stat.stdev(res_MAX_avg)
			print 'SARSA(0)   :', stat.mean(res_SARSA_avg), ',', stat.stdev(res_SARSA_avg)

__init__()
