import sys
import random as r
import numpy as np
import statistics as stat

RMAX = 9.0

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

''' Standard UCB-1 algorithm. The arm value is calculated as the arm's average
    plus Rmax * UCB-1 index.''' 
def UCB1(b, numIterations):
	n          = []
	avg        = []
	armValue   = []
	acumReturn = 0

	for i in xrange(b.numArms):
		n.append(1)
		armValue.append(0)
		avg.append(b.pullArm(i))

	for t in xrange(2, numIterations + 2):
		assert(t == sum(n))
		for i in xrange(b.numArms):
			armValue[i] = avg[i] + RMAX * np.sqrt(2 * np.log(t)/float(n[i]))

		i           = np.argmax(armValue)
		reward      = b.pullArm(i)
		acumReturn += reward
		avg[i]      = (reward + avg[i] * n[i])/(n[i] + 1)
		n[i]       += 1

	return acumReturn

''' This is a very simple algorithm. Each arm is pulled once and 
	then at every timestep the arm with highest average return is
	pulled. When an arm is pulled its average is updated.'''
def AVG(b, numIterations):
	n          = []
	avg        = []
	acumReturn = 0


	for i in xrange(b.numArms):
		n.append(1)
		avg.append(b.pullArm(i))

	for t in xrange(2, numIterations + 2):
		assert(t == sum(n))

		i           = np.argmax(avg)
		reward      = b.pullArm(i)
		acumReturn += reward
		avg[i]      = (reward + avg[i] * n[i])/(n[i] + 1)
		n[i]       += 1

	return acumReturn

def epsilonGreedy(theta, epsilon = 0.05):
	argmax = np.argmax(theta)

	if r.randint(1, 100) < int(epsilon * 100):
		return r.randint(0, len(theta)-1)	
	else:
		return argmax

''' Regular SARSA(0) implementation using a fixed step-size and
    an epsilon-greedy policy.'''
def SARSA(b, numIterations, optimism=0):

	theta = []
	acumReturn = 0
	alpha      = 0.5

	for i in xrange(b.numArms):
		theta.append(optimism)

	for t in xrange(numIterations):
		i = epsilonGreedy(theta)
		
		reward = b.pullArm(i)
		acumReturn += reward
		theta[i] = theta[i] + alpha * (reward + 0 - theta[i])

	return acumReturn

''' SARSA(0) implementation using our new algorithm of separating
    concerns. The value function regarding the reward decays at a
    1/t rate while the uncertainty decays at a 1/sqrt(t).'''
def SARSA_SQRT_SPLIT(b, numIterations, optimism):

	n          = []
	psi        = []
	theta      = []
	acumReturn = 0
	alpha      = 1.0

	for i in xrange(b.numArms):
		n.append(0)
		psi.append(optimism)
		theta.append(0)

	for t in xrange(numIterations):
		i = epsilonGreedy(np.add(theta, psi))
		n[i] += 1

		reward      = b.pullArm(i)
		acumReturn += reward
		theta[i]    = theta[i] + alpha * (1.0/n[i])                                 * (reward + 0 - theta[i])
		psi[i]      = psi[i]   + alpha * (1.0 - np.sqrt(float(n[i])/float(n[i]+1))) * (0      + 0 - psi[i])

	return acumReturn

''' SARSA(0) implementation using our new algorithm of separating
    concerns. Both the uncertainty and the value function regarding
    the reward itself decay at a 1/t rate.'''
def SARSA_SPLIT(b, numIterations, optimism):

	n          = []
	psi        = []
	theta      = []
	acumReturn = 0
	alpha      = 1.0

	for i in xrange(b.numArms):
		n.append(0)
		psi.append(optimism)
		theta.append(0)

	for t in xrange(numIterations):
		i = epsilonGreedy(np.add(theta, psi))
		n[i] += 1

		reward      = b.pullArm(i)
		acumReturn += reward
		theta[i]    = theta[i] + alpha * (1.0/(t+1))  * (reward + 0 - theta[i])
		psi[i]      = psi[i]   + alpha * (1.0/(t+1))  * (0      + 0 - psi[i])

	return acumReturn

def __init__():

 	lvlOptimism   = RMAX
	numSeeds      = 100
	numIterations = 1000

    #Variables that will store the results for all methods
	res_MAX         = []
	res_AVG         = []
	res_UCB         = []
	res_SARSA       = []
	res_SARSA_PESS  = []
	res_SARSA_SPLIT = []
	res_SARSA_SQRT_SPLIT = []

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
		res_AVG.append([])
		res_UCB.append([])
		res_SARSA.append([])
		res_SARSA_PESS.append([])
		res_SARSA_SPLIT.append([])
		res_SARSA_SQRT_SPLIT.append([])

		maxReturn = 0

		'''First I just calculate the max return one could've get.'''
		for i in xrange(numIterations):
			maxReturn += max(b[s-1].pullArm(0), b[s-1].pullArm(1))
			res_MAX[s-1].append(maxReturn)
			b[s-1].resetEnv()

		'''Agent pulling the arm with highest running average.'''
		r.seed(seed)
		res_AVG[s-1].append(AVG(b[s-1], numIterations))
		b[s-1].resetEnv()

		'''Agent following the UCB-1 algorithm.'''
		r.seed(seed)
		res_UCB[s-1].append(UCB1(b[s-1], numIterations))
		b[s-1].resetEnv()
		
		'''Agent following SARSA(0) with a fixed step-size.'''
		r.seed(seed)
		res_SARSA_PESS[s-1].append(SARSA(b[s-1], numIterations))
		b[s-1].resetEnv()

		'''Agent following SARSA(0) optimistically initialized
		   with a fixed step-size.'''
		r.seed(seed)
		res_SARSA[s-1].append(SARSA(b[s-1], numIterations, lvlOptimism))
		b[s-1].resetEnv()
		
		'''Agent following SARSA(0) optimistically initialized
		   following a square root decay regimen'''
		r.seed(seed)
		res_SARSA_SQRT_SPLIT[s-1].append(SARSA_SQRT_SPLIT(b[s-1], numIterations, lvlOptimism))
		b[s-1].resetEnv()
		
		'''Agent following SARSA(0) optimistically initialized
		   following a 1/t decay regimen'''
		r.seed(seed)
		res_SARSA_SPLIT[s-1].append(SARSA_SPLIT(b[s-1], numIterations, lvlOptimism))
		b[s-1].resetEnv()
		
	'''Now we can take the average return of each method:'''
	res_MAX_avg = []
	res_AVG_avg = []
	res_UCB_avg = []
	res_SARSA_avg = []
	res_SARSA_PESS_avg = []
	res_SARSA_SPLIT_avg = []
	res_SARSA_SQRT_SPLIT_avg = []

	for i in xrange(numSeeds):
		res_MAX_avg.append(stat.mean(res_MAX[i]))
		res_AVG_avg.append(stat.mean(res_AVG[i]))
		res_UCB_avg.append(stat.mean(res_UCB[i]))
		res_SARSA_avg.append(stat.mean(res_SARSA[i]))
		res_SARSA_PESS_avg.append(stat.mean(res_SARSA_PESS[i]))
		res_SARSA_SPLIT_avg.append(stat.mean(res_SARSA_SPLIT[i]))
		res_SARSA_SQRT_SPLIT_avg.append(stat.mean(res_SARSA_SQRT_SPLIT[i]))

	print 'Max return       :', stat.mean(res_MAX_avg), ',', stat.stdev(res_MAX_avg)
	print 'AVG              :', stat.mean(res_AVG_avg), ',', stat.stdev(res_AVG_avg)
	print 'UCB-1            :', stat.mean(res_UCB_avg), ',', stat.stdev(res_UCB_avg)
	print 'SARSA (pess. in.):', stat.mean(res_SARSA_PESS_avg), ',', stat.stdev(res_SARSA_PESS_avg)
	print 'SARSA (opt. in.) :', stat.mean(res_SARSA_avg), ',', stat.stdev(res_SARSA_avg)
	print 'SARSA (1/t)      :', stat.mean(res_SARSA_SPLIT_avg), ',', stat.stdev(res_SARSA_SPLIT_avg)
	print 'SARSA (1/sqrt(t)):', stat.mean(res_SARSA_SQRT_SPLIT_avg), ',', stat.stdev(res_SARSA_SQRT_SPLIT_avg)

__init__()
