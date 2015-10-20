import sys
import random as r
import numpy as np

class Bandits:
	numArms = 2
	mus     = [0.5, 0.6]
	sigmas  = [1.0, 1.0]

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
			armValue[i] = avg[i] + np.sqrt(2 * np.log(t)/float(n[i]+1))

		i           = np.argmax(armValue)
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

def SARSA(b, numIterations, optimism):

	n     = []
	phi   = []
	theta = []
	acumReturn = 0
	alpha      = 1.0

	for i in xrange(b.numArms):
		n.append(0)
		phi.append(0.0)
		theta.append(optimism)

	for t in xrange(numIterations - 1):
		i = epsilonGreedy(theta)
		n[i] += 1
		
		reward = b.pullArm(i)
		acumReturn += reward
		theta[i] = theta[i] + alpha/n[i] * (reward + 0 - theta[i])

	return acumReturn

def SARSA_SPLIT(b, numIterations, optimism):

	n          = []
	phi        = []
	psi        = []
	theta      = []
	acumReturn = 0
	alpha      = 1.0

	for i in xrange(b.numArms):
		n.append(0)
		phi.append(0.0)
		psi.append(optimism)
		theta.append(0)

	for t in xrange(numIterations - 1):
		i = epsilonGreedy(np.add(theta, psi))
		n[i] += 1

		reward      = b.pullArm(i)
		acumReturn += reward
		theta[i]    = theta[i] + alpha/n[i]                                   * (reward + 0 - theta[i])
		psi[i]      = psi[i]   + (alpha - np.sqrt(float(n[i])/float(n[i]+1))) * (0      + 0 - psi[i])

	return acumReturn

def __init__():

	res_MAX         = 0.0
	res_AVG         = 0.0
	res_UCB         = 0.0
	res_SARSA       = 0.0
	res_SARSA_SPLIT = 0.0

 	lvlOptimism   = 1.0
	numSeeds      = 100
	numIterations = 1000

	b = []
	for s in xrange(1, numSeeds+1):
		b.append(Bandits(s, numIterations))

	for s in xrange(1, numSeeds+1):
		seed = s
		r.seed(seed)

		maxReturn = 0
		for i in xrange(numIterations):
			maxReturn += max(b[s-1].pullArm(0), b[s-1].pullArm(1))


		res_MAX += maxReturn
		b[s-1].resetEnv()

		res_AVG += AVG(b[s-1], numIterations)
		b[s-1].resetEnv()
		r.seed(seed)

		res_UCB += UCB1(b[s-1], numIterations)
		b[s-1].resetEnv()
		r.seed(seed)

		res_SARSA += SARSA(b[s-1], numIterations, lvlOptimism)
		b[s-1].resetEnv()
		r.seed(seed)

		res_SARSA_SPLIT += SARSA_SPLIT(b[s-1], numIterations, lvlOptimism)
		b[s-1].resetEnv()
		
	print 'Max return :', res_MAX/numSeeds
	print 'AVG        :', res_AVG/numSeeds
	print 'UCB-1      :', res_UCB/numSeeds
	print 'SARSA      :', res_SARSA/numSeeds
	print 'SARSA_SPLIT:', res_SARSA_SPLIT/numSeeds

__init__()
