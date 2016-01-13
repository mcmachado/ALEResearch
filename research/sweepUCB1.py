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

''' Standard UCB-1 algorithm. The arm value is calculated as the arm's average
    plus Rmax * UCB-1 index.''' 
def UCB1(b, numIterations, stepSize):
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
			armValue[i] = avg[i] + stepSize * (RMAX * np.sqrt(2 * np.log(t)/float(n[i])))

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

def __init__():

	for param in params:

		print
		print 'Step Size ' + str(param)

	 	lvlOptimism   = RMAX
		numSeeds      = 2000
		numIterations = 1000

	    #Variables that will store the results for all methods
		res_MAX         = []
		res_AVG         = []
		res_UCB         = []

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
			res_UCB[s-1].append(UCB1(b[s-1], numIterations, param))
			b[s-1].resetEnv()

		'''Now we can take the average return of each method:'''
		res_MAX_avg = []
		res_AVG_avg = []
		res_UCB_avg = []

		for i in xrange(numSeeds):
			res_MAX_avg.append(stat.mean(res_MAX[i]))
			res_AVG_avg.append(stat.mean(res_AVG[i]))
			res_UCB_avg.append(stat.mean(res_UCB[i]))


		print 'Max return       :', stat.mean(res_MAX_avg), ',', stat.stdev(res_MAX_avg)
		print 'AVG              :', stat.mean(res_AVG_avg), ',', stat.stdev(res_AVG_avg)
		print 'UCB-1            :', stat.mean(res_UCB_avg), ',', stat.stdev(res_UCB_avg)

__init__()
