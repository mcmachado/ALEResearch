import sys
import random as r
import numpy as np
import statistics as stat

RMAX   = 9.0
FIRST_SEED = 1

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
		n[i] += 1

		reward      = b.pullArm(i)
		acumReturn += reward
		theta[i]    = theta[i] + stepSize1 * (1.0/(n[i]+1))  * (reward + 0 - theta[i])
		psi[i]      = psi[i]   + stepSize2 * (1.0/(n[i]+1))  * (0      + 0 - psi[i])

	return acumReturn

''' SARSA(0) implementation using our new algorithm of separating
    concerns. Both the uncertainty and the value function regarding
    the reward itself decay at a 1/t rate.'''
def SARSA_SPLIT_FIXED_1oT(b, numIterations, stepSize1, stepSize2, optimism):

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
		n[i] += 1

		reward      = b.pullArm(i)
		acumReturn += reward
		theta[i]    = theta[i] + stepSize1                   * (reward + 0 - theta[i])
		psi[i]      = psi[i]   + stepSize2 * (1.0/(n[i]+1))  * (0      + 0 - psi[i])

	return acumReturn

''' SARSA(0) implementation using our new algorithm of separating
    concerns. Both the uncertainty and the value function regarding
    the reward itself decay at a 1/t rate.'''
def SARSA_SPLIT_1oT_FIXED(b, numIterations, stepSize1, stepSize2, optimism):

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
		n[i] += 1

		reward      = b.pullArm(i)
		acumReturn += reward
		theta[i]    = theta[i] + stepSize1 * (1.0/(n[i]+1)) * (reward + 0 - theta[i])
		psi[i]      = psi[i]   + stepSize2                  * (0      + 0 - psi[i])

	return acumReturn

''' SARSA(0) implementation using our new algorithm of separating
    concerns. Both the uncertainty and the value function regarding
    the reward itself decay at a 1/t rate.'''
def SARSA_SPLIT_SQRT(b, numIterations, stepSize1, stepSize2, optimism):

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
		n[i] += 1

		reward      = b.pullArm(i)
		acumReturn += reward
		theta[i]    = theta[i] + stepSize1 * (1.0/(n[i]+1))                              * (reward + 0 - theta[i])
		psi[i]      = psi[i]   + stepSize2 * (1.0 - np.sqrt(float(n[i])/float(n[i]+1)))  * (0      + 0 - psi[i])

	return acumReturn

''' SARSA(0) implementation using our new algorithm of separating
    concerns. Both the uncertainty and the value function regarding
    the reward itself decay at a 1/t rate.'''
def SARSA_SPLIT_FIXED_SQRT(b, numIterations, stepSize1, stepSize2, optimism):

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
		n[i] += 1

		reward      = b.pullArm(i)
		acumReturn += reward
		theta[i]    = theta[i] + stepSize1                                               * (reward + 0 - theta[i])
		psi[i]      = psi[i]   + stepSize2 * (1.0 - np.sqrt(float(n[i])/float(n[i]+1)))  * (0      + 0 - psi[i])

	return acumReturn


def __init__():

	lvlOptimism   = RMAX
	numSeeds      = 2000
	numIterations = 1000

	#Variables that will store the results for all methods
	res_MAX              = []
	res_AVG              = []
	res_UCB              = []
	res_SARSA_P          = []
	res_SARSA_O          = []
	res_SARSA_1oT        = []
	res_SARSA_1oT1oT     = []
	res_SARSA_Fix1oT     = []
	res_SARSA_1oTFix     = []
	res_SARSA_1oT1oSqrtT = []
	res_SARSA_Fix1oSqrtT = []

	'''Already pull all the arms and store them, this way one can
	easily reproduce the results and come back to see other possibilities'''
	b = []
	for s in xrange(FIRST_SEED, numSeeds + FIRST_SEED):
		b.append(Bandits(s, numIterations))

	#Run the experiment x times, where x is the number of seeds
	for s in xrange(FIRST_SEED, numSeeds + FIRST_SEED):
		seed = s
		r.seed(seed)

		res_MAX.append([])
		res_AVG.append([])
		res_UCB.append([])
		res_SARSA_P.append([])
		res_SARSA_O.append([])
		res_SARSA_1oT.append([])
		res_SARSA_1oT1oT.append([])
		res_SARSA_Fix1oT.append([])
		res_SARSA_1oTFix.append([])
		res_SARSA_1oT1oSqrtT.append([])
		res_SARSA_Fix1oSqrtT.append([])

		maxReturn = 0

		'''First I just calculate the max return one could've get.'''
		for i in xrange(numIterations):
			maxReturn += max(b[s - FIRST_SEED].pullArm(0), b[s - FIRST_SEED].pullArm(1))
			
		res_MAX[s - FIRST_SEED].append(maxReturn)
		b[s - FIRST_SEED].resetEnv()

		'''Agent pulling the arm with highest running average.'''
		r.seed(seed)
		res_AVG[s - FIRST_SEED].append(AVG(b[s - FIRST_SEED], numIterations))
		b[s - FIRST_SEED].resetEnv()

		'''Agent following the UCB-1 algorithm.'''
		r.seed(seed)
		res_UCB[s - FIRST_SEED].append(UCB1(b[s - FIRST_SEED], numIterations, 1.0))
		b[s - FIRST_SEED].resetEnv()

		'''Agent following the SARSA(0) pessimistically initialized.'''
		r.seed(seed)
		res_SARSA_P[s - FIRST_SEED].append(SARSA(b[s - FIRST_SEED], numIterations, 0.05))
		b[s - FIRST_SEED].resetEnv()

		'''Agent following the SARSA(0) optimistically initialized.'''
		r.seed(seed)
		res_SARSA_O[s - FIRST_SEED].append(SARSA(b[s - FIRST_SEED], numIterations, 0.01, lvlOptimism))
		b[s - FIRST_SEED].resetEnv()

		'''Agent following the SARSA(0) with two value functions (same step-size).'''
		r.seed(seed)
		res_SARSA_1oT[s - FIRST_SEED].append(SARSA_SPLIT(b[s - FIRST_SEED], numIterations, 0.5, 0.5, lvlOptimism))
		b[s - FIRST_SEED].resetEnv()

		'''Agent following the SARSA(0) with two value functions (diff. step-size).'''
		r.seed(seed)
		res_SARSA_1oT1oT[s - FIRST_SEED].append(SARSA_SPLIT(b[s - FIRST_SEED], numIterations, 0.5, 0.5, lvlOptimism))
		b[s - FIRST_SEED].resetEnv()

		'''Agent following the SARSA(0) with two value functions (fixed step size and 1/t step size).'''
		r.seed(seed)
		res_SARSA_Fix1oT[s - FIRST_SEED].append(SARSA_SPLIT_FIXED_1oT(b[s - FIRST_SEED], numIterations, 0.005, 0.1, lvlOptimism))
		b[s - FIRST_SEED].resetEnv()

		'''Agent following the SARSA(0) with two value functions (1/t step size and fixed step size).'''
		r.seed(seed)
		res_SARSA_1oTFix[s - FIRST_SEED].append(SARSA_SPLIT_1oT_FIXED(b[s - FIRST_SEED], numIterations, 0.5, 0.05, lvlOptimism))
		b[s - FIRST_SEED].resetEnv()

		'''Agent following the SARSA(0) with two value functions (one decays at 1/t and the other 1/sqrt(t)).'''
		r.seed(seed)
		res_SARSA_1oT1oSqrtT[s - FIRST_SEED].append(SARSA_SPLIT_SQRT(b[s - FIRST_SEED], numIterations, 0.5, 0.5, lvlOptimism))
		b[s - FIRST_SEED].resetEnv()

		'''Agent following the SARSA(0) with two value functions (one is fixedand the other decays at 1/sqrt(t)).'''
		r.seed(seed)
		res_SARSA_Fix1oSqrtT[s - FIRST_SEED].append(SARSA_SPLIT_FIXED_SQRT(b[s - FIRST_SEED], numIterations, 0.001, 0.05, lvlOptimism))
		b[s - FIRST_SEED].resetEnv()

	'''Now we can take the average return of each method:'''
	res_MAX_avg = []
	res_AVG_avg = []
	res_UCB_avg = []
	res_SARSA_P_avg = []
	res_SARSA_O_avg = []
	res_SARSA_1oT_avg = []
	res_SARSA_1oT1oT_avg = []
	res_SARSA_Fix1oT_avg = []
	res_SARSA_1oTFix_avg = []
	res_SARSA_1oT1oSqrtT_avg = []
	res_SARSA_Fix1oSqrtT_avg = []

	for i in xrange(numSeeds):
		res_MAX_avg.append(stat.mean(res_MAX[i]))
		res_AVG_avg.append(stat.mean(res_AVG[i]))
		res_UCB_avg.append(stat.mean(res_UCB[i]))
		res_SARSA_P_avg.append(stat.mean(res_SARSA_P[i]))
		res_SARSA_O_avg.append(stat.mean(res_SARSA_O[i]))
		res_SARSA_1oT_avg.append(stat.mean(res_SARSA_1oT[i]))
		res_SARSA_1oT1oT_avg.append(stat.mean(res_SARSA_1oT1oT[i]))
		res_SARSA_Fix1oT_avg.append(stat.mean(res_SARSA_Fix1oT[i]))
		res_SARSA_1oTFix_avg.append(stat.mean(res_SARSA_1oTFix[i]))
		res_SARSA_1oT1oSqrtT_avg.append(stat.mean(res_SARSA_1oT1oSqrtT[i]))
		res_SARSA_Fix1oSqrtT_avg.append(stat.mean(res_SARSA_Fix1oSqrtT[i]))

	print 'Max return                   :', stat.mean(res_MAX_avg), ',', stat.stdev(res_MAX_avg)
	print 'AVG                          :', stat.mean(res_AVG_avg), ',', stat.stdev(res_AVG_avg)
	print 'UCB-1                        :', stat.mean(res_UCB_avg), ',', stat.stdev(res_UCB_avg)
	print 'SARSA(0) -- pess.            :', stat.mean(res_SARSA_P_avg), ',', stat.stdev(res_SARSA_P_avg)
	print 'SARSA(0) -- opt.             :', stat.mean(res_SARSA_O_avg), ',', stat.stdev(res_SARSA_O_avg)
	print 'SARSA(0) -- 1/t              :', stat.mean(res_SARSA_1oT_avg), ',', stat.stdev(res_SARSA_1oT_avg)
	print 'SARSA(0) -- 1/t, 1/t         :', stat.mean(res_SARSA_1oT1oT_avg), ',', stat.stdev(res_SARSA_1oT1oT_avg)
	print 'SARSA(0) -- Fixed, 1/t       :', stat.mean(res_SARSA_Fix1oT_avg), ',', stat.stdev(res_SARSA_Fix1oT_avg)
	print 'SARSA(0) -- 1/t, Fixed       :', stat.mean(res_SARSA_1oTFix_avg), ',', stat.stdev(res_SARSA_1oTFix_avg)
	print 'SARSA(0) -- 1/t, 1/SQRT(t)   :', stat.mean(res_SARSA_1oT1oSqrtT_avg), ',', stat.stdev(res_SARSA_1oT1oSqrtT_avg)
	print 'SARSA(0) -- Fixed, 1/SQRT(t) :', stat.mean(res_SARSA_Fix1oSqrtT_avg), ',', stat.stdev(res_SARSA_Fix1oSqrtT_avg)

__init__()
