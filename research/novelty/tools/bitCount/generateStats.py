import sys
import csv
import math
from scipy import spatial

#Reading CSV file and storing it in data
def readCSV(fileName):
	data     = []
	#str_data = []

	with open(fileName, 'r') as f:
	    reader = csv.reader(f)
	    for row in reader:
	    	data.append(row)
	#    	str_data.append(''.join(row))
	return data

def collectStatistics(data):
	totalNumFlips = 0
	totalNumSteps = len(data)
	indivNumFlips = 1025 * [0.0]

	for i, val in enumerate(data):
		for j, val in enumerate(data[i]):
			if i < len(data) - 1:
				if data[i][j] != data[i+1][j]:
					totalNumFlips += 1
					indivNumFlips[j] += 1.0

	avgNumFlips = float(totalNumFlips)/float(totalNumSteps)
	#numUniqStates = len(set(str_data))

	return totalNumFlips, totalNumSteps, avgNumFlips, indivNumFlips

def printData(human, random, options, numSeeds, numOptions):
	for i in xrange(numSeeds):
		sys.stdout.write('seed ' + str(i+1) + ', ')
		for j in xrange(numOptions):
			sys.stdout.write(str(options[j][i]) + ', ')	
		sys.stdout.write(str(random[i]) + ', ')
		sys.stdout.write(str(human[i]) + '\n')
		

	sys.stdout.write('average, ')
	for j in xrange(numOptions):
		sys.stdout.write(str(sum(options[j])/len(options[j])) + ', ')
	sys.stdout.write(str(sum(human)/len(human)) + ', ')
	sys.stdout.write(str(sum(random)/len(random)) + '\n')

def getEuclDistance(v1, v2):
	numFeatures = 1025
	sum_v1 = sum(v1)
	sum_v2 = sum(v2)
	norm_v1 = []
	norm_v2 = []

	euclDist = 0
	for i in xrange(numFeatures):
		norm_v1.append(v1[i]/sum_v1)
		norm_v2.append(v2[i]/sum_v2)
		euclDist += (norm_v1[i] - norm_v2[i]) * (norm_v1[i] - norm_v2[i])

	return math.sqrt(euclDist)

def printDistances(euclDistanceHumanRandom, euclDistanceHumanOptions, numSeeds, numOptions):
	for i in xrange(numOptions):
		sys.stdout.write(str(sum(euclDistanceHumanOptions[i])/len(euclDistanceHumanOptions[i])) + ', ')

	sys.stdout.write(str(sum(euclDistanceHumanRandom)/len(euclDistanceHumanRandom)) + '\n')



def main():

	numSeeds = 5
	## The first step is to read the parameters from the command line
	if len(sys.argv) != 5:
		print 'Usage: python', sys.argv[0], 'prefix_human_traj prefix_random_traj prefix_options_traj numOptions'
		exit(1)

	prefixHuman   = sys.argv[1]
	prefixRandom  = sys.argv[2]
	prefixOptions = sys.argv[3]
	numOptions    = int(sys.argv[4])

	## Then I need to save all trajectories in a huge matrix:
	dataTmpHuman = []
	dataRandom   = []
	dataOptions  = []
	for i in xrange(numSeeds):
		sys.stdout.write('.')
		sys.stdout.flush()
		dataTmpHuman.append(readCSV(prefixHuman + str(i+1) + '.out'))
		sys.stdout.write('.')
		sys.stdout.flush()
		dataRandom.append(readCSV(prefixRandom + str(i+1) + '.out'))
		for j in xrange(numOptions):
			sys.stdout.write('.')
			sys.stdout.flush()
			dataOptions.append([])
			dataOptions[j].append(readCSV(prefixOptions + str(j+1) + str(i+1) + '.out'))

	## The problem is that they are being saved as strings, I'll convert everything to float:
	for i in xrange(numSeeds):
		sys.stdout.write('.')
		sys.stdout.flush()
		for j in xrange(len(dataTmpHuman[i])):
			for k in xrange(len(dataTmpHuman[i][j])):
				if dataTmpHuman[i][j][k] != '':
					dataTmpHuman[i][j][k] = float(dataTmpHuman[i][j][k])
				else:
					dataTmpHuman[i][j].pop(k)

		sys.stdout.write('.')
		sys.stdout.flush()
		for j in xrange(len(dataRandom[i])):
			for k in xrange(len(dataRandom[i][j])):
				if dataRandom[i][j][k] != '':
					dataRandom[i][j][k] = float(dataRandom[i][j][k])
				else:
					dataRandom[i][j].pop(k)

		for j in xrange(len(dataOptions[i])):
			sys.stdout.write('.')
			sys.stdout.flush()			
			for k in xrange(len(dataOptions[i][j])):
				for r in xrange(len(dataOptions[i][j][k])):
					if dataOptions[i][j][k][r] != '':
						dataOptions[i][j][k][r] = float(dataOptions[i][j][k][r])
					else:
						dataOptions[i][j][k].pop(r)

	## Also, in the human trajectories I was not using frame skip, so I have to force it later:
	dataHuman = []
	for i in xrange(numSeeds):
		dataHuman.append([])
		for linha in xrange(len(dataTmpHuman[i])):
			if linha % 5 == 0:
				dataHuman[i].append(dataTmpHuman[i][linha])

	## Now I can finally collect the statistics:
	avgNumFlipsHuman = []
	avgNumFlipsRandom = []
	avgNumFlipsOptions = []
	indivFlipsHuman = []
	indivFlipsRandom = []
	indivFlipsOptions = []
	for i in xrange(numSeeds):
		sys.stdout.write('.')
		sys.stdout.flush()
		totalNumFlips, totalNumSteps, avgNumFlips, indivNumFlips = collectStatistics(dataHuman[i])
		indivFlipsHuman.append(indivNumFlips)
		avgNumFlipsHuman.append(avgNumFlips)
		sys.stdout.write('.')
		sys.stdout.flush()
		totalNumFlips, totalNumSteps, avgNumFlips, indivNumFlips = collectStatistics(dataRandom[i])
		indivFlipsRandom.append(indivNumFlips)
		avgNumFlipsRandom.append(avgNumFlips)
		for j in xrange(numOptions):
			sys.stdout.write('.')
			sys.stdout.flush()
			avgNumFlipsOptions.append([])
			indivFlipsOptions.append([])
			totalNumFlips, totalNumSteps, avgNumFlips, indivNumFlips = collectStatistics(dataOptions[j][i])
			indivFlipsOptions[j].append(indivNumFlips)
			avgNumFlipsOptions[j].append(avgNumFlips)

	print
	print '\nAverage number of flips:'
	printData(avgNumFlipsHuman, avgNumFlipsRandom, avgNumFlipsOptions, numSeeds, numOptions)
	print

	cosDistanceHumanRandom   = []
	euclDistanceHumanRandom  = []
	cosDistanceHumanOptions  = [list([]) for _ in xrange(numOptions)]
	euclDistanceHumanOptions = [list([]) for _ in xrange(numOptions)]

	for i in xrange(numSeeds):
		for j in xrange(numSeeds):
			if i >= j:
				euclDistanceHumanRandom.append(getEuclDistance(indivFlipsHuman[i], indivFlipsRandom[j]))
				cosDistanceHumanRandom.append(spatial.distance.cosine(indivFlipsHuman[i], indivFlipsRandom[j]))

	for i in xrange(numSeeds):
		for j in xrange(numSeeds):
			if i >= j:
				for k in xrange(numOptions):
					euclDistanceHumanOptions[k].append(getEuclDistance(indivFlipsHuman[i], indivFlipsOptions[k][j]))
					cosDistanceHumanOptions[k].append(spatial.distance.cosine(indivFlipsHuman[i], indivFlipsOptions[k][j]))

	print 'Average euclidian distance of frequency of flips from human trajectories:'
	printDistances(euclDistanceHumanRandom, euclDistanceHumanOptions, numSeeds, numOptions)
	print

	print 'Average cosine between frequency of flips of agents and human trajectories:'
	printDistances(cosDistanceHumanRandom, cosDistanceHumanOptions, numSeeds, numOptions)
	print

if __name__ == "__main__":
    main()