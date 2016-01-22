import sys
import statistics as stats


fname = sys.argv[1]

with open(fname) as f:
    results = f.readlines()

numSeeds = 30
gamma = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
alpha = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
lambd = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]

finalResults = [[],[]]
intermediateResults = []

i = 0
while i < len(results):
	for a1 in alpha:
		for l1 in lambd:
			for a2 in alpha:
				for l2 in lambd:
					for g in gamma:
						for s in xrange(numSeeds):
							intermediateResults.append(float(results[i].split(' ')[4].split(',')[0]))
							i += 1
						finalResults[0].append(stats.mean(intermediateResults)) 
						finalResults[1].append(stats.stdev(intermediateResults))
						intermediateResults = []

maximum = max(finalResults[0])
print maximum
'''

i = 0
while i < len(results):
	for a1 in alpha:
		for l1 in lambd:
			for s in xrange(numSeeds):
				intermediateResults.append(float(results[i].split(' ')[4].split(',')[0]))
				i += 1

			finalResults[0].append(stats.mean(intermediateResults)) 
			finalResults[1].append(stats.stdev(intermediateResults))
			intermediateResults = []

maximum = max(finalResults[0])
'''
#Print table with results:
'''
idx = 0
for g in gamma:
	for a2 in alpha:
		for l2 in lambd:
			sys.stdout.write('\n\ngamma: ' + str(g) + ', alpha: ' + str(a2) + ', lambda: ' + str(l2) + '\n')
			sys.stdout.write('alpha/lambda,')
			for l in lambd:
				sys.stdout.write(str(l) + ', ,')
			sys.stdout.write('\n')

			for a in alpha:
				sys.stdout.write(str(a) + ',')
				for l in lambd:
#					if finalResults[0][idx] == maximum:
#						print
#						print a1, a2, g, l, l2
#						sys.stdin.read(1)

					sys.stdout.write(str(finalResults[0][idx]) + ',' + str(finalResults[1][idx]) + ',')
					idx += 1

				sys.stdout.write('\n')

print
print maximum
'''

idx = 0
for a1 in alpha:
	for l1 in lambd:
		for a2 in alpha:
			for l2 in lambd:
				for g in gamma:
					print a1, l1, a2, l2, g, str(finalResults[0][idx]), str(finalResults[1][idx])
					idx += 1

print maximum

'''
for a in alpha:
	sys.stdout.write(str(a) + ',')
	for l in lambd:
		sys.stdout.write(str(finalResults[0][idx]) + ',' + str(finalResults[1][idx]) + ',')
		idx += 1

	sys.stdout.write('\n')

print
print maximum
'''