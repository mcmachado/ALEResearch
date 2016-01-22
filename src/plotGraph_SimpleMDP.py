import matplotlib.pyplot as plt
import numpy as np

data = []
numSteps = 1000

for i in xrange(100):
	fname = 'temp_simpleMDP/' + str(i+1) + '_1oT_global.txt'
	with open(fname) as f:
		content = f.readlines()

	data.append([])
	for line in content:
		data[i].append(line.split())

all_data = []

for i in xrange(4):
	all_data.append([])
	for j in xrange(numSteps):
		all_data[i].append([])
		for k in xrange(100):
			all_data[i][j].append(float(data[k][j][i]))

means   = []
std_dev = []
min_between = []
max_between = []

for i in xrange(4):
	means.append([])
	std_dev.append([])
	min_between.append([])
	max_between.append([])
	for j in xrange(numSteps):
		means[i].append(np.mean(all_data[i][j]))
		std_dev[i].append(np.std(all_data[i][j]))
		min_between[i].append(means[i][j] - 1.96 * (std_dev[i][j]/100))
		max_between[i].append(means[i][j] + 1.96 * (std_dev[i][j]/100))


toPlot = []
for i in xrange(4):
	toPlot.append([])
	for j in xrange(numSteps):
		toPlot[i].append(means[i][j]/(means[0][j] + means[1][j] + means[2][j] + means[3][j]))


plt.plot(toPlot[0], label = 'State 1', linewidth= 2.0, color='blue')
#plt.plot(means[0], label = 'State 1', linewidth= 2.0, color='blue')
plt.plot(toPlot[1], label = 'State 2', linewidth= 2.0, color='green')
#plt.plot(means[1], label = 'State 2', linewidth= 2.0, color='green')
plt.plot(toPlot[2], label = 'State 3', linewidth= 2.0, color='red')
#plt.plot(means[2], label = 'State 3', linewidth= 2.0, color='red')
plt.plot(toPlot[3], label = 'State 4', linewidth= 2.0, color='black')
#plt.plot(means[3], label = 'State 4', linewidth= 2.0, color='black')
plt.legend()
#plt.ylim([0.249,.2515])
plt.savefig('graph_normalized_1oT_global.pdf')