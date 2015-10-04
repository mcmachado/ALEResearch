import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

toPlot = int(sys.argv[1])
numFiles = 5

c = []

for i in xrange(1,numFiles):
	fname = sys.argv[i + 1]
	with open(fname) as f:
	    content = f.readlines()

	int_content = []

	for j in content:
		int_content.append(int(j))

	previous = -1
	for idx, val in enumerate(int_content):
		#print int_content[idx-1], int_content[idx], int_content[idx-1] != int_content[idx]
		if int_content[idx-1] != int_content[idx]:
			previous = int_content[idx-1]

		if int_content[idx] == 28 and (previous == 27 or previous == 33):
			int_content[idx] = 33
		if int_content[idx] == 15 and (previous == 16 or previous == 34):
			int_content[idx] = 34

	c.append(Counter(int_content))

numVisitations = []
for i in xrange(1, numFiles):
	z = c[i-1]
	numVisitations.append(z[toPlot])


#print numVisitations

fig = plt.figure()
ax  = plt.subplot(111)

ax.bar(1, numVisitations[0], width=1, color='b', label='random')
ax.bar(2, numVisitations[1], width=1, color='g', label=r'opt. init. ($\gamma = 0.99$)')
ax.bar(3, numVisitations[2], width=1, color='y', label=r'opt. init. ($\gamma = 0.999$)')
ax.bar(4, numVisitations[3], width=1, color='r', label=r'option')

ax.set_ylim([0,25000])
ax.set_xlim([0,5.5])
plt.axis('off')

#plt.legend()
#plt.show()
splt.savefig(str(toPlot) + '.pdf')