import sys
import matplotlib.pyplot as plt
import numpy as np

fileName = sys.argv[1]

matrix = []

for i in xrange(30):
	data = []
	with open(fileName + str(i+1) + ".out") as f:
	    content = f.readlines()

	for j in content:
		data.append(int(j))

	matrix.append(data)


np_matrix = np.column_stack((matrix[0], matrix[1]))

for i in xrange(2,30):
	np_matrix = np.column_stack((np_matrix, matrix[i]))

#Freeway: plt.ylim([1,250])
plt.ylim([1,30])

plt.pcolor(np_matrix)
#Freeway: plt.clim(0,100)
plt.clim(0,200)
#Freeway: plt.title('Freeway - Chicken\'s height across games')
plt.title('Private Eye - Screen visitation frequency')
plt.xlabel('Game #')
#Freeway: plt.ylabel('Height')
plt.ylabel('Screen #')
#plt.show()
plt.savefig('options-5.pdf')