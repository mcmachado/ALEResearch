import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=np.nan)
eigenvalueThreshold = 0.1
eigenvectorThreshold = 0.05

'''This program receives a file containing a feature vector per line. Each line has the
	indices that are different from zero when flipping. The first X are 0->1 flips and
	the last X are 1->0 flips. It then runs an SVD on them, reporting the highest
	eigenvalues and their respective eigenvectors.'''

if __name__ == "__main__":

	#########################################
	### Deal with command line/parameters ###
	#########################################
	if len(sys.argv) != 5:
		print 'please run \'python SVD.py <input_file> <num_features> <center_matrix> <output_prefix>\''
		sys.exit(1)
	inputFile      = sys.argv[1]
	numFeatures    = int(sys.argv[2])
	toCenterMatrix = sys.argv[3]
	if toCenterMatrix == '1':
		toCenterMatrix = True
	elif toCenterMatrix == '0':
		toCenterMatrix = False
	else:
		print 'The third parameter has an invalid value'
		sys.exit(1)
	outputFile     = sys.argv[4]

	#####################################
	### Read file passed as parameter ###
	#####################################
	Data = []
	with open(inputFile, 'rb') as f:
		content = f.readlines()
		#For each row:
		for row in content:
			#Split string in commas and save the values:
			tmp = np.zeros(2 * numFeatures)
			indices = row.strip().split(",")
			for elem in indices:
				if elem != '':
					tmp[int(elem)] = 1.0
			Data.append(tmp)
	Data = np.array(Data, dtype=np.uint8)

    ###########################
	### Really runs the SVD ###
	###########################
	if toCenterMatrix:
		X_std = StandardScaler().fit_transform(Data)
		cov_mat = np.cov(X_std.T)
	else:
		cov_mat = np.cov(Data.T)

	U, eig_vals, eig_vecs = np.linalg.svd(cov_mat)


	#####################
	### Print results ###
	#####################
	numInterestingEigValues = 0
	f = open(outputFile + '_eig.out', 'w')
	for counter, val in enumerate(eig_vals):
		if val > eigenvalueThreshold:
			numInterestingEigValues += 1
			f.write('\n' + str(counter + 1) + ') ' + str(val) + ' [')
			for idx, e in enumerate(eig_vecs[counter]):
				if abs(round(e, 2)) > eigenvectorThreshold:
					f.write(str(idx) + ':' + str(round(e, 2)) + ', ')
			f.write(']\n')

	###################
	### Plot graphs ###
	###################
	plt.xlabel('Eigenvalue #')
	plt.xlim([-1, numInterestingEigValues + 1])
	plt.ylabel('Eigenvalue')
	plt.title(outputFile)
	plt.plot(xrange(numInterestingEigValues + 1), eig_vals[0:numInterestingEigValues + 1], marker='o', linestyle = ' ')
	plt.savefig(outputFile + '.pdf')