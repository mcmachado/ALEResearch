import sys
import numpy as np
from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold='nan')

'''This program receives a file containing a feature vector per line. Each line has the
	indices that are different from zero when flipping. The first X are 0->1 flips and
	the last X are 1->0 flips. It then runs an SVD on them, dumping the top k eigenvectors
	with one element per line on a file. It also prints out the centering vector.'''

if __name__ == "__main__":

	#########################################
	### Deal with command line/parameters ###
	#########################################
	if len(sys.argv) != 5:
		print 'please run \'python ' + sys.argv[0] + ' <input_file> <num_features> <k> <output_file>\''
		sys.exit(1)
	inputFile      = sys.argv[1]
	numFeatures    = int(sys.argv[2])
	numToDump      = int(sys.argv[3])
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
	
	X_std = StandardScaler().fit_transform(Data)
	Stats = StandardScaler().fit(Data)
	mean  = Stats.mean_
	std   = Stats.std_

	cov_mat = np.cov(X_std.T)
	
	U, eig_vals, eig_vecs = np.linalg.svd(cov_mat)

	#############################
	### Print top eigenvector ###
	#############################
	for i in xrange(numToDump):
		f = open(outputFile + '_' + str(i) + '.out', 'w')
		for j in eig_vecs[i]:
			f.write(str(j) + '\n')

	################################
	### Print stats to be loaded ###
	################################

	f = open(outputFile + '_std.out', 'w')
	for i in std:
		f.write(str(i) + '\n')

	f = open(outputFile + '_mean.out', 'w')
	for i in mean:
		f.write(str(i) + '\n')
