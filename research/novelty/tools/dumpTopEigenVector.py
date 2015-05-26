import sys
import numpy as np
from sklearn.preprocessing import StandardScaler

'''This program receives a file containing a feature vector per line. Each line has the
	indices that are different from zero when flipping. The first X are 0->1 flips and
	the last X are 1->0 flips. It then runs an SVD on them, dumping the top eigenvector
	with one element per line on a file.'''

if __name__ == "__main__":

	#########################################
	### Deal with command line/parameters ###
	#########################################
	if len(sys.argv) != 5:
		print 'please run \'python dumpTopEigenVector.py <input_file> <num_features> <center_matrix> <output_file>\''
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

	#############################
	### Print top eigenvector ###
	#############################
	f = open(outputFile + '.out', 'w')
	for i in eig_vecs[0]:
		f.write(str(i) + '\n')
