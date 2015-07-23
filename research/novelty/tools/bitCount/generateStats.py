import sys
import csv

## Reading the input
if len(sys.argv) != 2:
	print 'Usage: python', sys.argv[0], 'file_to_read'
	exit(1)
inputFile = sys.argv[1]

#Reading CSV file and storing it in data
data     = []
str_data = []

with open(inputFile, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
    	data.append(row)
    	str_data.append(''.join(row))

#Now I can finally collect the statistics:
totalNumFlips = 0
totalNumSteps = len(data)
indivNumFlips = 1024 * [0]

for i, val in enumerate(data):
	for j, val in enumerate(data[i]):
		if i < len(data) - 1:
			if data[i][j] != data[i+1][j]:
				totalNumFlips += 1
				indivNumFlips[j] += 1

avgNumFlips = float(totalNumFlips)/float(totalNumSteps)
numUniqStates = len(set(str_data))

print "file , totalNumFlips , totalNumSteps , avgNumFlips , numUniqStates"
print inputFile, ",", totalNumFlips, ",", totalNumSteps, ",", avgNumFlips, ",", numUniqStates

#for i, val in enumerate(indivNumFlips):
#	indivNumFlips[i] = float(indivNumFlips[i])/float(totalNumSteps)

#print indivNumFlips