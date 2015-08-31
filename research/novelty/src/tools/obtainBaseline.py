import sys

prefix    = sys.argv[1]
maxFrames = int(sys.argv[2])

scores    = []
numFrames = []

for i in xrange(1,25):
	prevNumFrames = 0
	scores.append([])
	numFrames.append([])
	fname = prefix + str(i) + '.out'
	with open(fname) as f:
		content = f.readlines()
		lineNumber = 0
		for k in content:
			lineNumber += 1
			stop = False
			if lineNumber > 16 and not stop:
				if k == '':
					stop = True
				score = int(((((k.split(','))[1].split('\t'))[1]).split(' '))[0])
				scores[i-1].append(score)
				currentFrames = int(((((k.split(','))[3].split('\t'))[1]).split(' '))[0])
				numFrames[i-1].append(prevNumFrames + currentFrames)
				prevNumFrames += currentFrames


	lastValidEpisode = -1
	for j in xrange(len(numFrames[i-1])):
		if numFrames[i-1][j] > maxFrames and lastValidEpisode == -1:
			lastValidEpisode = j

	for j in xrange(5):
		print scores[i-1][lastValidEpisode - j]