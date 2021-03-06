This directory contain all the code one is required to run the results presented in the paper

[Reference].

As described in the paper, each iteration starts as follows:

1. The rare events are identified through the features. This is done in folder gen_samples
	through genSamples. One can use it to generate events loading weights (corresponding
	to individual options) or not (only from primitive actions):

	./genSamples -s <SEED> -r <ROM> -t <ACTIONS_FILE> -n <FREQ_THRESHOLD> -o <OUTPUT_FILE_PREFIX> -c <REPORT_IRR_TRANS> -n <NUM_OPTIONS> <OPTION_1> <OPTION_2> ... <OPTION_N>


2. Once the rare events are generated, the output is given as input to the SVD. The SVD will generate
	N + 1 outputs: the top-N eigenvectors and the two vectors required to center future data: the 
	mean and the variance of each individual coordinate. To obtain it you have to run:

	python SVDonRareEvents.py <input_file> <num_features> <k> <output_file>


3. Finally, once we have the eigenvectors that will generate our reward for each of the individual
	options, we can now learn policies that maximize each of these rewards. I call each of these
	policies an option:

	./learnOption -s <SEED> -r <ROM> -t <EIGENVECTOR> -i <STATS_SVD> -o <OUTPUT_FILE> -n <NUM_OPTIONS> <OPTION_1> <OPTION_2> ... <OPTION_N>