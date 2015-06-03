import argparse

parser = argparse.ArgumentParser(description='Generates the proper PBS files to be used, as well as the shell script that executes them.')

parser.add_argument('--walltime', metavar='t', dest='walltime', 
	action='store', default=None, help='time to be allocated in the machine (format: HHH:MM:SS)')

parser.add_argument('--memory', metavar='m', type=int, dest='memory', 
	action='store', default=1024, help='amount of megabytes to be allocated (default: 1024)')

parser.add_argument('--num_seeds', metavar='s', type=int, dest='num_seeds', 
	action='store', default=30, help='num seeds to be evaluated (default: 30)')

parser.add_argument('--config_file', metavar='c', dest='config_file', 
	action='store', default=None, help='path and filename to the configuration file (.cfg)')

parser.add_argument('--rom_file', metavar='r', dest='rom_file', 
	action='store', default=None, help='path and filename to the rom file')

parser.add_argument('--weights_save', metavar='w', dest='weights_save', 
	action='store', default=None, help='path and filename prefix to be concatenated with the seed to save the weights at a pre-defined sequence (optional).')

parser.add_argument('--weights_load', metavar='l', dest='weights_load', 
	action='store', default=None, help='path and filename to the file containing the weights to be loaded (optional).')

parser.add_argument('--output_file', metavar='o', dest='output_file', 
	action='store', default=None, help='path and filename to the output file of the execution (the seed will be concatenated at the end).')

parser.add_argument('--pbs_filename', metavar='p', dest='pbs_filename', 
	action='store', default=None, help='path and filename to the pbs files to be generated.')

parser.add_argument('--script_to_submit', metavar='t', dest='script_to_submit', 
	action='store', default=None, help='path and filename to the script that will contain the submission instructions.')

args = parser.parse_args()

if args.walltime == None:
	parser.error("The parameter --walltime must be informed.")

if args.config_file == None:
	parser.error("The parameter --config_file must be informed.")

if args.rom_file == None:
	parser.error("The parameter --rom_file must be informed.")

if args.output_file == None:
	parser.error("The parameter --output_file must be informed.")

if args.pbs_filename == None:
	parser.error("The parameter --pbs_filename must be informed.")

if args.script_to_submit == None:
	parser.error("The parameter --script_to_submit must be informed.")

for s in xrange(args.num_seeds):
	f = open(args.pbs_filename + str(s + 1), 'w')
	f.write('#!/bin/bash\n')
	f.write('#PBS -S /bin/bash\n')

	f.write('# Script for running\n\n')

	f.write('#PBS -l nodes = 1\n')
	f.write('#PBS -l mem = ' + str(args.memory) + 'mb\n')
	f.write('#PBS -l walltime = ' + str(args.walltime) + '\n')
	f.write('#PBS -M machado@cs.ualberta.ca\n')

	f.write('LD_LIBRARY_PATH=../../MyALE/\n')
	f.write('export LD_LIBRARY_PATH\n\n')

	f.write('cd $PBS_O_WORKDIR\n\n')

	f.write('./learner -c ' + str(args.config_file) + ' -r ' + str(args.rom_file) + ' -s ' + str(s))

	if args.weights_save != None:
		f.write(' -w ' + str(args.weights_save))
	if args.weights_load != None:
		f.write(' -l ' + str(args.weights_load))

	f.write(' > results/enduro_' + str(s + 1) + '.out')
f.close()

f = open(args.script_to_submit, 'w')
for s in xrange(args.num_seeds):
	f.write('bqsub ' + args.pbs_filename + str(s+1) + '\n')

f.write('bqsub --submit')
f.close()