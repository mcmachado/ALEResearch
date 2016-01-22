import os
gamma = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
alpha = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
lambd = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]

#print 'SARSA_FIXED_PESSIMISTIC'
#for a in alpha:
#	for l in lambd:
#		os.system("./sarsa_fixed -s 1 -a " + str(a) + " -b 0.01 -o 0.99 -e " + str(l) + " -u 0.9 -c ../conf/rs.cfg -r IdontCare >> results_sarsa_fixed_pessimistic.txt")

#print 'SARSA_FIXED_OPTIMISTIC'
#for a in alpha:
#	for l in lambd:
#		os.system("./sarsa_fixed -s 1 -a " + str(a) + " -b 0.01 -o 0.99 -e " + str(l) + " -u 0.9 -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_fixed_optimistic.txt")


#print 'SARSA_1oT_OPTIMISTIC'
#for a in alpha:
#	for l in lambd:
#		os.system("./sarsa_1oT -s 1 -a " + str(a) + " -b 0.01 -o 0.99 -e " + str(l) + " -u 0.9 -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_optimistic.txt")


#print 'SARSA_1oT_1oT'
#for a1 in alpha:
#	for l1 in lambd:
#		for a2 in alpha:
#			for l2 in lambd:
#				for g in gamma:
#					os.system("./sarsa_1oT_1oT -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_1oT.txt")


#print 'SARSA_FIXED_1oT'
#for a1 in alpha:
#	for l1 in lambd:
#		for a2 in alpha:
#			for l2 in lambd:
#				for g in gamma:
#					os.system("./sarsa_fixed_1oT -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_fixed_1oT.txt")


#print 'SARSA_1oT_FIXED'
#for a1 in alpha:
#	for l1 in lambd:
#		for a2 in alpha:
#			for l2 in lambd:
#				for g in gamma:
#					os.system("./sarsa_1oT_fixed -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_fixed.txt")


#print 'SARSA_1oT_1oSQRTT'
#for a1 in alpha:
#	for l1 in lambd:
#		for a2 in alpha:
#			for l2 in lambd:
#				for g in gamma:
#					os.system("./sarsa_1oT_1oSqrtT -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_1oSqrtT.txt")

#print 'SARSA_FIXED_1oSQRTT'
#for a1 in alpha:
#	for l1 in lambd:
#		for a2 in alpha:
#			for l2 in lambd:
#				for g in gamma:
#					os.system("./sarsa_fixed_1oSqrtT -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare  >> results_sarsa_fixed_1oSqrtT.txt")


####################

#print 'SARSA_FIXED_PESSIMISTIC'
#for a in alpha:
#	for l in lambd:
#		os.system("./sarsa_fixed_start -s 1 -a " + str(a) + " -b 0.01 -o 0.99 -e " + str(l) + " -u 0.9 -c ../conf/rs.cfg -r IdontCare >> results_sarsa_fixed_pessimistic_start.txt")

#print 'SARSA_FIXED_OPTIMISTIC'
#for a in alpha:
#	for l in lambd:
#		os.system("./sarsa_fixed_start -s 1 -a " + str(a) + " -b 0.01 -o 0.99 -e " + str(l) + " -u 0.9 -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_fixed_optimistic_start.txt")


#print 'SARSA_1oT_OPTIMISTIC'
#for a in alpha:
#	for l in lambd:
#		os.system("./sarsa_1oT_start -s 1 -a " + str(a) + " -b 0.01 -o 0.99 -e " + str(l) + " -u 0.9 -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_optimistic_start.txt")


#print 'SARSA_1oT_1oT'
#for a1 in alpha:
#	for l1 in lambd:
#		for a2 in alpha:
#			for l2 in lambd:
#				for g in gamma:
#					os.system("./sarsa_1oT_1oT_start -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_1oT_start.txt")


#print 'SARSA_FIXED_1oT'
#for a1 in alpha:
#	for l1 in lambd:
#		for a2 in alpha:
#			for l2 in lambd:
#				for g in gamma:
#					os.system("./sarsa_fixed_1oT_start -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_fixed_1oT_start.txt")


#print 'SARSA_1oT_FIXED'
#for a1 in alpha:
#	for l1 in lambd:
#		for a2 in alpha:
#			for l2 in lambd:
#				for g in gamma:
#					os.system("./sarsa_1oT_fixed_start -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_fixed_start.txt")


#print 'SARSA_1oT_1oSQRTT'
#for a1 in alpha:
#	for l1 in lambd:
#		for a2 in alpha:
#			for l2 in lambd:
#				for g in gamma:
#					os.system("./sarsa_1oT_1oSqrtT_start -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_1oSqrtT_start.txt")

#print 'SARSA_FIXED_1oSQRTT'
#for a1 in alpha:
#	for l1 in lambd:
#		for a2 in alpha:
#			for l2 in lambd:
#				for g in gamma:
#					os.system("./sarsa_fixed_1oSqrtT_start -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_fixed_1oSqrtT_start.txt")

####################

#print 'SARSA_FIXED_PESSIMISTIC'
#for a in alpha:
#	for l in lambd:
#		os.system("./sarsa_fixed_bias -s 1 -a " + str(a) + " -b 0.01 -o 0.99 -e " + str(l) + " -u 0.9 -c ../conf/rs.cfg -r IdontCare >> results_sarsa_fixed_pessimistic_bias.txt")


#print 'SARSA_FIXED_OPTIMISTIC'
#for a in alpha:
#	for l in lambd:
#		os.system("./sarsa_fixed_bias -s 1 -a " + str(a) + " -b 0.01 -o 0.99 -e " + str(l) + " -u 0.9 -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_fixed_optimistic_bias.txt")


#print 'SARSA_1oT_OPTIMISTIC'
#for a in alpha:
#	for l in lambd:
#		os.system("./sarsa_1oT_bias -s 1 -a " + str(a) + " -b 0.01 -o 0.99 -e " + str(l) + " -u 0.9 -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_optimistic_bias.txt")


#print 'SARSA_1oT_1oT'
#for a1 in alpha:
#	for l1 in lambd:
#		for a2 in alpha:
#			for l2 in lambd:
#				for g in gamma:
#					os.system("./sarsa_1oT_1oT_bias -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_1oT_bias.txt")


#print 'SARSA_FIXED_1oT'
for a1 in alpha:
	for l1 in lambd:
		for a2 in alpha:
			for l2 in lambd:
				for g in gamma:
					os.system("./sarsa_fixed_1oT_bias -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_fixed_1oT_bias.txt")


#print 'SARSA_1oT_FIXED'
for a1 in alpha:
	for l1 in lambd:
		for a2 in alpha:
			for l2 in lambd:
				for g in gamma:
					os.system("./sarsa_1oT_fixed_bias -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_fixed_bias.txt")


#print 'SARSA_1oT_1oSQRTT'
for a1 in alpha:
	for l1 in lambd:
		for a2 in alpha:
			for l2 in lambd:
				for g in gamma:
					os.system("./sarsa_1oT_1oSqrtT_bias -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_1oSqrtT_bias.txt")

#print 'SARSA_FIXED_1oSQRTT'
for a1 in alpha:
	for l1 in lambd:
		for a2 in alpha:
			for l2 in lambd:
				for g in gamma:
					os.system("./sarsa_fixed_1oSqrtT_bias -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_fixed_1oSqrtT_bias.txt")

######

#print 'SARSA_FIXED_PESSIMISTIC'
for a in alpha:
	for l in lambd:
		os.system("./sarsa_fixed_bias_start -s 1 -a " + str(a) + " -b 0.01 -o 0.99 -e " + str(l) + " -u 0.9 -c ../conf/rs.cfg -r IdontCare >> results_sarsa_fixed_pessimistic_bias_start.txt")

#print 'SARSA_FIXED_OPTIMISTIC'
for a in alpha:
	for l in lambd:
		os.system("./sarsa_fixed_bias_start -s 1 -a " + str(a) + " -b 0.01 -o 0.99 -e " + str(l) + " -u 0.9 -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_fixed_optimistic_bias_start.txt")


#print 'SARSA_1oT_OPTIMISTIC'
for a in alpha:
	for l in lambd:
		os.system("./sarsa_1oT_bias_start -s 1 -a " + str(a) + " -b 0.01 -o 0.99 -e " + str(l) + " -u 0.9 -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_optimistic_bias_start.txt")


#print 'SARSA_1oT_1oT'
for a1 in alpha:
	for l1 in lambd:
		for a2 in alpha:
			for l2 in lambd:
				for g in gamma:
					os.system("./sarsa_1oT_1oT_bias_start -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_1oT_bias_start.txt")


#print 'SARSA_FIXED_1oT'
for a1 in alpha:
	for l1 in lambd:
		for a2 in alpha:
			for l2 in lambd:
				for g in gamma:
					os.system("./sarsa_fixed_1oT_bias_start -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_fixed_1oT_bias_start.txt")


#print 'SARSA_1oT_FIXED'
for a1 in alpha:
	for l1 in lambd:
		for a2 in alpha:
			for l2 in lambd:
				for g in gamma:
					os.system("./sarsa_1oT_fixed_bias_start -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_fixed_bias_start.txt")


#print 'SARSA_1oT_1oSQRTT'
for a1 in alpha:
	for l1 in lambd:
		for a2 in alpha:
			for l2 in lambd:
				for g in gamma:
					os.system("./sarsa_1oT_1oSqrtT_bias_start -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_1oT_1oSqrtT_bias_start.txt")

#print 'SARSA_FIXED_1oSQRTT'
for a1 in alpha:
	for l1 in lambd:
		for a2 in alpha:
			for l2 in lambd:
				for g in gamma:
					os.system("./sarsa_fixed_1oSqrtT_bias_start -s 1 -a " + str(a1) + " -b " + str(a2) + " -o " + str(g) + " -e " + str(l1) + " -u " + str(l2) + " -c ../conf/rs_opt.cfg -r IdontCare >> results_sarsa_fixed_1oSqrtT_bias_start.txt")