#include "OffPolicyLearner.hpp"

OffPolicyLearner::OffPolicyLearner(unsigned nF, unsigned nA, Parameters* param){
	gamma               = param->getGamma();
	epsilon             = param->getEpsilon();
    lambda = param->getLambda();
	numActions = nA;
    numFeatures = nF;
	traceThreshold = param->getTraceThreshold();
	alpha = param->getAlpha();
    beta = param->getBeta();
}
