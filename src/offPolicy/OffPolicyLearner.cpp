#include "OffPolicyLearner.hpp"

OffPolicyLearner::OffPolicyLearner(unsigned nF,const std::vector<Action>& actions, Parameters* param){
	gamma               = param->getGamma();
	epsilon             = param->getEpsilon();
    lambda = param->getLambda();
	numActions = actions.size();
    numFeatures = nF;
	traceThreshold = param->getTraceThreshold();
	alpha = param->getAlpha();
    beta = param->getBeta();
    available_actions = actions;
}
