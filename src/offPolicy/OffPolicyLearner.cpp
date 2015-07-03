#include "OffPolicyLearner.hpp"

OffPolicyLearner::OffPolicyLearner(Parameters* param){
	gamma               = param->getGamma();
	epsilon             = param->getEpsilon();
    lambda = param->getLambda();
}
