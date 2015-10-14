template<typename FeatureType>
RLLearner<FeatureType>::RLLearner(Environment<FeatureType>& env, Parameters *param, int seed){
	randomActionTaken   = 0;

	gamma               = param->getGamma();
	epsilon             = param->getEpsilon();
	toUseOnlyRewardSign = param->getUseRewardSign();
	toBeOptimistic      = param->getOptimisticInitialization();
	degreeOfOptimism    = 0.0;
	
	episodeLength       = param->getEpisodeLength();
	numEpisodesEval     = param->getNumEpisodesEval();
	totalNumberOfFramesToLearn = param->getLearningLength();

	//Get the number of effective actions:
	if(param->isMinimalAction()){
		actions = env.getMinimalActionSet();
	}
	else{
		actions = env.getLegalActionSet();
	}
	numActions = actions.size();
    agentRand.seed(seed);
}

template<typename FeatureType>
int RLLearner<FeatureType>::epsilonGreedy(std::vector<float> &QValues){
	randomActionTaken = 0;

	int action = Mathematics::argmax(QValues);
	//With probability epsilon: a <- random action in A(s)
	long random = agentRand();
	if(float(random % 1000) < float(epsilon*1000)) {
		randomActionTaken = 1;
		action = agentRand() % numActions;
	}
	return action;
}

template<typename FeatureType>
void RLLearner<FeatureType>::updateQValues(std::vector<int> &Features, std::vector<std::vector<float> > &w, std::vector<float> &QValues){
	for(int a = 0; a < numActions; a++){
		float sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[a][Features[i]];
		}
		QValues[a] = sumW;
	}

	if(toBeOptimistic){
		float optimism = degreeOfOptimism / (1.0 - gamma);
		for(int a = 0; a < numActions; a++){
			QValues[a] += optimism;
		}
	}
}

/**
 * The first parameter is the one that is used by Sarsa. The second is used to
 * pass aditional information to the running algorithm (like 'real score' if one
 * is using a surrogate reward function).
 */
template<typename FeatureType>
void RLLearner<FeatureType>::act(Environment<FeatureType>& env, int action, std::vector<float> &reward){
	float r_alg = 0.0, r_real = 0.0;

    //compute probability of taking current action
    double prob_action = epsilon/double(numActions) + (randomActionTaken ? 0 : 1.0 - epsilon);

	r_real = env.act(actions[action], prob_action);
	if(toUseOnlyRewardSign){
		if(r_real > 0){ 
			r_alg = 1.0;
		}
		else if(r_real < 0){
			r_alg = -1.0;
		}
	//Normalizing reward according to the first
	//reward, Marc did this in his JAIR paper:
	} else{
		if(r_real != 0.0){
			if(!sawFirstReward){
				firstReward = std::abs(r_real);
				sawFirstReward = 1;
			}
		}
		if(sawFirstReward){
			r_alg = r_real/firstReward;	
		}
	}
	reward[0] = r_alg;
	reward[1] = r_real;
}
