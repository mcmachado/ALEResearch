template<typename FeatureType>
RLLearner<FeatureType>::RLLearner(Environment<FeatureType>& env, Parameters *param, int seed){
	randomActionTaken   = 0;

	gamma               = param->getGamma();
	epsilon             = param->getEpsilon();
	toUseOnlyRewardSign = param->getUseRewardSign();
	toBeOptimistic      = param->getOptimisticInitialization();
	
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
			if(toBeOptimistic){
				r_alg = (r_real - firstReward)/firstReward + gamma;
			}
			else{
				r_alg = r_real/firstReward;	
			}
		}
		else{
			if(toBeOptimistic){
				r_alg = gamma - 1.0;
			}
		}
	}
	reward[0] = r_alg;
	reward[1] = r_real;
	//If doing optimistic initialization, to avoid the agent
	//to "die" soon to avoid -1 as reward at each step, when
	//the agent dies we give him -1 for each time step remaining,
	//this would be the worst case ever...
	if(env.isTerminal() && toBeOptimistic){
		int missedSteps = episodeLength - env.getEpisodeFrameNumber() + 1;
		double penalty = pow(gamma, missedSteps) - 1;
		reward[0] -= penalty;
	}
}
