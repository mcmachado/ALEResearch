#include "GQLearner.hpp"


GQLearner::GQLearner(Parameters* param) : OffPolicyLearner(param)
{
}




void GQLearner::receiveSample(const std::vector<int>& features_current_state, int action, float reward, const std::vector<int>& features_next_state, float proba_action_bpolicy)
{
    assert(action>=0 && action<numActions);
    //first, we compute the q values with respect to the next state, and we compute the argmax simultaneously
    std::vector<float> NextQValues(numActions);
    int argmax_nextQ = 0;
    for(int a = 0; a < numActions ; a++){
        NextQValues[a] = std::accumulate(features_next_state.begin(),features_next_state.end(),0.0,
                                         [&a](const float& elem, const int& id){ return elem+weights[a][id];});
        if(NextQValues[a] > NextQValues[argmax_nextQ]){
            argmax_nextQ = a;
        }
    }

    //now we compute the dot product theta*\bar{phi}_{t+1}
    float dotProd = 0.0;
    for(int a = 0; a < numActions ; a++){
        float coeff = (a==argmax_nextQ) ? epsilon + NextQValues[argmax_nextQ] : epsilon;
        dotProd += Mathematics::weighted_sparse_dotprod(weights[a],features_next_state,coeff);
    }

    //compute delta_t
    float delta = reward + gamma*dotProd
        - Mathematics::weighted_sparse_dotprod(weights[action],features_current_state, 1.0);

    //compute rho_t, the ratio of policies
    float rho = (action == argmax_nextQ) ? epsilon + NextQvalues[argmax_nextQ] : epsilon;
    rho /= proba_action_bpolicy;

    //update eligibility traces
    //e <- rho * gamma * lambda * e
    
}
