#include "GQLearner.hpp"


GQLearner::GQLearner(unsigned numFeatures, unsigned numActions, Parameters* param) : OffPolicyLearner(numFeatures, numActions, param)
{
    weights.resize(numActions,std::vector<float>(numFeatures,0.0));
    aux_weights.resize(numActions,std::vector<float>(numFeatures,0.0));
    e.resize(numActions);
}




void GQLearner::receiveSample(const std::vector<int>& features_current_state, unsigned action, float reward, const std::vector<int>& features_next_state, float proba_action_bpolicy)
{
    assert(action>=0 && action<numActions);
    //first, we compute the q values with respect to the current state and the next state, and we compute the argmax simultaneously
    std::vector<float> nextQValues(numActions),currentQValues(numActions);
    unsigned argmax_nextQ = 0, argmax_currentQ = 0;
    for(unsigned a = 0; a < numActions ; a++){
        nextQValues[a] = std::accumulate(features_next_state.begin(),features_next_state.end(),0.0,
                                         [&a,this](const float& elem, const int& id){ return elem+weights[a][id];});
        currentQValues[a] = std::accumulate(features_current_state.begin(),features_current_state.end(),0.0,
                                            [&a,this](const float& elem, const int& id){ return elem+weights[a][id];});
        if(nextQValues[a] > nextQValues[argmax_nextQ]){
            argmax_nextQ = a;
        }
        if(currentQValues[a] > currentQValues[argmax_currentQ]){
            argmax_currentQ = a;
        }
    }

    //now we compute the dot product theta*\bar{phi}_{t+1}
    float dotProd = 0.0;
    for(unsigned a = 0; a < numActions ; a++){
        float coeff = epsilon/double(numActions) + ((a==argmax_nextQ) ? 1.0 - epsilon : 0);
        dotProd += Mathematics::weighted_sparse_dotprod(weights[a],features_next_state,coeff);
    }

    //compute delta_t
    float delta = reward + gamma*dotProd
        - Mathematics::weighted_sparse_dotprod(weights[action],features_current_state, 1.0);

    //compute rho_t, the ratio of policies
    float rho = epsilon/double(numActions) + ((action == argmax_currentQ) ? 1.0 - epsilon : 0);
    rho /= proba_action_bpolicy;

    //update eligibility traces
    //e <- rho * gamma * lambda * e
    for(unsigned a = 0; a < numActions; a++){
        for (auto it = e[a].begin(); it != e[a].end() /* not hoisted */; /* no increment */){
            //here it is an iterator on the map. it.first hold the index of the value, and it.second, the value itself
            (*it).second = rho * gamma * lambda * (*it).second;
            if ((*it).second < traceThreshold)
                e[a].erase(it++);
            else
                ++it;
		}
	}

    //e <- e + phi
    for(const auto& feat : features_current_state){
        e[action][feat] += 1;
    }

    //update weights
    //compute dot product (phi_t * w_t)
    float phi_w_dotprod = Mathematics::weighted_sparse_dotprod(aux_weights[action],features_current_state,1.0);
    
    //we do three computations at the same time :
    //theta <- theta + alpha * delta_t * e_t
    //w <- w + beta * delta_t * e_t
    //and compute dot product e_t * w_t
    dotProd = 0.0;
    float c1 = alpha*delta, c2 = beta*delta;
    for(unsigned a = 0; a < numActions; a++){
        for(const auto& it : e[a]){
            dotProd = it.second * aux_weights[a][it.first];
            weights[a][it.first] += c1 * it.second;
            aux_weights[a][it.first] += c2 * it.second;
        }
    }

    //theta <- theta - alpha * gamma * (1 - delta) * (e_t * w_t) * \bar{phi}_{t+1}
    float coeff = -1.0 * alpha * gamma * (1.0 - delta) * dotProd;
    for(unsigned a = 0; a < numActions; a++){
        float policy_coeff = epsilon/double(numActions) + ((a==argmax_nextQ) ? 1.0 - epsilon : 0);
        for(const auto& feat : features_current_state){
            weights[a][feat] += coeff * policy_coeff; 
        }
    }

    //w <- w - beta * (phi_t * w_t) * phi_t
    for(const auto& feat : features_current_state){
        aux_weights[action][feat] -= beta * phi_w_dotprod;
    }
}
