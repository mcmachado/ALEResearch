#include "GQLearner.hpp"


GQLearner::GQLearner(unsigned numFeatures, const std::vector<Action>& actions, Parameters* param) : OffPolicyLearner(numFeatures, actions, param)
{
    weights.resize(numActions,std::vector<float>(numFeatures,0.0));
    aux_weights.resize(numActions,std::vector<float>(numFeatures,0.0));
    e.resize(numActions);
}



void GQLearner::receiveSample(const std::vector<int>& features_current_state, Action A, float reward, const std::vector<int>& features_next_state, float proba_action_bpolicy)
{
    alpha = 0.005;
    beta = 0;
    gamma = 0.99;
    lambda = 0.4;
    epsilon = 0.05;
    unsigned action = -1;
    for(unsigned i = 0;i<available_actions.size();i++){
        if(available_actions[i] == A){
            action = i; break;
        }
    }
    std::ofstream f("samples.txt",std::ios_base::app);
    f<<features_current_state[0]<<" "<<action<<" "<<reward<<" "<<features_next_state[0]<<" "<<proba_action_bpolicy<<std::endl;
    f.close();
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
    //std::cout<<"currQ :";
    for(const auto & q : currentQValues){
        //std::cout<<q<<" ";
    }
    //std::cout<<std::endl;
    //std::cout<<"NextQ :";
    for(const auto & q : nextQValues){
        //std::cout<<q<<" ";
    }
    //std::cout<<std::endl;
    //std::cout<<"currA "<<argmax_currentQ<<std::endl;
    //std::cout<<"nextA "<<argmax_nextQ<<std::endl;

    //now we compute the dot product theta*\bar{phi}_{t+1}
    float dotProd = 0.0;
    for(unsigned a = 0; a < numActions ; a++){
        float coeff = epsilon/double(numActions) + ((a==argmax_nextQ) ? 1.0 - epsilon : 0);
        dotProd += Mathematics::weighted_sparse_dotprod(weights[a],features_next_state,coeff);
    }
    //std::cout<<"theta * bar_phi "<<dotProd<<std::endl;
    //std::cout<<"theta * phi "<<Mathematics::weighted_sparse_dotprod(weights[action],features_current_state, 1.0)<<std::endl;
    //compute delta_t
    float delta = reward + gamma*dotProd
        - Mathematics::weighted_sparse_dotprod(weights[action],features_current_state, 1.0);

    //std::cout<<"delta "<<delta<<std::endl;
    //compute rho_t, the ratio of policies
    float rho = epsilon/double(numActions) + ((action == argmax_currentQ) ? 1.0 - epsilon : 0);
    rho /= proba_action_bpolicy;
    //std::cout<<"rho "<<rho<<std::endl;

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
        if(e[action].count(feat) == 0){
            e[action][feat] = 0;
        }
        e[action][feat] += 1;
    }

    //std::cout<<"e"<<std::endl;
    for(unsigned i = 0; i<numActions; i++){
        for(const auto& elem : e[i]){
            //std::cout<<"action "<<i<<" id "<<elem.first<<" value "<<elem.second<<std::endl;
        }
    }
    
    //update weights
    //compute dot product (phi_t * w_t)
    float phi_w_dotprod = Mathematics::weighted_sparse_dotprod(aux_weights[action],features_current_state,1.0);
    //std::cout<<"phi*w "<<phi_w_dotprod<<std::endl;
    
    //we do three computations at the same time :
    //theta <- theta + alpha * delta_t * e_t
    //w <- w + beta * delta_t * e_t
    //and compute dot product e_t * w_t
    dotProd = 0.0;
    float c1 = alpha*delta, c2 = beta*delta;
    for(unsigned a = 0; a < numActions; a++){
        for(const auto& it : e[a]){
            dotProd += it.second * aux_weights[a][it.first];
            weights[a][it.first] += c1 * it.second;
            aux_weights[a][it.first] += c2 * it.second;
        }
    }
    //std::cout<<"e*weights "<<dotProd<<std::endl;

    //theta <- theta - alpha * gamma * (1 - lambda) * (e_t * w_t) * \bar{phi}_{t+1}
    float coeff = -1.0 * alpha * gamma * (1.0 - lambda) * dotProd;
    for(unsigned a = 0; a < numActions; a++){
        float policy_coeff = epsilon/double(numActions) + ((a==argmax_nextQ) ? 1.0 - epsilon : 0);
        for(const auto& feat : features_next_state){
            weights[a][feat] += coeff * policy_coeff; 
        }
    }

    //w <- w - beta * (phi_t * w_t) * phi_t
    for(const auto& feat : features_current_state){
        aux_weights[action][feat] -= beta * phi_w_dotprod;
    }
    //std::cout<<"aux_weights"<<std::endl;
    for(unsigned a = 0; a<numActions; a++){
        for(const auto& w : aux_weights[a]){
            //std::cout<<w<<" ";
        }
        //std::cout<<std::endl;
    }
    //std::cout<<"weights"<<std::endl;
    for(unsigned a = 0; a<numActions; a++){
        for(const auto& w : weights[a]){
            //std::cout<<w<<" ";
        }
        //std::cout<<std::endl;
    }
}



void GQLearner::showGreedyPol()
{
    for(int i=0;i<11;i++){
        for(int j=0;j<11;j++){
            int idx = j + i*11;
            int act = 0;
            //std::cout<<weights[0].size()<<std::endl;
            for(unsigned a = 0; a<numActions; a++){
                if(weights[a][idx] > weights[act][idx]){
                    act = a;
                }
            }
            // std::cout<<act;
            switch(available_actions[act]){
            case PLAYER_A_NOOP:
                std::cout<<".\t";
                break;
            case PLAYER_A_LEFT:
                std::cout<<"<\t";
                break;
            case PLAYER_A_RIGHT:
                std::cout<<">\t";
                break;
            case PLAYER_A_UP:
                std::cout<<"^\t";
                break;
            case PLAYER_A_DOWN:
                std::cout<<"|\t";
                break;
            default:
                std::cout<<act<<"\t";
            }
        }
        std::cout<<std::endl;
    }
    for(unsigned a = 0; a<numActions; a++){
        for(const auto& w : weights[a]){
            std::cerr<<w<<" ";
        }
        std::cerr<<std::endl;
    }
    std::cout<<weights[0].size()<<std::endl;
    std::cout<<weights[3][99]<<std::endl;
}
