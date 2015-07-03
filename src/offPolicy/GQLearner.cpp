#include "GQLearner.hpp"


GQLearner::GQLearner(Parameters* param) : OffPolicyLearner(param)
{
}


void GQLearner::receiveSample(const std::vector<int>& features_current_state, float reward, const std::vector<int>& features_next_state)
{
    std::vector<std::pair<int,float> > weighted_features(features_next_state.size());
    //std::transform(features_next_state.begin(),features_next_state.end(),weighted_features.begin(),
    //              [&]

}
