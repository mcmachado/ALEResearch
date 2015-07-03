/**
 * @file   GQLearner.hpp
 * @author Nicolas Carion
 * @date   Thu Jul  2 19:29:55 2015
 * 
 * @brief  This file implements the GQ(lambda) off-policy learning algorithm
 * 
 * 
 */

#ifndef GQLEARNER_H
#define GQLEARNER_H


class GQLearner : public OffPolicyLearner
{
public:
    GQLearner(Parameters* param);
    virtual void receiveSample(const std::vector<int>& features_current_state, int action, float reward, const std::vector<int>& features_next_state, float proba_action_bpolicy);

};


#endif
