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
#include <vector>
#include <unordered_map>
#include <numeric>
#include "OffPolicyLearner.hpp"
#include "../common/Parameters.hpp"
#include "../common/Mathematics.hpp"
#include <cassert>
#include <iostream>
class GQLearner : public OffPolicyLearner
{
public:
    GQLearner(unsigned numFeatures,const std::vector<Action>& actions, Parameters* param);
    virtual void receiveSample(const std::vector<int>& features_current_state, Action action, float reward, const std::vector<int>& features_next_state, float proba_action_bpolicy);
    void showGreedyPol();
public:
    std::vector<std::vector<float> > weights, aux_weights;
    std::vector<std::unordered_map<unsigned,float> > e;
};


#endif
