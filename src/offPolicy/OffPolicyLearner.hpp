/**
 * @file   OffPolicyLearner.hpp
 * @author Nicolas Carion
 * @date   Thu Jul  2 19:31:14 2015
 * 
 * @brief  This file describes the general interface of an off-policy learning algorithm
 * 
 * 
 */
#ifndef OFFLEARN_H
#define OFFLEARN_H

class OffPolicyLearner
{
public:
    OffPolicyLearner(Parameters *param);

    /** 
     * 
     * 
     * @param features_current_state indices of active features in current state
     * @param action id of action taken by behavior policy
     * @param reward reward observed
     * @param features_next_state indiced of active features in next state
     * @param proba_action_bpolicy pi_b(a_t | s_t) = proba of taking current action in current state (in the behavior policy. Set to one if the behavior policy is deterministic.
     */
    void receiveSample(const std::vector<int>& features_current_state, int action, float reward, const std::vector<int>& features_next_state, float proba_action_bpolicy) = 0;
protected:
    float gamma,epsilon,lambda;
};


#endif
