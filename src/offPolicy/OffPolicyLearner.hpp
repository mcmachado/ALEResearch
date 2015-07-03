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

    void receiveSample(const std::vector<int>& features_current_state, float reward, const std::vector<int>& features_next_state);
protected:
    float gamma,epsilon,lambda;
};


#endif
