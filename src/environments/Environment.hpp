/**
 * @file   Environment.hpp
 * @author Nicolas Carion
 * @date   Mon Jun  1 11:54:58 2015
 *
 * @brief  This abstract class define a generic interface to what an environment should be
 *
 *
 */

#ifndef ENV_H
#define ENV_H

#include <vector>
#include "../offPolicy/OffPolicyLearner.hpp"

template<typename FeatureType>
class Environment{

public:

    Environment() {m_offPolicyLearner = nullptr;}

    /**@brief This function puts back the environment in its original state
     */
    virtual void reset() = 0;
    /**@brief Same as reset ; provided for compatibility 
     */
    void reset_game(){reset();}
    
    /** @brief This function tells whether the environment has reached a terminal state
     * @return a boolean
     */
    virtual bool isTerminal() = 0;
    
    /**@brief Same as isTerminal, provided for compatibility 
     * @return a boolean
     */
    bool game_over(){return isTerminal();}
    
    /** @brief This function is used to simulate one step in the environment
     * This function should not be overloaded by derived classes. Overload doAct instead.
     *
     * @param action an integer describing the action taken by the agent
     * @param probaAction the probability that the agent took this action. The default, 1.0, corresponds to a deterministic agent
     * @return the reward obtained by triggering the action
     */
    virtual double act(Action action, double probaAction = 1.0) = 0;

    /**  @brief This function is used to simulate one step in the environment
     * This is the one that must be overloaded in derived classes.
     * @param action an integer describing the action taken by the agent
     * @return the reward obtained by triggering the action
     */
    virtual double doAct(Action action) = 0;


    /** @brief Return the set of actions that can be taken in this environment
     *
     * @return a vector of integers representing the actions
     */
    virtual std::vector<Action> getLegalActionSet(){
        return getLegalActionSet();
    }


    /** @brief Return the minimal set of actions that can be taken in this environment
     * In some cases, some actions are legal but not usefull. This actions are not returned by this function
     * @return a vector of integers representing the actions
     */
    virtual std::vector<Action> getMinimalActionSet(){
        return getMinimalActionSet();
    }

    /** @brief Return all the features, as computed by the featurecomputer
     *
     * @param features a return parameter containing the features
     */
    virtual void getCompleteFeatureVector(std::vector<FeatureType>& features) = 0;

    /** @brief Return the indices of the non-zero features (usefull only if the features are boolean)
     *
     * @param active_feat a return parameter containing the active features
     */    
    virtual void getActiveFeaturesIndices(std::vector<int>& active_feat) = 0;

    /** 
     * @return an integer representing the number of unique features 
     */
    virtual int getNumberOfFeatures() = 0;
    
    /**      
     * @return the number of the current frame in the episode
     */
    virtual int getEpisodeFrameNumber() = 0;

    void setOffPolicyLearner(std::shared_ptr<OffPolicyLearner> l){
        m_offPolicyLearner = l;
    }


    virtual void setFlavor(unsigned f){};
protected:
    std::shared_ptr<OffPolicyLearner> m_offPolicyLearner;

};

template<typename FeatureComputer>
class t_Environment : public Environment<typename FeatureComputer::FeatureType>{
public:
    t_Environment(FeatureComputer* feat) : m_feat(feat) {}

    virtual double act(Action action, double prob_action = 1.0) final
    {
        if(this->m_offPolicyLearner == nullptr){
            return this->doAct(action);
        }
        std::vector<int> curState,nextState;
        this->getActiveFeaturesIndices(curState);
        double reward = this->doAct(action);
        this->getActiveFeaturesIndices(nextState);
        this->m_offPolicyLearner->receiveSample(curState, action, reward, nextState, prob_action);
        return reward;
    }
protected:
    FeatureComputer* m_feat;

};
#endif
