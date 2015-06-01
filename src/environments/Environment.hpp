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
template<typename FeatureComputer>
class Environment{

public:
    Environment(FeatureComputer* feat) : m_feat(feat){}

    

    /**@brief This function puts back the environment in its original state 
     * 
     */
    virtual void reset()=0;
    
    /** @brief This function tells whether the environment has reached a terminal state
     * 
     * @return a boolean
     */
    virtual bool isTerminal() = 0;
    
    /** @brief This function is used to simulate one step in the environment 
     * 
     * 
     * @param action an integer describing the action taken by the agent
     * @param score a return parameter corresponding to the raw score obtained in the environment
     * @param reward a return parameter which tells the corresponding reward.
     */
    virtual void act(int action, double& score, double& reward) = 0;

    /** @brief Return all the features, as computed by the featurecomputer
     * 
     * @param features a return parameter containing the features
     */
    virtual void getFeatures(std::vector<typename FeatureComputer::FeatureType>& features){
        m_feat->getFeatures(features,this);
    }

    /** @brief Return the indices of the non-zero features, allong with their values
     * 
     * @param active_feat a return parameter containing the active features
     */
    virtual void getActiveFeatures(std::vector<std::pair<unsigned,typename FeatureComputer::FeatureType> >& active_feat)
    {
        m_feat->getActiveFeatures(active_feat,this);
    }
protected:
    FeatureComputer* m_feat;
    
};


#endif
