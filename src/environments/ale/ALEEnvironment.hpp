/**
 * @file   ALEEnvironment.hpp
 * @author Nicolas Carion
 * @date   Mon Jun  1 14:22:30 2015
 * 
 * @brief This class is a thin wrapper around the ALE
 * 
 * 
 */


#ifndef ALEENV_H
#define ALEENV_H

#include<type_traits>
#include <vector>
#include <ale_interface.hpp>
#include "../../common/Traits.hpp"
#include "../Environment.hpp"
//here we do a little bit of template black magic:
//two cases may arise either the features can be binary, or they can be general (doubles,...)
//There are two possible interfaces for the active features: either we return only the indices of the active features, or we return the indices and the corresponding feature value.
//For speed purposes, the features computer working with binary features are required to provide the first interface.
//We then check if they provide also the second interface, otherwise, we make up one.

//we call the macro to create the member detector
CREATE_MEMBER_DETECTOR(getActiveFeaturesIndices)

//this the case where the features are not binary (hence have to provide the second interface)
template < typename FeatureComputer, typename = void>
class impl_ALEEnvironment : public t_Environment<FeatureComputer>
{
public:
    typedef typename FeatureComputer::FeatureType FeatureType;
    impl_ALEEnvironment(ALEInterface* ale,FeatureComputer* feat) : t_Environment<FeatureComputer>(feat),m_ale(ale){}

    virtual void getActiveFeaturesIndices(std::vector<std::pair<int,FeatureType>>& active_feat)
    {
        this->m_feat->getActiveFeatureIndices(active_feat,m_ale);
    }

    //this function makes no sense in that case, so we just return an empty vector
    virtual void getActiveFeaturesIndices(std::vector<int >& active_feat)
    {
        active_feat.clear();
    }

    ALEInterface* m_ale;
};


//this is the case where features are boolean, but the class seemingly doesn't provide a fast interface
template<typename FeatureComputer>
class impl_ALEEnvironment<FeatureComputer,typename std::enable_if<std::is_same<typename FeatureComputer::FeatureType,bool>::value>::type> : public t_Environment<FeatureComputer>
{
public:
    typedef typename FeatureComputer::FeatureType FeatureType;
    impl_ALEEnvironment(ALEInterface* ale,FeatureComputer* feat) : t_Environment<FeatureComputer>(feat),m_ale(ale){}

    //in this case, we have to make up this interface, since it is not provided
    virtual void getActiveFeaturesIndices(std::vector<std::pair<int,FeatureType>>& active_feat)
    {
        std::vector<int> temp;
        getActiveFeaturesIndices(temp);
        active_feat = std::vector<std::pair<int,bool> >(temp.size());
        for(unsigned i=0;i<temp.size();i++){
            active_feat[i] = {temp[i], true};
        }
       
    }

    virtual void getActiveFeaturesIndices(std::vector<int >& active_feat)
    {
        this->m_feat->getActiveFeaturesIndices(m_ale->getScreen(),m_ale->getRAM(),active_feat);
    }
protected:
    ALEInterface* m_ale;
};


//This is the last case, where the feature are boolean and an interface is provided.
template<typename FeatureComputer>
class impl_ALEEnvironment<FeatureComputer,typename std::enable_if<
                                              function_traits::_implem::static_and<
                                                  function_traits::has_getActiveFeaturesIndices<FeatureComputer,void(std::vector<std::pair<int,typename FeatureComputer::FeatureType>>&,ALEInterface*)>::value,
                                                  std::is_same<typename FeatureComputer::FeatureType,bool>::value
                                                  >::value>::type> : public t_Environment<FeatureComputer>
{
public:
    typedef typename FeatureComputer::FeatureType FeatureType;
    impl_ALEEnvironment(ALEInterface* ale,FeatureComputer* feat) : t_Environment<FeatureComputer>(feat),m_ale(ale){}

    virtual void getActiveFeatureIndices(std::vector<std::pair<int,FeatureType>>& active_feat)
    {
        this->m_feat->getActiveFeatureIndices(active_feat,m_ale);
    }
    virtual void getActiveFeaturesIndices(std::vector<int >& active_feat)
    {
        this->m_feat->getActiveFeaturesIndices(m_ale->getScreen(),m_ale->getRAM(),active_feat);
    }
protected:
    ALEInterface* m_ale;
};


template<typename FeatureComputer>
class ALEEnvironment : public impl_ALEEnvironment<FeatureComputer>{

public:
    typedef typename FeatureComputer::FeatureType FeatureType;
    ALEEnvironment(ALEInterface* ale,FeatureComputer* feat) : impl_ALEEnvironment<FeatureComputer>(ale,feat){}

    

    /**@brief This function puts back the environment in its original state 
     * 
     */
    virtual void reset(){
        this->m_ale->reset_game();
    }
    
    /** @brief This function tells whether the environment has reached a terminal state
     * 
     * @return a boolean
     */
    virtual bool isTerminal(){
        return this->m_ale->game_over();
    }
    
    /** @brief This function is used to simulate one step in the environment 
     * 
     * 
     * @param action an integer describing the action taken by the agent
     * @param score a return parameter corresponding to the raw score obtained in the environment
     * @param reward a return parameter which tells the corresponding reward.
     */
    virtual double act(Action action){
        std::vector<double> r;
        return this->m_ale->act(action);
    }

    
    /** @brief Return the set of actions that can be taken in this environment
     * 
     * @return a vector of integers representing the actions
     */
    virtual std::vector<Action> getLegalActionSet()
    {
        return this->m_ale->getLegalActionSet();
    }

    
    /** @brief Return the minimal set of actions that can be taken in this environment
     * In some cases, some actions are legal but not usefull. This actions are not returned by this function
     * @return a vector of integers representing the actions
     */
    virtual std::vector<Action> getMinimalActionSet()
    {
        return this->m_ale->getMinimalActionSet();
    }

    virtual int getEpisodeFrameNumber()
    {
        return this->m_ale->getEpisodeFrameNumber();
    }

    virtual void getCompleteFeatureVector(std::vector<FeatureType>& features){
        this->m_feat->getCompleteFeatureVector(this->m_ale->getScreen(),this->m_ale->getRAM(),features);
    }



    virtual int getNumberOfFeatures()
    {
        return this->m_feat->getNumberOfFeatures();
    }
protected:
};


#endif
