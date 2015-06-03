/****************************************************************************************
** Implementation of BASS Features, described in details in the paper below. 
**       "The Arcade Learning Environment: An Evaluation Platform for General Agents.
**        Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling.
**        Journal of Artificial Intelligence Research, 47:253–279, 2013."
**
** The idea is to divide the screen in tiles and to answer, for each tile, if one of the
** n colors defined are present in that tile. Additionally we then evaluate whether there
** are pairwise combinations of these features.
**
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef BASSFEAT_H
#define BASSFEAT_H
#include "../common/Parameters.hpp"
#include "Features.hpp"


class BASSFeatures : public Features::Features{
	private:
		Parameters *param;
		int numPureFeatures;
    	int numPairwiseFeatures;
		/**
		* Once we receive the active features from Basic, this method adds to features the
		* indices that are activated by the interaction between two Basic features active.
		*
		* @param vector<int>& features vector containing the indices of active Basic features
		* @param int offset maximum possible number of Basic features, to shift when defining
		*            the indices of the features active due to interaction.
		*
		* @return nothing since the result is stored in the features vector
		*/
		void addPairwiseFeatures(vector<int>& features, int offset);

	public:
    typedef bool FeatureType;
    /**
		* Destructor, not necessary in this class.
		*/
		~BASSFeatures();
		/**
 		* Constructor. Since every operation in this class has to be done knowing the number of
 		* columns, rows and colors to generate the feature vector, this information is given in
 		* param, as any other relevant information such as background.
 		*
 		* @param Parameters *param, which gives access to the number of columns, number of rows,
 		*                   number of colors and the background information
 		* @return nothing, it is a constructor.
 		*/
		BASSFeatures(Parameters *param);
		/**
 		* This method is the instantiation of the virtual method in the class Features (also check
 		* its documentation). It iterates over all tiles defined by the columns and rows and checks
 		* if each of the colors to be evaluated are present in the tile. Then it iterates over all
 		* features to check if they interact with each other (BASS extension).
 		* 
 		* REMARKS: - It is necessary to provide both the screen and the ram because of the superclass,
 		* despite the RAM being useless here. In fact a null pointer works just fine.
 		*          - To avoid return huge vectors, this method is void and the appropriate
 		* vector is returned trough a parameter passed by reference.
 		*          - This method was adapted from Sriram Srinivasan's code
 		* 
 		* @param ALEScreen &screen is the current game screen that one may use to extract features.
 		* @param ALERAM &ram is the current game RAM that one may use to extract features.
 		* @param vector<bool>& features an empy vector that will be filled with the requested information,
 		*        therefore it must be passed by reference. Its i-th position is TRUE if the i-th feature is active.
 		* @return nothing as one will receive the requested data by the last parameter, by reference.
 		*/
		void getActiveFeaturesIndices(const ALEScreen &screen, const ALERAM &ram, vector<int>& features);	
		/**
 		* Obtain the total number of features that are generated by this feature representation.
 		* Since the constructor demands the number of colors, rows and columns to be set, ideally this
 		* method will always return a correct number. For this representation it is only the product
 		* between these three quantities.
 		* Using Bellemare et. al approach, cited above, the total number of features is ????.
 		*
 		* @param none.
 		* @return int number of features generated by this method.
 		*/
		int getNumberOfFeatures();
};


#endif
