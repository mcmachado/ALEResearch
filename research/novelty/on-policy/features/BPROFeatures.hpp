/****************************************************************************************
** Implementation of a variation of BASS Features, which has features to encode the 
**  relative position between tiles.
**
** REMARKS: - This implementation is basically Erik Talvitie's implementation, presented
**            in the AAAI'15 LGCVG Workshop.
**
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef BACKGROUND_H
#define BACKGROUND_H
#include "Background.hpp"
#endif

#include <ale_interface.hpp>

using namespace std;

class BPROFeatures{
	private:
		Background *background;
		
		int numBasicFeatures;
    	int numRelativeFeatures;
    	int rowLess0Shift, row0Shift, rowMore0Shift;
        int numColumns, numRows, numColors;
        int getBasicFeaturesIndices(const ALEScreen &screen, int blockWidth, int blockHeight, 
        							vector<vector<vector<int> > > &whichColors, vector<int>& features);
		void addRelativeFeaturesIndices(const ALEScreen &screen, int featureIndex,
									vector<vector<vector<int> > > &whichColors, vector<int>& features);
	public:
		/**
		* Destructor, used to delete the background, which is allocated dynamically.
		*/
		~BPROFeatures();
		/**
 		* TODO: COMMENT
 		* 
 		* @param Parameters *param, which gives access to the number of columns, number of rows,
 		*                   number of colors and the background information
 		* @return nothing, it is a constructor.
 		*/
		BPROFeatures(string gameName);
		/**
 		* This method is the instantiation of the virtual method in the class Features (also check
 		* its documentation). It iterates over all tiles defined by the columns and rows and checks
 		* if each of the colors to be evaluated are present in the tile. 
 		* 
 		* REMARKS: - It is necessary to provide both the screen and the ram because of the superclass,
 		* despite the RAM being useless here. In fact a null pointer works just fine.
 		*          - To avoid return huge vectors, this method is void and the appropriate
 		* vector is returned trough a parameter passed by reference.
 		*          - This method was adapted from Sriram Srinivasan's code
 		* 
 		* @param ALEScreen &screen is the current game screen that one may use to extract features.
 		* @param ALERAM &ram is the current game RAM that one may use to extract features.
 		* @param vector<int>& features an empy vector that will be filled with the requested information,
 		*        therefore it must be passed by reference. It contain the active indices.
 		* @return nothing as one will receive the requested data by the last parameter, by reference.
 		*/
		void getActiveFeaturesIndices(const ALEScreen &screen, const ALERAM &ram, vector<int>& features);	
		/**
 		* Obtain the total number of features that are generated by this feature representation.
 		* Since the constructor demands the number of colors, rows and columns to be set, ideally this
 		* method will always return a correct number. For this representation it is only the product
 		* between these three quantities.
 		* Using Bellemare et. al approach, cited above, the total number of features is 28,672.
 		*
 		* @param none.
 		* @return int number of features generated by this method.
 		*/
		int getNumberOfFeatures();
};
