/***************************************************************************************
*** Implementation of a variation of BASS Features, which has features to encode the  **
***  relative position between tiles.                                                 **
***          																		  **
*** REMARKS: - This implementation is based on  Erik Talvitie's work, presented       **
***            in the AAAI'15 LGCVG Workshop.                                         **
***                           														  **
*** Author: Marlos C. Machado / Yitao Liang											  **
****************************************************************************************/

#ifndef BPRO_FEATURES_H
#define BPRO_FEATURES_H

#include <ale_interface.hpp>
#include "Background.hpp"
#include <tuple>

class BPROFeatures{
	private:
		Background *background;
		
		int numBasicFeatures;
    	int numRelativeFeatures;
    	int rowLess0Shift, row0Shift, rowMore0Shift;
        int numColumns, numRows, numColors;
        vector<vector<bool> > bproExistence;
        vector<tuple<int,int> > changed;
    
        int getBasicFeaturesIndices(const ALEScreen &screen, int blockWidth, int blockHeight,
            vector<vector<tuple<int,int> > > &whichColors, vector<int>& features);
		void addRelativeFeaturesIndices(const ALEScreen &screen, int featureIndex,
            vector<vector<tuple<int,int> > > &whichColors, vector<int>& features);
    void resetBproExistence(vector<vector<bool> >& bproExistence, vector<tuple<int, int> >& changed);
	
	public:
		BPROFeatures(std::string gameName);
		~BPROFeatures();
		
		int getNumberOfFeatures();
		void getActiveFeaturesIndices(const ALEScreen &screen, vector<int>& features);	
};

#endif
